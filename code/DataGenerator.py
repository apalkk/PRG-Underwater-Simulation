import os
import cv2
import numpy as np
import tensorflow as tf

import imgaug as ia
import imgaug.augmenters as iaa

"""
## Building a data pipeline

1. The pipeline takes a dataframe containing the path for the RGB images,
as well as the depth and depth mask files.
2. It reads and resize the RGB images.
3. It reads the depth and depth mask files, process them to generate the depth map image and
resize it.
4. It returns the RGB images and the depth map images for a batch.
"""


class Generator(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size=6, dim=(256, 256), n_channels=3, shuffle=True):
        """
        Initialization
        """
        self.data = data
        self.indices = self.data.index.tolist()
        self.dim = dim
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.min_depth = 0.1
        self.on_epoch_end()

        self.sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        self.seq = iaa.Sequential([

                                    iaa.Fliplr(0.5),
                                    iaa.Flipud(0.3),

                                    iaa.SomeOf((0, 3), [
                                                        self.sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                                                        iaa.OneOf([
                                                            iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                                                            iaa.AverageBlur(k=(2, 7)),  # blur image using local means with kernel sizes between 2 and 7
                                                            iaa.MedianBlur(k=(3, 11)),  # blur image using local medians with kernel sizes between 2 and 7
                                                        ]),
                                                        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                                                        iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                                                        # search either for all edges or for directed edges,
                                                        # blend the result with the original image using a blobby mask
                                                        # iaa.SimplexNoiseAlpha(iaa.OneOf([
                                                        #     iaa.EdgeDetect(alpha=(0.5, 1.0)),
                                                        #     iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                                                        # ])),
                                                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),  # add gaussian noise to images
                                                        iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                                                        # iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                                                        # either change the brightness of the whole image (sometimes
                                                        # per channel) or change the brightness of subareas
                                                        # iaa.OneOf([
                                                        #     iaa.Multiply((0.5, 1.5), per_channel=0.5),
                                                        #     iaa.FrequencyNoiseAlpha(
                                                        #         exponent=(-4, 0),
                                                        #         first=iaa.Multiply((0.5, 1.5), per_channel=True),
                                                        #         second=iaa.LinearContrast((0.5, 2.0))
                                                        #     )
                                                        # ]),
                                                        iaa.LinearContrast((0.5, 1.5), per_channel=0.5),  # improve or worsen the contrast
                                    ], random_order=True)
                                ], random_order=True)

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        if (index + 1) * self.batch_size > len(self.indices):
            self.batch_size = len(self.indices) - index * self.batch_size
        # Generate one batch of data
        # Generate indices of the batch
        index = self.indices[index * self.batch_size: (index + 1) * self.batch_size]
        # Find list of IDs
        batch = [self.indices[k] for k in index]
        x, y = self.data_generation(batch)

        return x, y

    def on_epoch_end(self):

        """
        Updates indexes after each epoch
        """
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def augment_data(self, img1, img2):
        bgr_img = img1.copy()
        depth_img = img2.copy()

        depth = np.true_divide(depth_img, 255)

        img_batch = np.reshape(bgr_img, (1, *bgr_img.shape))
        depth_batch = np.reshape(depth, (1, depth.shape[0], depth.shape[1], 1)).astype(np.float32)
        im_g, d_g = self.seq(images=img_batch, heatmaps=depth_batch)

        bgr_aug = np.squeeze(im_g)
        depth_aug = np.squeeze(d_g)

        depth_aug *= 255.0
        depth_aug = depth_aug.astype(np.uint8)

        return bgr_aug, depth_aug

    def load(self, image_path, depth_path):
        """Load input and target image."""

        image_ = cv2.imread(image_path)
        image_ = cv2.resize(image_, (self.dim[1], self.dim[0]))


        depth_map = cv2.imread(depth_path, 0)
        depth_map = cv2.resize(depth_map, (self.dim[1], self.dim[0]))

        # apply same random augmentation to image and depth map
        image_, depth_map = self.augment_data(image_, depth_map)

        # add 1 dimension to depth map
        depth_map = np.reshape(depth_map, (*self.dim, 1))

        # conver to tensorflow data type
        image_ = tf.image.convert_image_dtype(image_, tf.float32)
        depth_map = tf.image.convert_image_dtype(depth_map, tf.float32)

        return image_, depth_map

    def data_generation(self, batch):

        x = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, 1))

        for i, batch_id in enumerate(batch):
            x[i, ], y[i, ] = self.load(self.data["image"][batch_id], self.data["depth"][batch_id])

        return x, y


def read_data(rgb_images_path, depth_images_path):
    rgb_names = os.listdir(rgb_images_path)
    rgb_paths = []
    depth_paths = []
    for name in rgb_names:
        rgb_paths.append(os.path.join(rgb_images_path, name))
        depth_paths.append(os.path.join(depth_images_path, name))

    return rgb_paths, depth_paths


def load_data(data_folder):
    rgb_images_path = os.path.join(data_folder, "RGB_imgs")
    depth_images_path = os.path.join(data_folder, "DEPTH_imgs")
    rgb_images, depth_images = read_data(rgb_images_path, depth_images_path)

    return rgb_images, depth_images


if __name__ == "__main__":
    import Config as cf
    import pandas as pd

    rgb_images, depth_images = load_data(cf.DATA_DIR)

    data = {"image": rgb_images, "depth": depth_images}
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42)

    train_samples = int(df["image"].size * cf.TRAIN_PERCENT)
    train_loader = Generator(data=df[:train_samples].reset_index(drop="true"), batch_size=cf.BATCH_SIZE,
                                dim=(cf.HEIGHT, cf.WIDTH))

    for idx, (x, y) in enumerate(train_loader):
        print(x.shape)
        print(y.shape)
        exit()
