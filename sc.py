import sys # to access the system
import cv2
import os

import os

def get_latest_image(dirpath, valid_extensions=('jpg','jpeg','png')):
    """
    Get the latest image file in the given directory
    """

    # get filepaths of all files and dirs in the given dir
    valid_files = [os.path.join(dirpath, filename) for filename in os.listdir(dirpath)]
    # filter out directories, no-extension, and wrong extension files
    valid_files = [f for f in valid_files if '.' in f and \
        f.rsplit('.',1)[-1] in valid_extensions and os.path.isfile(f)]

    if not valid_files:
        raise ValueError("No valid images in %s" % dirpath)

    return max(valid_files, key=os.path.getmtime) 

path = '/Users/aadipalnitkar/Underwater-share/data/final_output_3_10_23_test/render_output/third_cam'

# get the latest image file path
latest_image_path = get_latest_image(path,'png')

# read the latest image
img = cv2.imread(latest_image_path)

cv2.imshow("BlueRov", img)
while True:
    # wait for 1 second
    key = cv2.waitKey(1000)

    # check if any new images have been added to the folder
    new_latest_image_path = get_latest_image(path,'png')
    if new_latest_image_path != latest_image_path:
        # if a new image is found, update the latest image and display it
        latest_image_path = new_latest_image_path
        img = cv2.imread(latest_image_path)
        cv2.imshow("Latest Image", img)

    # check if the user has pressed the 'q' key to quit
    if key == ord('q'):
        break
cv2.destroyAllWindows() # destroy all windows
