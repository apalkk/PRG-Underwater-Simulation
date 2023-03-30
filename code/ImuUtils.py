"""
Code stripped and modified from 
https://github.com/Aceinna/gnss-ins-sim
"""

import numpy as np
import math


# global
VERSION = '1.0'
D2R = math.pi/180


# low accuracy, from AHRS380
#http://www.memsic.cn/userfiles/files/Datasheets/Inertial-System-Datasheets/AHRS380SA_Datasheet.pdf
gyro_low_accuracy = {'b': np.array([0.0, 0.0, 0.0]) * D2R,
                     'b_drift': np.array([10.0, 10.0, 10.0]) * D2R/3600.0,
                     'b_corr':np.array([100.0, 100.0, 100.0]),
                     'arw': np.array([0.75, 0.75, 0.75]) * D2R/60.0}
accel_low_accuracy = {'b': np.array([0.0e-3, 0.0e-3, 0.0e-3]),
                      'b_drift': np.array([2.0e-4, 2.0e-4, 2.0e-4]),
                      'b_corr': np.array([100.0, 100.0, 100.0]),
                      'vrw': np.array([0.05, 0.05, 0.05]) / 60.0}

# mid accuracy, partly from IMU381
gyro_mid_accuracy = {'b': np.array([0.0, 0.0, 0.0]) * D2R,
                     'b_drift': np.array([3.5, 3.5, 3.5]) * D2R/3600.0,
                     'b_corr': np.array([100.0, 100.0, 100.0]),
                     'arw': np.array([0.25, 0.25, 0.25]) * D2R/60}
accel_mid_accuracy = {'b': np.array([0.0e-3, 0.0e-3, 0.0e-3]),
                      'b_drift': np.array([5.0e-5, 5.0e-5, 5.0e-5]),
                      'b_corr': np.array([100.0, 100.0, 100.0]),
                      'vrw': np.array([0.03, 0.03, 0.03]) / 60}

# high accuracy, partly from HG9900, partly from
# http://www.dtic.mil/get-tr-doc/pdf?AD=ADA581016
gyro_high_accuracy = {'b': np.array([0.0, 0.0, 0.0]) * D2R,
                      'b_drift': np.array([0.1, 0.1, 0.1]) * D2R/3600.0,
                      'b_corr':np.array([100.0, 100.0, 100.0]),
                      'arw': np.array([2.0e-3, 2.0e-3, 2.0e-3]) * D2R/60}
accel_high_accuracy = {'b': np.array([0.0e-3, 0.0e-3, 0.0e-3]),
                       'b_drift': np.array([3.6e-6, 3.6e-6, 3.6e-6]),
                       'b_corr': np.array([100.0, 100.0, 100.0]),
                       'vrw': np.array([2.5e-5, 2.5e-5, 2.5e-5]) / 60}


def vib_from_env(env, fs):
    '''
    Args:
        env: vibration model
        fs: sample frequency
    '''
    vib_def = {}
    if env is None:
        vib_def = None
    elif isinstance(env, str):       # specify simple vib model
        env = env.lower()
        if 'random' in env:         # normal distribution
            vib_def['type'] = 'random'
            env = env.replace('-random', '')
        elif 'sinusoidal' in env:   # sinusoidal vibration
            vib_def['type'] = 'sinusoidal'
            env = env.replace('-sinusoidal', '')
            if env[-2:] == 'hz':
                try:
                    idx_first_mark = env.find('-')
                    vib_def['freq'] = math.fabs(float(env[idx_first_mark+1:-2]))
                    env = env[:idx_first_mark]
                except:
                    raise ValueError('env = \'%s\' is not valid (invalid vib freq).'% env)
            else:
                raise ValueError('env = \'%s\' is not valid (No vib freq).'% env)
        else:
            raise ValueError('env = \'%s\' is not valid.'% env)
        vib_amp = 1.0   # vibration amplitude, 1sigma for random, peak value for sinusoidal
        if env[-1] == 'g' or env[-1] == 'G':    # acc vib in unit of g
            vib_amp = 9.8
            env = env[:-1]  # remove 'g' or 'G'
        elif env[-1] == 'd' or env[-1] == 'D':  # gyro vib in unit of deg/s
            vib_amp = D2R
            env = env[:-1]  # remove 'd' or 'D'
        try:
            env = env[1:-1] # remove '[]' or '()'
            env = env.split(' ')
            vib_amp *= np.array(env, dtype='float64')
            vib_def['x'] = vib_amp[0]
            vib_def['y'] = vib_amp[1]
            vib_def['z'] = vib_amp[2]
        except Exception as e:
            raise ValueError('Cannot convert \'%s\' to float'% env)
    elif isinstance(env, np.ndarray):           # customize the vib model with PSD
        if env.ndim == 2 and env.shape[1] == 4: # env is a np.array of size (n,4)
            vib_def['type'] = 'psd'
            n = env.shape[0]
            half_fs = 0.5*fs
            if env[-1, 0] > half_fs:
                n = np.where(env[:, 0] > half_fs)[0][0]
            vib_def['freq'] = env[:n, 0]
            vib_def['x'] = env[:n, 1]
            vib_def['y'] = env[:n, 2]
            vib_def['z'] = env[:n, 3]
        else:
            raise TypeError('env should be of size (n,2)')
    else:
        raise TypeError('env should be a string or a numpy array of size (n,2)')
    return vib_def


def time_series_from_psd(sxx, freq, fs, n):
    """
    Generate 1-D time series from a given 1-D single-sided power spectal density.
    To save computational efforts, the max length of time series is 16384.
    ****If desired length>16384, time series will be repeated. Notice repeat will
    cause PSD of generated time series differs from the reference PSD.****
    Args:
        sxx: 1D single-sided PSD.
        freq: frequency responding to sxx.
        fs: samplling frequency.
        n: samples of the time series.
    Returns:
        status: true if sucess, false if error.
        x: time series
    """
    x = np.zeros((n,))
    ### check input sampling frequency
    if fs < 2.0*freq[-1] or fs < 0.0:
        return False, x
    ### check if interpolation is needed
    repeat_output = False
    N = n
    if n%2 != 0:                                # N should be even
        N = n+1
        repeat_output = True
    if N > 16384:                               # max data length is 16384
        N = 16384
        repeat_output = True
    ### convert psd to time series
    L = freq.shape[0]                           # num of samples in psd
    if L != N//2+1:                             # interp psd instead of output
        L = N//2 + 1
        freq_interp = np.linspace(0, fs/2.0, L)
        sxx = np.interp(freq_interp, freq, sxx)
    sxx[1:L-1] = 0.5 * sxx[1:L-1]               # single-sided psd amplitude to double-sided
    ax = np.sqrt(sxx*N*fs)                      # double-sided frequency spectrum amplitude
    phi = math.pi * np.random.randn(L)          # random phase
    xk = ax * np.exp(1j*phi)                    # single-sided frequency spectrum
    xk = np.hstack([xk, xk[-2:0:-1].conj()])    # double-sided frequency spectrum
    xm = np.fft.ifft(xk)                        # inverse fft
    x_tmp = xm.real                             # real part
    ### repeat x to output time series of desired lenght n
    if repeat_output is True:
        repeat_num = n // N
        repeat_remainder = n % N
        x = np.hstack([np.tile(x_tmp, (repeat_num,)), x_tmp[0:repeat_remainder]])
    else:
        x = x_tmp
    return True, x


def bias_drift(corr_time, drift, n, fs):
    """
    Bias drift (instability) model for accelerometers or gyroscope.
    If correlation time is valid (positive and finite), a first-order Gauss-Markov model is used.
    Otherwise, a simple normal distribution model is used.
    Args:
        corr_time: 3x1 correlation time, sec.
        drift: 3x1 bias drift std, rad/s.
        n: total data count
        fs: sample frequency, Hz.
    Returns
        sensor_bias_drift: drift of sensor bias
    """
    # 3 axis
    sensor_bias_drift = np.zeros((n, 3))
    for i in range(0, 3):
        if not math.isinf(corr_time[i]):
            # First-order Gauss-Markov
            a = 1 - 1/fs/corr_time[i]
            # For the following equation, see issue #19 and
            # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3812568/ (Eq. 3).
            b = drift[i] * np.sqrt(1.0 - np.exp(-2/(fs * corr_time[i])))
            #sensor_bias_drift[0, :] = np.random.randn(3) * drift
            drift_noise = np.random.randn(n, 3)
            for j in range(1, n):
                sensor_bias_drift[j, i] = a*sensor_bias_drift[j-1, i] + b*drift_noise[j-1, i]
        else:
            # normal distribution
            sensor_bias_drift[:, i] = drift[i] * np.random.randn(n)
    return sensor_bias_drift


def acc_gen(fs, ref_a, acc_err, vib_def=None):
    """
    Add error to true acc data according to acclerometer model parameters
    Args:
        fs: sample frequency, Hz.
        ref_a: nx3 true acc data, m/s2.
        acc_err: accelerometer error parameters.
            'b': 3x1 acc constant bias, m/s2.
            'b_drift': 3x1 acc bias drift, m/s2.
            'vrw': 3x1 velocity random walk, m/s2/root-Hz.
        vib_def: Vibration model and parameters. Vibration type can be random, sinunoida or
            specified by single-sided PSD.
            Generated vibrating acc is expressed in the body frame.
            'type' == 'random':
                Normal distribution. 'x', 'y' and 'z' give the 1sigma values along x, y and z axis.
                units: m/s2
            'type' == 'sinunoidal'
                Sinunoidal vibration. 'x', 'y' and 'z' give the amplitude of the sine wave along
                x, y and z axis. units: m/s2.
            'type' == 'psd'. Single sided PSD.
                'freq':  frequency, in unit of Hz
                'x': x axis, in unit of (m/s^2)^2/Hz.
                'y': y axis, in unit of (m/s^2)^2/Hz.
                'z': z axis, in unit of (m/s^2)^2/Hz.
    Returns:
        a_mea: nx3 measured acc data
    """
    dt = 1.0/fs
    # total data count
    n = ref_a.shape[0]
    ## simulate sensor error
    # static bias
    acc_bias = acc_err['b']
    # bias drift
    acc_bias_drift = bias_drift(acc_err['b_corr'], acc_err['b_drift'], n, fs)
    
    # vibrating acceleration
    acc_vib = np.zeros((n, 3))
    if vib_def is not None:
        if vib_def['type'].lower() == 'psd':
            acc_vib[:, 0] = time_series_from_psd(vib_def['x'], vib_def['freq'], fs, n)[1]
            acc_vib[:, 1] = time_series_from_psd(vib_def['y'], vib_def['freq'], fs, n)[1]
            acc_vib[:, 2] = time_series_from_psd(vib_def['z'], vib_def['freq'], fs, n)[1]
        elif vib_def['type'] == 'random':
            acc_vib[:, 0] = vib_def['x'] * np.random.randn(n)
            acc_vib[:, 1] = vib_def['y'] * np.random.randn(n)
            acc_vib[:, 2] = vib_def['z'] * np.random.randn(n)
        elif vib_def['type'] == 'sinusoidal':
            acc_vib[:, 0] = vib_def['x'] * np.sin(2.0*math.pi*vib_def['freq']*dt*np.arange(n))
            acc_vib[:, 1] = vib_def['y'] * np.sin(2.0*math.pi*vib_def['freq']*dt*np.arange(n))
            acc_vib[:, 2] = vib_def['z'] * np.sin(2.0*math.pi*vib_def['freq']*dt*np.arange(n))
    # accelerometer white noise
    acc_noise = np.random.randn(n, 3)
    acc_noise[:, 0] = acc_err['vrw'][0] / math.sqrt(dt) * acc_noise[:, 0]
    acc_noise[:, 1] = acc_err['vrw'][1] / math.sqrt(dt) * acc_noise[:, 1]
    acc_noise[:, 2] = acc_err['vrw'][2] / math.sqrt(dt) * acc_noise[:, 2]

    # true + constant_bias + bias_drift + noise
    a_mea = ref_a + acc_bias + acc_bias_drift + acc_noise + acc_vib
    return a_mea


def gyro_gen(fs, ref_w, gyro_err, vib_def=None):
    """
    Add error to true gyro data according to gyroscope model parameters
    Args:
        fs: sample frequency, Hz.
        ref_w: nx3 true acc data, rad/s.
        gyro_err: gyroscope error parameters.
            'b': 3x1 constant gyro bias, rad/s.
            'b_drift': 3x1 gyro bias drift, rad/s.
            'arw': 3x1 angle random walk, rad/s/root-Hz.
        vib_def: Vibration model and parameters. Vibration type can be random, sinunoida or
            specified by single-sided PSD.
            Generated vibrating acc is expressed in the body frame.
            'type' == 'random':
                Normal distribution. 'x', 'y' and 'z' give the 1sigma values along x, y and z axis.
                units: rad/s
            'type' == 'sinunoidal'
                Sinunoidal vibration. 'x', 'y' and 'z' give the amplitude of the sine wave along
                x, y and z axis. units: rad/s.
            'type' == 'psd'. Single sided PSD.
                'freq':  frequency, in unit of Hz
                'x': x axis, in unit of (rad/s)^2/Hz.
                'y': y axis, in unit of (rad/s)^2/Hz.
                'z': z axis, in unit of (rad/s)^2/Hz.
    Returns:
        w_mea: nx3 measured gyro data
    """
    dt = 1.0/fs
    # total data count
    n = ref_w.shape[0]
    ## simulate sensor error
    # static bias
    gyro_bias = gyro_err['b']
    # bias drift
    gyro_bias_drift = bias_drift(gyro_err['b_corr'], gyro_err['b_drift'], n, fs)
    # vibrating gyro
    gyro_vib = np.zeros((n, 3))
    if vib_def is not None:
        if vib_def['type'].lower() == 'psd':
            gyro_vib[:, 0] = time_series_from_psd(vib_def['x'], vib_def['freq'], fs, n)[1]
            gyro_vib[:, 1] = time_series_from_psd(vib_def['y'], vib_def['freq'], fs, n)[1]
            gyro_vib[:, 2] = time_series_from_psd(vib_def['z'], vib_def['freq'], fs, n)[1]
        elif vib_def['type'] == 'random':
            gyro_vib[:, 0] = vib_def['x'] * np.random.randn(n)
            gyro_vib[:, 1] = vib_def['y'] * np.random.randn(n)
            gyro_vib[:, 2] = vib_def['z'] * np.random.randn(n)
        elif vib_def['type'] == 'sinusoidal':
            gyro_vib[:, 0] = vib_def['x'] * np.sin(2.0*math.pi*vib_def['freq']*dt*np.arange(n))
            gyro_vib[:, 1] = vib_def['y'] * np.sin(2.0*math.pi*vib_def['freq']*dt*np.arange(n))
            gyro_vib[:, 2] = vib_def['z'] * np.sin(2.0*math.pi*vib_def['freq']*dt*np.arange(n))
    # gyroscope white noise
    gyro_noise = np.random.randn(n, 3)
    gyro_noise[:, 0] = gyro_err['arw'][0] / math.sqrt(dt) * gyro_noise[:, 0]
    gyro_noise[:, 1] = gyro_err['arw'][1] / math.sqrt(dt) * gyro_noise[:, 1]
    gyro_noise[:, 2] = gyro_err['arw'][2] / math.sqrt(dt) * gyro_noise[:, 2]
    # true + constant_bias + bias_drift + noise
    w_mea = ref_w + gyro_bias + gyro_bias_drift + gyro_noise + gyro_vib
    return w_mea


def run_acc_demo():
    fs = 100  # Hz
    num_samples = 1000
    ref_a = np.zeros((num_samples, 3))

    data = []
    for i in range(1000):
        x = i +1
        y = i +2
        z = i + 3
        d = np.array([x, y, z])
        data.append(d)

    data = np.asarray(data)

    print(d)
    print(d.shape)
    print(data)
    print(data.shape)


    print(ref_a.shape)
    acc_err = accel_high_accuracy

    # sets random vibration to accel with RMS for x/y/z axis - 1/2/3 m/s^2, can be zero or changed to other values
    env = '[0.03 0.001 0.01]-random'
    vib_def = vib_from_env(env, fs)

    real_acc = acc_gen(fs, ref_a, acc_err, vib_def)

    # print(real_acc)



def run_gyro_demo():
    fs = 100  # Hz
    num_samples = 10
    ref_w = np.zeros((num_samples, 3))
    print(ref_w)
    gyro_err = accel_high_accuracy

    # sets sinusoidal vibration to gyro with frequency being 0.5 Hz and amp for x/y/z axis being 6/5/4 deg/s
    env = '[6 5 4]d-0.5Hz-sinusoidal'
    vib_def = vib_from_env(env, fs)

    real_gyro = acc_gen(fs, ref_w, gyro_err, vib_def)

    print(real_gyro)


def get_vel(p2, p1, dt):
    v = (p2 - p1)/dt
    return v


def get_acc(p3, p2, p1, dt):
    v1 = get_vel(p2, p1, dt)
    v2 = get_vel(p3, p2, dt)

    a = (v2 - v1)/dt
    return a


def cal_imu_step(imu_rate, frame_rate):
    """
    f_dt = 1.0/frame_rate
    imu_dt = 1.0/imu_rate

    1 imu_step correspond to f_dt, i.e after reading ith value from position array, read the (i+imu_step) indexed value
    """

    imu_step = int(imu_rate/frame_rate)  # (f_dt/imu_dt)
    if imu_step <= 1:
        print("ERROR: IMU RATE SMALLER THAN TO FRAME RATE")
        exit()

    return imu_step


def cal_linear_acc(x_array, y_array, z_array, imu_rate=30.0):
    """
    param: array of positions for x, y, z
    brief: at least 3 positions values for x y z each from time t1 to t2 to t3 or tn
            we will be calculating the values at frame_rate of video, i.e at 30 frame_rate if not specified
    return: ndarray of size Nx3, N is the size of the array
    """

    dt = 1.0/imu_rate
    step = 1
    i = 0
    accel_data = []

    while len(x_array) - 1 >= i:
        if i-2*step >= 0:
            ax = get_acc(x_array[i], x_array[i-step], x_array[i-2*step], dt)
            ay = get_acc(y_array[i], y_array[i-step], y_array[i-2*step], dt)
            az = get_acc(z_array[i], z_array[i-step], z_array[i-2*step], dt)
            
            data = np.array([ax, ay, az])
            accel_data.append(data)

        i += step

    return np.asarray(accel_data)


def cal_angular_vel(roll_array, pitch_array, yaw_array, imu_rate=30.0):
    """
    param: array of positions for roll, pitch, yaw
    brief: at least 2 positions values for roll pitch yaw each from time t1 to t2
    return: ndarray of size Nx3, N is the size of the array
    """

    dt = 1.0/imu_rate
    step = 1
    i = 0
    gyro_data = []

    while len(roll_array) - 1 >= i:

        if i-step >= 0:
            wx = get_vel(roll_array[i], roll_array[i - step], dt)
            wy = get_vel(pitch_array[i], pitch_array[i - step], dt)
            wz = get_vel(yaw_array[i],  yaw_array[i - step], dt)

            data = np.array([wx, wy, wz])
            gyro_data.append(data)

        i += step

    return np.asarray(gyro_data)


def accelerometer_demo():
    t_array = []
    x_array = []
    y_array = []
    z_array = []

    dist = 0.0
    dt = 0.0
    offset = 150  # initial position
    accel = 2  # meter/second squared
    imu_rate = 120.0
    v_rate = 30.0

    acc_is_const = True

    """
    generate some position data according to a fixed acceleration
    check whether we can get back the same value for acceleration       
    """
    for i in range(100):
        dist = offset + accel * 0.5 * (dt ** 2)
        x_array.append(dist)
        y_array.append(dist)
        z_array.append(dist)
        t_array.append(dt)
        dt += 1.0 / imu_rate

    print(t_array[:5])
    print(x_array[:5])
    print(y_array[:5])
    print(z_array[:5])
    accel_data = cal_linear_acc(x_array, y_array, z_array, imu_rate, v_rate)
    print('\n')
    print(accel_data[0])
    print(accel_data[1])
    print(accel_data[2])


def gyroscope_demo():
    t_array = []
    roll_array = []
    pitch_array = []
    yaw_array = []

    angle = 0.0
    dt = 0.0
    offset_angle = 359.5  # initial position
    angular_vel = 2  # degrees/second
    imu_rate = 120.0
    v_rate = 30.0

    """
    generate some angle data according to a fixed angular velocity
    check whether we can get back the same value for angular velocity      
    """
    for i in range(100):
        angle = offset_angle + angular_vel * dt
        roll_array.append(angle)
        pitch_array.append(angle)
        yaw_array.append(angle)
        t_array.append(dt)
        dt += 1.0 / imu_rate

    print(t_array[:5])
    print(roll_array[:5])
    print(pitch_array[:5])
    print(yaw_array[:5])
    gyro_data = cal_angular_vel(roll_array, pitch_array, yaw_array, imu_rate, v_rate)
    print('\n')
    print(gyro_data[0])
    print(gyro_data[1])
    print(gyro_data[2])


if __name__ == "__main__":
    pass
    # run_acc_demo()
