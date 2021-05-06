import tensorflow as tf
import numpy as np
import scipy.io as sio
from scipy.stats import poisson

###############################################################################
# Sensor model
###############################################################################

# Sensors calibrated for ISO100 (format is Poissonian scale, Gaussian std)
# Iso 100, 200, 400, 800, 1600, 3200
sensors = {'Nexus_6P_rear': [0.00018724, 0.0004733],
           'Nexus_6P_front': [0.00015, 0.0003875],
           'SEMCO': [0.000388, 0.0025],
           'OV2740': [0.000088021, 0.00022673],
           #'GAUSSIAN': [0,0.005],
           'GAUSS': [0,1],
           'POISSON': [1,0],
           'Pixel': [0.0153, 0.0328], #[0.00019856, 0.0017],
           'Pixel3x3': [2.2682e-4, 0.0017],
           'Pixel5x5': [1.2361e-4, 0.0043],
           'Pixel7x7': [7.3344e-05, 0.0077],
           }

sensorpositions = {'center': 0.5,
            'offaxis': 0.9,
           'periphery': 1.0}

light_levels = 3 * np.array([2 ** i for i in range(6)]) / 2000.0

def std2ll(std, mean=0.5, sensor='Nexus_6P_rear'):
    #light_level = sensors[sensor][1]/std
    #print('Sensor', sensor)
    alpha, beta = sensors[sensor]
    alpha_mean = alpha*mean
    num = np.sqrt(alpha_mean**2 + 4*beta**2*std**2) - alpha_mean
    light_level = (2*beta**2)/num
    return light_level

def get_bayer_mask(height, width):
    # Mask based on Bayer pattern. (assume RGB order of colors)
    # B G
    # G R
    bayer_mask = np.zeros([height, width, 3])
    bayer_mask[1::2,1::2,0:1] = 1 # R
    bayer_mask[1::2,::2,1:2] = 1 # G
    bayer_mask[::2,1::2,1:2] = 1 # G
    bayer_mask[::2,::2,2:3] = 1 # B
    return bayer_mask

def optics_model(psfs, sensorpos='center', visualize=True ):
    #Expects calibrated PSFs (in matlab format) as input

    #Compute positions on grid
    psf_shape = np.array(psfs.shape)
    selected_pos = (psf_shape*sensorpositions[sensorpos]).astype(int)

    #Extract the position
    psf_sel = psfs[selected_pos[0] - 1,selected_pos[1] - 1]['PSF'][0,0]
    psf_sel = np.maximum(psf_sel, 0.0)

    #Normalize
    for ch in range(psf_sel.shape[2]):
        psf_sel[:,:,ch] = psf_sel[:,:,ch]/np.sum(psf_sel[:,:,ch])

    return psf_sel

def psf_iterator():
    sensor_positions = ['center', 'offaxis', 'periphery']
    psfs = sio.loadmat('PSFs/bloc_256_Nexus_defective.mat')['bloc']
    for sensor_position in sensor_positions:
        psf_kernel = np.asfortranarray(optics_model(psfs, sensorpos=sensor_position, visualize=False).astype(np.float32))
        yield sensor_position, psf_kernel

def load_psfs():
    sensor_positions = ['center', 'offaxis', 'periphery']
    psfs = sio.loadmat('PSFs/bloc_256_Nexus_defective.mat')['bloc']

    kernels = []
    for sensor_position in sensor_positions:
        kernel = np.asfortranarray(optics_model(psfs, sensorpos=sensor_position, visualize=False).astype(np.float32))
        for channel in xrange(3):
            kernel[:,:,channel] /= np.sum(kernel[:,:,channel])
        kernels.append(kernel)

    return kernels

def get_noise_params(iso, sensor):
    sensor = 'Nexus_6P_rear'
    poisson = sensors[sensor][0]
    sigma = sensors[sensor][1]

    a = poisson * iso / 100.0 #Poisson scale
    b = (sigma * iso / 100.0)**2

    return a, np.sqrt(b)

def sensor_model(y):
    # Invalid sensor
    iso = 1.0 / 0.0015 * 100
    sensor='Nexus_6P_rear'

    poisson = sensors[sensor][0]
    sigma = sensors[sensor][1]

    #Output stats
    #print( 'Sensor {0} ISO {1} Poisson {2} Gaussian {3}'.format(sensor, iso, poisson, sigma) )

    # Assume linear ISO model
    a = poisson * iso / 100.0 #Poisson scale
    b = (sigma * iso / 100.0)**2

    #Return Poissonian-Gaussian response
    #noisy_img = poisson_gaussian_np(y, a, b, True, True)
    noisy_img = poisson_gaussian_np(y, a, b, True, True)
    return noisy_img.astype(np.float32)

def sensor_noise_rand_sigma(img_batch, sigma_range, scale=1.0, sensor='Nexus_6P_rear'):
    # Define in terms of Gaussian noise after Anscombe.
    batch_size = img_batch.get_shape()[0].value
    poisson = sensors[sensor][0]
    gauss = sensors[sensor][1]
    sigma = tf.random_uniform([batch_size], sigma_range[0], sigma_range[1])*scale/255.0
    if poisson == 0:
        noisy_batch = img_batch + sigma[:,None,None,None] * tf.random_normal(shape=img_batch.get_shape(), dtype=tf.float32)
        noisy_batch = tf.clip_by_value(noisy_batch, 0.0, scale)
        return noisy_batch, None, sigma
    sigma_hat = gauss/poisson
    offset = 2*tf.sqrt(3./8. + sigma_hat**2)
    tmp = (1./sigma + offset)**2/4 - 3./8. - sigma_hat**2
    light_level = poisson*tmp
    iso = 1.0 / light_level * 100.
    #iso = tf.Print(iso, [light_level])

    # Assume linear ISO model
    a = poisson * iso / 100.0  * scale #Poisson scale
    gauss_var = tf.square(gauss * iso / 100.0) * scale**2

    upper = 2*tf.sqrt(light_level/poisson + 3./8. + sigma_hat**2)
    lower = 2*tf.sqrt(3./8. + sigma_hat**2)
    tf.summary.scalar('noise_level', 255./(upper - lower)[0])
    tf.summary.scalar('iso', tf.reduce_mean(iso))
    tf.summary.scalar('light_level', tf.reduce_mean(light_level))
    tf.summary.scalar('a', tf.reduce_mean(a)/scale)
    tf.summary.scalar('gauss_variance', tf.reduce_mean(gauss_var)/scale**2)

    # a = tf.Print(a, [255./(upper - lower)])
    print("Simulating sensor {0}.".format(sensor))

    noisy_batch = poisson_gauss_tf(img_batch, a, gauss_var, clip=(0.,scale))
    # Return Poissonian-Gaussian response
    return noisy_batch, a, tf.sqrt(gauss_var)

def get_coeffs(light_levels, sensor='Nexus_6P_rear'):
    #print('Sensor', sensor)
    poisson = sensors[sensor][0]
    gauss = sensors[sensor][1]
    iso = 1.0 / light_levels * 100.
    a = poisson * iso / 100.0  #Poisson scale
    b = (gauss * iso / 100.0)
    return a, b

def sensor_noise_rand_light_level(img_batch, ll_range, scale=1.0, sensor='Nexus_6P_rear'):
    print("Sensor = %s, scale = %s" % (sensor, scale))
    batch_size = img_batch.get_shape()[0].value
    poisson = sensors[sensor][0]
    gauss = sensors[sensor][1]

    # Sample uniformly in logspace.
    # low ll * exp(u), u ~ [0, log(high ll/low ll)]
    ll_ratio = ll_range[1]/ll_range[0]
    ll_factor = tf.random_uniform([batch_size], minval=0, maxval=tf.log(ll_ratio), dtype=tf.float32)
    light_level = ll_range[0]*tf.exp(ll_factor)
    iso = 1.0 / light_level * 100.

    # Assume linear ISO model
    a = poisson * iso / 100.0  * scale #Poisson scale

    gauss_var = tf.square(gauss * iso / 100.0) * scale**2
    if poisson == 0:
        noisy_batch = img_batch + tf.sqrt(gauss_var[:,None,None,None]) * tf.random_normal(shape=img_batch.get_shape(), dtype=tf.float32)
        noisy_batch = tf.clip_by_value(noisy_batch, 0.0, scale)
        return noisy_batch, np.zeros(batch_size), tf.sqrt(gauss_var)

    tf.summary.scalar('iso', tf.reduce_mean(iso))
    tf.summary.scalar('light_level', tf.reduce_mean(light_level))
    tf.summary.scalar('a', tf.reduce_mean(a)/scale)
    tf.summary.scalar('gauss_variance', tf.reduce_mean(gauss_var)/scale**2)

    print("Simulating sensor {0}.".format(sensor))

    noisy_batch = poisson_gauss_tf(img_batch, a, gauss_var, clip=(0.,scale))
    sigma_hat = gauss/poisson

    return noisy_batch, a, tf.sqrt(gauss_var)

def poisson_gauss_tf(img_batch, a, gauss_var, clip=(0.,1.)):
    # Apply poissonian-gaussian noise model following A.Foi et al.
    # Foi, A., "Practical denoising of clipped or overexposed noisy images",
    # Proc. 16th European Signal Process. Conf., EUSIPCO 2008, Lausanne, Switzerland, August 2008.
    batch_shape = tf.shape(img_batch)

    a_p = a[:,None,None,None]
    out = tf.random_poisson(shape=[], lam=tf.maximum(img_batch/a_p, 0.0), dtype=tf.float32) * a_p
    #out = tf.Print(out, [tf.reduce_max(out), tf.reduce_min(out)])
    gauss_var = tf.maximum(gauss_var, 0.0)

    gauss_noise = tf.sqrt(gauss_var[:,None,None,None]) * tf.random_normal(shape=batch_shape, dtype=tf.float32) #Gaussian component

    out += gauss_noise

    # Clipping
    if clip is not None:
        out = tf.clip_by_value(out, clip[0], clip[1])

    # Return the simulated image
    return out


def poisson_gaussian_np(y, a, b, clip_below=True, clip_above=True):
    # Apply poissonian-gaussian noise model following A.Foi et al.
    # Foi, A., "Practical denoising of clipped or overexposed noisy images",
    # Proc. 16th European Signal Process. Conf., EUSIPCO 2008, Lausanne, Switzerland, August 2008.

    # Check method
    if(a==0):   # no Poissonian component
        z = y
    else:      # Poissonian component
        z = np.random.poisson( np.maximum(y/a,0.0) )*a;

    if(b<0):
        raise warnings.warn('The Gaussian noise parameter b has to be non-negative  (setting b=0)')
        b  = 0.0

    z = z + np.sqrt(b) * np.random.randn(*y.shape) #Gaussian component

    # Clipping
    if(clip_above):
        z = np.minimum(z, 1.0);

    if(clip_below):
        z = np.maximum(z, 0.0);

    # Return the simulated image
    return z

# Currently only implements one method
NoiseEstMethod = {'daub_reflect': 0, 'daub_replicate': 1}


def estimate_std(z, method='daub_reflect'):
    import cv2
    # Estimates noise standard deviation assuming additive gaussian noise

    # Check method
    if (method not in NoiseEstMethod.values()) and (method in NoiseEstMethod.keys()):
        method = NoiseEstMethod[method]
    else:
        raise Exception("Invalid noise estimation method.")

    # Check shape
    if len(z.shape) == 2:
        z = z[..., np.newaxis]
    elif len(z.shape) != 3:
        raise Exception("Supports only up to 3D images.")

    # Run on multichannel image
    channels = z.shape[2]
    dev = np.zeros(channels)

    # Iterate over channels
    for ch in range(channels):

        # Daubechies denoising method
        if method == NoiseEstMethod['daub_reflect'] or method == NoiseEstMethod['daub_replicate']:
            daub6kern = np.array([0.03522629188571, 0.08544127388203, -0.13501102001025,
                                  -0.45987750211849, 0.80689150931109, -0.33267055295008],
                                 dtype=np.float32, order='F')

            if method == NoiseEstMethod['daub_reflect']:
                wav_det = cv2.sepFilter2D(z, -1, daub6kern, daub6kern,
                                          borderType=cv2.BORDER_REFLECT_101)
            else:
                wav_det = cv2.sepFilter2D(z, -1, daub6kern, daub6kern,
                                          borderType=cv2.BORDER_REPLICATE)

            dev[ch] = np.median(np.absolute(wav_det)) / 0.6745

    # Return standard deviation
    return dev
