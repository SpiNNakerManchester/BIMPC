from ..common import *


def center_surround_kernel(width=3, ctr_sigma=0.8, sigma_mult=6.7, on_center=True):
    '''Compute the convolution kernel for center-surround behaviour,
       it sums to 0 and self-convolves to 1
       :param width: matrix width (we use square matrices)
       :param ctr_sigma: standard deviation for the center component
       :param sigma_mult: sigma_mult*ctr_sigma is the standard deviation 
                          for the surround component of the kernel

       :return cs_kernel: center-surround kernel
    '''
    cs_kernel = None
  
    ctr = gaussian2D(width, ctr_sigma)
    srr = gaussian2D(width, sigma_mult*ctr_sigma)
    if on_center:
      cs_kernel = ctr - srr
    else:
      cs_kernel = srr - ctr

    cs_kernel = sum2zero(cs_kernel)
    cs_kernel, w = conv2one(cs_kernel)
    
    return cs_kernel


def gaussian2D(width, sigma):
    '''Create a matrix with values that follow a 2D Gaussian
       function. 
       :param width: width of matrix
       :param sigma: standard deviation for the Gaussian
  
       :return gauss: 2D Gaussian
    '''
    half_width = width//2
    sigma_2 = sigma**2
    x, y = np.meshgrid(np.arange(-half_width, half_width + 1),
                       np.arange(-half_width, half_width + 1))
    x_2_plus_y_2 = x**2 + y**2
  
    norm_weight = (1./(2.*np.pi*sigma_2))
    gauss = (norm_weight*np.exp((-x_2_plus_y_2)/(2.*sigma_2)))
    
    return gauss


def split_center_surround_kernel(width=3, ctr_sigma=0.8, sigma_mult=6.7, on_center=True):
    
    cs_kernel = None
  
    ctr = gaussian2D(width, ctr_sigma)
    ctr, w = normalize(ctr)
    
    srr = gaussian2D(width, sigma_mult*ctr_sigma)
    srr, w = normalize(srr)
    
    if on_center:
        cs_kernel = ctr - srr
    else:
        cs_kernel = srr - ctr

    cs_kernel, w = conv2one(cs_kernel)
    
    ctr *= w
    srr *= w
  
    if on_center:
        srr = -srr
        return [ctr, srr] #excitatory first, inhibitory later
    else:
        ctr = -ctr
        return [srr, ctr]

    
