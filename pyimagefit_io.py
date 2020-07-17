import astropy.io.fits as fits
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from photutils import detect_sources, make_source_mask
from astropy.stats import sigma_clipped_stats

class Morph_Data():
    pass

def setup_data(images, band_indexes, psf = None):

    data = Morph_Data()

    data.images = np.array(images)
    data.image_h = data.images[0].shape[0]
    data.image_w = data.images[0].shape[1]
    data.psf = psf
    data.bands = band_indexes
    data.n_bands = len(data.bands)
    
    data.clumps = pull_clumps()
    data.n_clumps = len(data.clumps)

    data.obj_list = [0,1] + [2]*data.n_clumps
    
    data.segm, data.rect_region, data.std, data.dof = extract_source(data.images)
    
    data.band_flux = np.zeros(len(images))
    for i,im in enumerate(images):

        flux = calculate_source_flux(im, data.segm)
        data.band_flux[i] = flux
        
    return data

# Generates segmap for our source
def extract_source(images):

    # Calculate image noise
    stack = np.sum(images, axis = 0)
    mask = make_source_mask(stack, snr=2, npixels=5, dilate_size=11)
    mean, median, std = sigma_clipped_stats(stack, sigma=3.0, mask=mask)

    # Pull out all sources
    segm = detect_sources(stack, 2*std, npixels = 5)

    # Screen for our source
    cx = int(stack.shape[0]/2)
    cy = int(stack.shape[1]/2)
    center_pixel = segm.data[cx,cy]
    segm.data[np.where(segm.data != center_pixel)] = 0

    region = np.zeros(stack.shape, dtype = bool)
    region[segm.data == center_pixel] = True
    cutout = np.where(region)

    pad = 3
    xmin,xmax = min(cutout[0]) - pad, max(cutout[0]) + pad
    ymin,ymax = min(cutout[1]) - pad, max(cutout[1]) + pad

    rect_region = [xmin,xmax,ymin,ymax]
    dof2 = (xmax - xmin)*(ymax - ymin)
    dof = len(stack[region])
    
    fig = plt.figure()
    ax1 = plt.subplot(121)
    ax1.imshow(stack[xmin:xmax,ymin:ymax], interpolation = 'none', cmap = 'gray')
    ax2 = plt.subplot(122)
    ax2.imshow(segm.data[xmin:xmax,ymin:ymax], interpolation = 'none')
    plt.show()
    
    return region, rect_region, std, dof

def calculate_source_flux(image, segm):
    
    density = image[segm].flatten()
    flux = np.trapz(density, dx = 1.0)

    return flux
    
# Returns set of clumps with fractional lum
def pull_clumps():

    # General lum func
    def scheter(norm, char_lum, alpha, lum):

        return (norm)*np.power(lum/char_lum, alpha)*np.exp(-lum/char_lum)

    # Normalize Scheter Function and Define Params
    char_lum = np.exp(-1.2)
    alpha = -1.

    l = np.linspace (.05,1.)
    norm = 1./np.trapz(scheter(1., .3, alpha, l), x = l)
    
    # Pull clumps from distribution
    clump_lums = []
    tot_frac_lum = 0
        
    while tot_frac_lum < 1.:
        
        frac_lum  = np.random.rand()*0.9 + 0.05
        threshold = np.random.rand()
        prob = scheter(norm, char_lum, alpha, frac_lum)/1.6

        if threshold < prob:

            clump_lums = np.append(clump_lums, frac_lum)
            tot_frac_lum += frac_lum
         
        else:
            
            continue

    # Coin flip to keep final pull or not
    coin = np.random.rand()
    
    if coin > .5:

        clump_lums = clump_lums[:-1]
        
    clump_lums = clump_lums*(1./np.sum(clump_lums))
            
    return clump_lums
