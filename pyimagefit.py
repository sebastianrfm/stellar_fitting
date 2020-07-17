import numpy as np
import math
import time

from scipy import stats
from scipy.signal import convolve2d
from scipy.special import beta, gamma
from scipy.optimize import minimize

import matplotlib.pyplot as plt

# todo:
# 1. how to define likelihood func

def calculate_model_image(params, image, obj_list, psf, rect_region):

    h, w = image.shape[0],image.shape[1]
    xmin, xmax, ymin, ymax = rect_region
    mw, mh = xmax - xmin, ymax - ymin
    model_image = np.zeros((h,w)) 

    # Generate each layer and flatten
    ind = 0
    for o in obj_list:
        F_tot, r_e, x0, y0, q, ang = params[ind:ind + 6]

        if o == 0:
            model_layer = object_devac(mh, mw, F_tot, r_e, x0, y0, q, ang, xmin, ymin)
        elif o == 1:
            model_layer = object_disk(mh, mw, F_tot, r_e, x0, y0, q, ang, xmin, ymin)
        elif o == 2:
            model_layer = object_gaussian(mh, mw, F_tot, r_e, x0, y0, q, ang, xmin, ymin)
            
        model_image[xmin:xmax,ymin:ymax] += model_layer
        ind = ind + 6
        
    # Convolve telescope psf with model image
    if psf is not None:
        model_image[xmin:xmax,ymin:ymax] = convolve2d(model_image[xmin:xmax,ymin:ymax], psf, mode = 'same')
        
    return model_image

def calculate_log_likelihood(image, fit, segm, sig):

    d = im[segm]
    f = fit[segm]
    n = len(im)
    res = np.power(np.sum(d - f),2)
    ll = -(.5*n)*np.log(2*pi)-(.5*n)*np.log(sig**2)-(1/(2*(sig**2)))*chisq
    
    if ll is None:
        ll = -np.inf
    
    return ll

def fast_morph_fit(p0, decomp, data, phot_data):
    
    def func(p, data, phot_data, phot_p):

        tot_chisq = 0
        for i, image in enumerate(data.images):
            
            image_params, obj_list = generate_image_params(p, phot_p, i, data, phot_data)
            model = calculate_model_image(image_params, image, obj_list, data.psf[i], data.rect_region)
            chisq = np.sum(np.power((model[data.segm] - image[data.segm])/data.std,2))
            tot_chisq += chisq

        #print(tot_chisq/(len(data.images)*data.dof - len(p)))
        return tot_chisq

    xmin,xmax,ymin,ymax = data.rect_region
    b = [(xmin,xmax),(ymin,ymax),(0,2.*np.pi),(.01,None),(.01,None),(1.,None),(1.,None)]
    b += [(xmin,xmax),(ymin,ymax)]*data.n_clumps
    b += [(0.,None)]*len(data.images)
    t0 = time.clock()
    res = minimize(func, p0,  args = (data, phot_data, decomp), bounds = b)
    print('Minimizing Time:')
    print(time.clock() - t0)
    print('Chisq Value:')
    chisq = func(p0, data, phot_data, decomp)
    print(chisq/(len(data.images)*data.dof - len(res.x)))

    return res.x, chisq

def generate_image_params(p, phot_p, image_id, data, phot_data):

    # Input Params Organization:
    # 0-1       -- x_c and y_c of devac + disk        
    # 2         -- PA of devac + disk                  
    # 3-4       -- ellipticities of devac + disk       
    # 5-6       -- size components of devac + disk    
    # 7-7+n     -- x_c and y_c of young clumps in format [x_c1,y_c1,x_c2,y_c2,...,x_cn,y_cn]
    # 7+n-7+n+m -- flux scaling for m images
    
    # Object Params Organization:
    # F_tot, r_e, x0, y0, q, ang    
    
    band = np.where(phot_data.filter_ids == data.bands[image_id])
    band_flux = phot_data.flux[band]

    weight_norm = np.sum(phot_p[0:3])
    med_flux = (phot_p[0]/weight_norm)*band_flux
    old_flux = (phot_p[1]/weight_norm)*band_flux
    young_flux = (phot_p[2]/weight_norm)*band_flux

    norms = phot_data.continuum_norms
    fit_flux = phot_data.flux_vectors[:,band].flatten()/norms

    #maybe something funky going on here, a lot of young light where it seems weird
    #but normalized fractions work, maybe its an index problem (i could have them mixed up)
    med_frac = fit_flux[0]*phot_p[0]
    old_frac = fit_flux[1]*phot_p[1]
    young_frac = fit_flux[2]*phot_p[2]
    frac_norm = np.sum([med_frac,old_frac,young_frac])

    med_flux = med_frac*p[7 + 2*data.n_clumps + image_id]
    old_flux = old_frac*p[7 + 2*data.n_clumps + image_id]
    young_flux = young_frac*p[7 + 2*data.n_clumps + image_id]

    clump_flux = data.clumps*young_flux
    clump_sfr = data.clumps*phot_data.sfr

    # Generate params for med + old profiles
    med_params = [med_flux,p[6],p[0],p[1],p[4],p[2]]
    old_params = [old_flux,p[5],p[0],p[1],p[3],p[2]]

    image_params = old_params + med_params

    # Generate params for each clump
    for j, flx in enumerate(clump_flux):
            
        size = sfr_to_rad(clump_sfr[j], phot_data.z)
        pos_id = 7 + 2*j

        params = [flx, size, p[pos_id], p[pos_id + 1],1.,0.]
        image_params += params

    return image_params, data.obj_list

def object_devac(h, w, F_tot, r_e, x0, y0, q, ang, xmin, ymin):

    x,y = np.meshgrid(np.arange(h) + xmin,np.arange(w) + ymin)
    img = sersic_profile_func(x,y,x0,y0,F_tot,4,r_e,0,q,ang)
    
    return img

def object_disk(h, w, F_tot, r_e, x0, y0, q, ang, xmin, ymin):
    
    x,y = np.meshgrid(np.arange(h) + xmin,np.arange(w) + ymin)
    img = exponential_disk_func(x,y,x0,y0,F_tot,r_e, 0, q, ang)
                      
    return img
    
def object_gaussian(h, w, F_tot, r_e, x0, y0, q, ang, xmin, ymin):

    x,y = np.meshgrid(np.arange(h) + xmin,np.arange(w) + ymin)
    img = gaussian_profile_func(x,y,x0,y0,F_tot,r_e, 0,q, ang)

    return img

def exponential_disk_func(x, y, x0, y0, F_tot, r_e, c = 0., q = 1., ang = 0.):

    r_s = r_e*1.678
    amp_cent = (F_tot*area_ratio(c))/(2*np.pi*r_s*q)
    r = elliptical_radial_coord(x, y, x0, y0, c, q, ang)

    amp = amp_cent*np.exp(-r/r_s)

    return amp

def gaussian_profile_func(x, y, x0, y0, F_tot, r_e, c = 0, q = 1, ang = 0):

    sig = r_e/1.1774
    amp_cent = (F_tot*area_ratio(c))/(2*np.pi*(sig**2)*q)
    r = elliptical_radial_coord(x, y, x0, y0, c, q, ang)

    amp = amp_cent*np.exp(-.5*np.power(r/sig,2))
    
    return amp

def sersic_profile_func(x, y, x0, y0, F_tot, n , r_e, c = 0, q = 1, ang = 0):

    k = 2*n - 0.331
    amp_cent = (F_tot*area_ratio(c))/(2*np.pi*np.power(r_e, 2.)*np.exp(k)*n*np.power(k, -2*n)*gamma(2*n)*q)
    r = elliptical_radial_coord(x, y, x0, y0, c, q, ang)
    
    amp = amp_cent*np.exp(-k*(np.power(r/r_e, 1./n) - 1.))
    
    return amp

def area_ratio(c):

    if c == 0:
        return 1.

    R = (np.pi*c)/(4*beta(1./c,1 + 1./c))

    return R

def elliptical_radial_coord(x, y, x0, y0, c, q, ang):

    x_mod = x - x0
    y_mod = y - y0

    cos = np.cos(ang)
    sin = np.sin(ang)

    x_rot = x_mod*cos - y_mod*sin
    y_rot = y_mod*cos + x_mod*sin

    r = np.power(np.power(x_rot, c + 2.) + np.power(y_rot/q, c + 2.), 1./(c + 2.))

    return r

# Pixel to Arcsec conversion found from CANDELS HST Imaging README
# ACS/WFC and WFC3/UVIS have same scale 0.03, IR scale is 0.06
# Parsecs calculated from the digitized plots of Livermore et al 2015
def sfr_to_rad(sfr, z, scale = 0.03):

    lookup = np.loadtxt('d_angular_lookup.txt')
    
    radius = np.exp(.5*np.log(sfr) + 6.275)
    distance = np.interp(z,lookup[:,0],lookup[:,1])*(10**6)
    diameter_rad = 2*np.arctan(radius/distance)
    diameter_arcs = diameter_rad*(648000/np.pi)
    diameter_pixels = diameter_arcs/scale

    return diameter_pixels
    
def calculate_model_likelihood(modelim, image, segm, std):

    chisq = -.5*np.power((modelim[segm] - image[segm])/std,2)
    log_like = np.sum(chisq)

    return log_like

def plot_model_image(image):

    ln_image = np.log(image)
    
    plt.figure()
    plt.imshow(ln_image, origin='lower', interpolation='nearest', cmap = 'gray',  vmin=-1, vmax=2)
    plt.xlabel('x')
    plt.ylabel('y')
    cbar = plt.colorbar()
    cbar.set_label('Log Brightness', rotation=270, labelpad=25)
    cbar.set_ticks([-1, 0, 1, 2], update_ticks=True)
    plt.show()

    return

# F_tot, r_e, x0, y0, q, ang)
#if __name__ == '__main__':

    #dummy_img = np.zeros((200,200))
    #dummy_data = setup_data(dummy_img)
    #data_image = object_devac(dummy_data, 1000, 15, 100, 100, .8, 1.1)
    #print(data_image)
    #plot_model_image(data_image)

    #p = np.array([1000, 15, 100, 100, .8, 1.1, 700, 25, 150, 150, .6, 2.5, 1300, 5, 45, 72, 1.4, .5])
    #o = np.array([0,1,2])
    #img = calculate_model_image(p,dummy_data,o)
    #plot_model_image(img)

    
