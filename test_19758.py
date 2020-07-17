import pyphotfit_io as ppf_io
import pyimagefit_io as pif_io
import pyimagefit as pif
import pyphotfit as ppf
import matplotlib.pyplot as plt
import numpy as np
import time
import emcee
from contextlib import closing

from astropy.io import fits
from scipy.optimize import minimize
from matplotlib.colors import LogNorm
from multiprocessing import Pool

# degbugs :)
debug_morph_fitting = False

# Setup
# 19708, 1.42, 1.5, 1.87
# 202, 1.82, .9, 1.59
source_id = 19708
z = 1.42
a_v = 1.5
sfr = 1.87

phot_data = ppf_io.setup_data(source_id, z, a_v, np.exp(sfr),[1216,2797,3727,5007],[(0,None),(0,None),(0,None),(0,None)])

hdul1 = fits.open('testfiles/19708.ACS_F814W.F12.s40.cuts.fits')
#hdul1 = fits.open('testfiles/202.ACS_F814W.F9.s40.cuts.fits')
im1 = hdul1[0].data

hdul2 = fits.open('testfiles/19708.WFC3_F125W.F13.s40.cuts.fits')
#hdul2 = fits.open('testfiles/202.WFC3_F125W.F10.s40.cuts.fits')
im2 = hdul2[0].data

psf1 = fits.open('psf/UDS01_F160_smallPSF.fits')[0].data
psf2 = fits.open('psf/UDS01_F814_smallPSF.fits')[0].data

morph_data = pif_io.setup_data([im1,im2],[3,7],[psf1,psf2])

# Fast Initial Photometric Fit
p0 = [.33,.33,.33,10.,20.,5.,15.,15.]
pi, chisq = ppf.fast_phot_fit(p0, phot_data)

fit = ppf.calculate_model_photometry(pi,phot_data)
wl = phot_data.wl
data = phot_data.flux
err = phot_data.err

plt.plot(wl,fit,'rx')
plt.errorbar(wl,data,yerr=err,fmt='bx')
plt.show()

# Fast Initial Morphology Fit
xmin,xmax,ymin,ymax = morph_data.rect_region
dx = xmax - xmin
dy = ymax - ymin
randx = xmin + np.random.rand(morph_data.n_clumps)*dx
randy = ymin + np.random.rand(morph_data.n_clumps)*dy
pclump = np.reshape(np.array([randx,randy]),2*len(randx),order = 'F').tolist()

p0 = [morph_data.image_h/2,morph_data.image_w/2,0.,1.,1.,4.,4.]
p0 += pclump
p0 += morph_data.band_flux.tolist()

# tests to see if the morph fitting works against synthetic data
if debug_morph_fitting:
    morph_data.psf = np.array([None,None])
    test_randx = xmin + np.random.rand(morph_data.n_clumps)*dx
    test_randy = ymin + np.random.rand(morph_data.n_clumps)*dy
    test_pclump = np.reshape(np.array([test_randx,test_randy]),2*len(randx),order = 'F').tolist()
    test_p = [morph_data.image_h/2 + 5,morph_data.image_w/2 + 5,1.2,.8,1.1,10.,12.] + test_pclump + morph_data.band_flux.tolist()
    test_params1, obj1 = pif.generate_image_params(test_p, pi, im1, 0, morph_data, phot_data)
    test_params2, obj2 = pif.generate_image_params(test_p, pi, im2, 1, morph_data, phot_data)
    t_im1 = pif.calculate_model_image(test_params1, im1, obj1, morph_data.psf[0], morph_data.rect_region)
    t_im2 = pif.calculate_model_image(test_params2, im1, obj2, morph_data.psf[1], morph_data.rect_region)
    morph_data.images = np.array([t_im1,t_im2])
    
pm, chisq = pif.fast_morph_fit(p0, pi, morph_data, phot_data)

print('Phot Values:')
print(pi)
print('Morph Values:')
print(pm)

len_pi = len(pi)
len_pm = len(pm)

# mcmc fit

def total_log_likelihood(theta):

    theta_p = theta[0:len_pi]
    theta_m = theta[len_pi:]
    
    # Priors ll
    lp = log_priors(theta)

    # Photometric ll
    #model_phot = ppf.calculate_model_photometry(theta_p, phot_data)
    #ll_phot = ppf.calculate_log_likelihood(model_phot, phot_data.flux, phot_data.err)

    # Morphological ll
    #ll_morph = 0
    #for i, image in enumerate(data.images):

    #    params = pif.generate_image_params(theta_m, theta_p, i, morph_data, phot_data)
    #    model_image = pif.calculate_model_image(params, image, morph_data.obj_list, morph_data.psf[i], morph_data.rect_region)
    #    ll_morph += pif.calculate_log_likelihood(image, model_image, data.segm, data.std)

    #ll_total = lp + ll_phot + ll_morph
    ll_total = lp
    
    return ll_total

def log_priors(theta):

    # Theta organization by index
    # -Photometric Params
    #   0-2         -- med, old, young weights
    #   3-6         -- hb (clumped), la, mg2, o2, o3 equivalent widths
    # -Morphological Params:
    #   7-8         -- x_c and y_c of devac + disk        
    #   9           -- PA of devac + disk                  
    #   10-11       -- ellipticities of devac + disk       
    #   12-13       -- size components of devac + disk    
    #   14-14+n     -- x_c and y_c of young clumps in format [x_c1,y_c1,x_c2,y_c2,...,x_cn,y_cn]
    #   14+n-14+m   -- flux "fudge" factor for each image where m is number of images

    d,e,s = theta[0:3] # 0< and <1
    x,y = theta[7:9] #in rect region
    pa = theta[9] #<2pi and 0<
    ell = theta[10:12] # > .01
    size = theta[12:14] # >1 pizel
    x_c = theta[14:14 + 2*morph_data.n_clumps - 1:2] # in rect region
    y_c = theta[14 + 1:14 + 2*morph_data.n_clumps:2] # in rect region
    fudge = theta[14 + 2*morph_data.n_clumps:] # 0<
    xmin,xmax,ymin,ymax = morph_data.rect_region
    
    if (0. < d < 1.) and (0. < e < 1.) and (0. < s < 1.) and (xmin < x < xmax) and (ymin < y < ymax) and (0 < pa < 2*np.pi) and np.all(ell > 0.01) and np.all(size > 1) and np.all(xmin < x_c < xmax) and np.all(ymin < y_c < ymax) and np.all(fudge > 0):
        return 0
    else:
        return -np.inf

def run_mcmc(p0, phot_data, morph_data, fname = 'mcmc_output.h5'):

    ndim,nwalkers = len(p0),64
    pos = np.array([np.abs(p0 + 1e-2*np.random.randn(ndim)) for i in range(nwalkers)])

    backend = emcee.backends.HDFBackend(fname)
    backend.reset(nwalkers, ndim)
    
    with closing(Pool(4)) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, total_log_likelihood, pool = pool, backend = backend) #change a = scaling
        max_n = 30000

        index = 0
        autocorr = np.empty(max_n)
        old_tau = np.inf

        for sample in sampler.sample(pos, iterations = max_n, progress = True):
            if sampler.iteration % 1000:
                continue

            tau = sampler.get_autocorr_time(tol = 0)
            autocorr[index] = np.mean(tau)
            index += 1

            converged = np.all(tau*100 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau)/tau < 0.01)

            if converged:
                break
            old_tau = tau

    tau = sampler.get_autocorr_time()
    burnin = int(2*np.max(tau))

    samples = sampler.get_chain(discard=burnin, flat=True)
    log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True)
    log_prior_samples = sampler.get_blobs(discard=burnin, flat=True)

    outputs = np.zeros(ndim)
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        outputs[i] = mcmc[1]
        
    return outputs

p0_mcmc = np.append(pi, pm)
outputs = run_mcmc(p0_mcmc, phot_data, morph_data, fname = 'testing.h5')
