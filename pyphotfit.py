import numpy as np
from pyphotfit_io import setup_data
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def calculate_model_photometry(guess, data):

    # Continuum calculation
    ext = np.power(10.,-0.4*kl(data.wl/(1 + data.z))*data.E_bv)
    flux_noext = data.flux/ext
    err_noext = data.err/ext

    norms = np.sum(flux_noext)/np.sum(data.flux_vectors, axis = 1)
    continuum_vectors = norms[:,np.newaxis]*data.flux_vectors

    for i,g in enumerate(guess[0:3]):
        continuum_vectors[i] = g*continuum_vectors[i]

    continuum = np.sum(continuum_vectors, axis = 0)

    # Emission Calculation
    free_emission = np.zeros(continuum.shape)
    continuum_flux = np.zeros(data.continuum_means.shape[0])
    for j in range(len(data.bases)):
        continuum_flux += data.continuum_means[:,j]*guess[j]*norms[j]

    line_flux = guess[3:]*continuum_flux*(1 + data.z)
    clumped_emission = data.emission_vectors*line_flux[0]
    free_emission = np.sum(data.free_vectors*line_flux[1:,np.newaxis],axis = 0)

    total_simulated_flux = (continuum + clumped_emission + free_emission)*ext
    
    return total_simulated_flux

def calculate_log_likelihood(fit, data, err):

    chisq = np.sum((fit - data)/err, 2)
    norm = np.prod(np.power(2*np.pi*np.power(err,2), -.5))
    l = norm*chisq
    
    if l > 0:
        ll= np.log(prob)
    else:
        ll = -np.inf
        
    return ll

def kl(wavelengths):

    k = np.zeros(len(wavelengths))
    for i,l0 in enumerate(wavelengths):
        l = l0/10000.
        rv = 4.05

        if (l < .12):
            k[i] = -77.70*(l - .12) + 12.12
        elif (l <= .63):
            k[i] = 2.659*(-2.156+(1.509/l)-(.198/(l**2))+(.011/(l**3))) + rv
        elif (l <= 2.2):
            k[i] = 2.659*(-1.857+(1.040/l)) + rv
        else:
            k[i] = -0.5740*(l - 2.2) + .3692

    return k
    
def fast_phot_fit(p0, data):

    def func(p,data):

        model = calculate_model_photometry(p, data)
        chisq = np.sum(np.power((model - data.flux)/data.err,2))
        
        return chisq/len(data.flux)

    b = [(0.,1.),(0.,1.),(0.,1.),(0.,40.)] + data.free_bounds
    res = minimize(func, p0, args = (data,), bounds = b)
        
    return res.x, func(res.x,data)
    
