import numpy as np
import matplotlib.pyplot as plt

from os import listdir
from astropy.io import fits

#formatted for uds field right now

class Phot_Data():
    pass

def setup_data(CANDELS_id, z, a_v, sfr, free_lines, free_bounds = None):

    data = Phot_Data()

    data.wl, data.flux, data.err = read_CANDELS(CANDELS_id)
    data.z = z
    data.E_bv = a_v/4.05
    data.sfr = sfr

    data.free_lines = free_lines
    data.free_bounds = free_bounds

    data.filters = read_filters('filtercurves_uds/')
    data.filter_ids = np.array(range(len(data.filters)))
    data.bases = read_bases('fitsfiles/')
    data.emission = read_emission(free_lines)
    
    pos_flags = np.where(data.flux > 0)
    data.wl = data.wl[pos_flags]
    data.err = data.err[pos_flags]
    data.filters = data.filters[pos_flags]
    data.filter_ids = data.filter_ids[pos_flags]
    data.flux = data.flux[pos_flags]
    
    bad_flags = flag_bad_filters(data.filters, free_lines, z)
    data.wl = data.wl[bad_flags]
    data.err = data.err[bad_flags]
    data.filters = data.filters[bad_flags]
    data.filter_ids = data.filter_ids[bad_flags]
    data.flux = data.flux[bad_flags]
    
    data.filters = normalize_filters(data.filters)
    
    data.flux_vectors, data.continuum_norms = generate_continuum_phot(data.bases,data.filters,z)
    data.emission_vectors, data.free_vectors = generate_emission_phot(data.emission, free_lines, data.filters, z)

    lines = np.insert(data.free_lines,0,4861.)        
    data.continuum_means = calculate_mean_continuum(data,lines)
    
    return data

def read_CANDELS(catalog_id):

    hdulist = fits.open('CANDELS.UDS.F160W.v1.fits')
    tbldata = hdulist[1].data
    
    # goods-s formatting
    #wl = np.array([4301.20,5927.30,7704.20,8086.80,9061.30,9875.20,10584.0,12516.2,15392.3,21609.9,35612.3,45095.7])
    #flux_column_titles = ['ACS_F435W_FLUX','ACS_F606W_FLUX','ACS_F775W_FLUX','ACS_F814W_FLUX','ACS_F850LP_FLUX','WFC3_F098M_FLUX','WFC3_F105W_FLUX','WFC3_F125W_FLUX','WFC3_F160W_FLUX','HAWKI_KS_FLUX','IRAC_CH1_FLUX','IRAC_CH2_FLUX']
    #err_column_titles = ['ACS_F435W_FLUXERR','ACS_F606W_FLUXERR','ACS_F775W_FLUXERR','ACS_F814W_FLUXERR','ACS_F850LP_FLUXERR','WFC3_F098M_FLUXERR','WFC3_F105W_FLUXERR','WFC3_F125W_FLUXERR','WFC3_F160W_FLUXERR','ISAAC_KS_FLUXERR','IRAC_CH1_FLUXERR','IRAC_CH2_FLUXERR']
    
    # uds formatting
    wl = np.array([4458.32,5927.30,8086.80,9036.88,12516.2,15392.3,21460,35612.3,45095.7])
    flux_column_titles = np.array(hdulist[1].columns.names)[[8, 18, 20, 16, 22, 24, 28, 36, 38]]
    err_column_titles = np.array(hdulist[1].columns.names)[[9, 19, 21, 17, 23, 25, 29, 37, 39]]
    
    fluxes = [None]*len(flux_column_titles)
    errors = [None]*len(err_column_titles)
    for i in range(len(flux_column_titles)):
        fluxes[i] = tbldata[flux_column_titles[i]]
        errors[i] = tbldata[err_column_titles[i]]

    flux = np.zeros(len(fluxes))
    err = np.zeros(len(errors))
    for i in range(len(fluxes)):
        flux[i] = (fluxes[i][catalog_id - 1])
        err[i] = (errors[i][catalog_id - 1])

    for i in range(len(flux)):
        flux[i] *= 3e-9/(wl[i]**2)
        err[i] *= 3e-9/(wl[i]**2)
                   
    return wl, flux, err

# Input files are two columns: wl, transmission
def read_filters(path):

    files = sorted(listdir(path))
    filter_curves = [None]*len(files)
    
    for i,f in enumerate(files):
        dat = np.loadtxt(path + f, usecols = (0,1))
        filter_curves[i] = dat
        
    return np.array(filter_curves)

# Input file formats are PEGASE outputs
def read_bases(path):

    files = sorted(listdir(path))
    spectra = [None]*len(files)

    for i,s in enumerate(files):
        hdulist = fits.open(path + s)
        w = hdulist[3].data['BFIT']
        s = hdulist[0].data[0]
        spectra[i] = [w,s]    
    
    return np.array(spectra)

# Temporary input will change to be more general later
def read_emission(free_lines):

    f = 'fitsfiles/PEGASE_continuum_starburstC.fits'
    hdulist = fits.open(f)
    lines = hdulist[1].data

    hb_flag = [i for i, tup in enumerate(lines) if tup[0] == 4861]
    norm = 1./lines[hb_flag][0][1]

    free_lines_flag = [i for i, tup in enumerate(lines) if tup[0] in free_lines]    
    lines = np.delete(lines,free_lines_flag)

    for l in lines:
        l[1] *= norm
    
    return lines

def normalize_filters(filters):

    norms = [None]*len(filters)
    normalized_filters = filters
    
    for i,dat in enumerate(filters):
        norms[i] = 1./(np.trapz(dat[:,1],dat[:,0]))
        normalized_filters[i][:,1] = norms[i]*dat[:,1]
    
    return normalized_filters

def generate_continuum_phot(bases, filters, z):

    vectors = np.zeros((len(bases),len(filters)))
    total_fluxes = np.zeros((len(bases),))
    
    for i,b in enumerate(bases):
        photometry = calculate_photometry(b, filters, z)
        total_fluxes[i] = np.sum(photometry)
        vectors[i] = photometry
        
    return vectors, total_fluxes

def generate_emission_phot(emission, free_lines, filters, z):

    clumped_vectors = np.zeros((len(filters),))
    throughputs = np.zeros((len(filters),len(emission)))

    for i,f in enumerate(filters):
        for j,line in enumerate(emission):
            transmission = np.interp(line[0]*(1 + z),f[:,0],f[:,1])
            flux = transmission * line[1]
            clumped_vectors[i] += flux

    free_vectors = np.zeros((len(free_lines), len(filters)))

    for i,f in enumerate(filters):
        for j,wl in enumerate(free_lines):
            transmission = np.interp(wl*(1 + z),f[:,0],f[:,1])
            free_vectors[j,i] = transmission
            
    return clumped_vectors, free_vectors

def flag_bad_filters(filters, free_lines, z):

    flags = [True]*len(filters)

    for i,f in enumerate(filters):
        for wl in free_lines:
            line_throughput = np.interp(wl*(1 + z),f[:,0],f[:,1])

            if line_throughput > 0.005 and line_throughput < .1:
                flags[i] = False

    return flags    

def calculate_photometry(spectra, filters, z):

    photometry = [None]*len(filters)

    for i,f in enumerate(filters):
        interp = np.interp(f[:,0], spectra[0] * (1 + z), spectra[1])
        transmitted_flux = f[:,1] * interp
        total_flux = np.trapz(transmitted_flux, x = f[:,0])
        photometry[i] = total_flux
            
    return np.array(photometry)

def calculate_mean_continuum(data, wavelengths):

    means = np.zeros((len(wavelengths),len(data.bases)))

    for i,wl in enumerate(wavelengths):
        points = [wl - 50, wl - 25, wl, wl + 25, wl + 50]
        for j,b in enumerate(data.bases):
            fluxes = np.interp(points,b[0],b[1])
            mean = np.mean(fluxes)
            means[i,j] = mean
            
    return means

if __name__ == '__main__':

    #fl = np.array([1216,3727,5007])
    #dummy_data = setup_data(2644,2.12,.3, fl)

    #guess = [0.008,0.,.93,1.,1.,1.,1.]
    #fit = ppf.calculate_model_photometry(np.array(guess),dummy_data)
    #chisq = ppf.calculate_model_likelihood(dummy_data.flux,dummy_data)

    #print(chisq)
    print('Compiled!')
