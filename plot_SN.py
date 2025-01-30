"""
This module is meant to improve the visualization of the supernova
by providing custom external plotting functions
"""
import matplotlib.pyplot as plt
import numpy as np

# plot_supernova(wlref,cols,filters,fluxes,ref_stack
def plot_supernova(wavelenghts,colors,bands,fluxes,times):
    """
    THis function saves a 3d plot of the supernova
    """

    axs = plt.figure(4).add_subplot(projection='3d')

    for wls,band,flux in zip(wavelenghts,bands,fluxes.T):
        #print(np.array(times).shape,np.array(wl).shape,np.array(flux).shape)
        _ok = axs.scatter(times,np.ones_like(times)*wls,flux,c=colors[band], label=band)
        _ok = axs.plot(times,np.ones_like(times)*wls,flux,c="k")

    for time, plankian in zip(times,fluxes):
        #print(np.array(wavelenghts).shape,np.array(plankian).shape)
        _ok = axs.plot(np.ones_like(plankian)*time,wavelenghts,plankian,c="gray")

    if np.mean(times)<1000:
        axs.set_xlabel('MJD from peak')
    else:
        axs.set_xlabel('Julian Date')
    axs.set_ylabel('WaveLenght in Armstrong')
    axs.set_zlabel('Normalized Flux')

    _ok = plt.show()
