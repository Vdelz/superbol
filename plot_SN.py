import matplotlib.pyplot as plt
import numpy as np

  # plot_supernova(wlref,cols,filters,fluxes,ref_stack
def plot_supernova(wavelenghts,colors,bands,fluxes,times):
    
    ax = plt.figure(666).add_subplot(projection='3d')
    
    for wl,band,flux in zip(wavelenghts,bands,fluxes.T):
        #print(np.array(times).shape,np.array(wl).shape,np.array(flux).shape)
        ok = ax.scatter(times,np.ones_like(times)*wl,flux,c=colors[band], label=band)
        ok = ax.plot(times,np.ones_like(times)*wl,flux,c="k")
    
    for time, plankian in zip(times,fluxes):
        #print(np.array(wavelenghts).shape,np.array(plankian).shape)
        ok = ax.plot(np.ones_like(plankian)*time,wavelenghts,plankian,c="gray")
    
    if np.mean(times)<1000:
        ax.set_xlabel('MJD from peak')
    else:
        ax.set_xlabel('Julian Date')
    ax.set_ylabel('WaveLenght in Armstrong')
    ax.set_zlabel('Normalized Flux')
    
    ok = plt.show()
    input("")