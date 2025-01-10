<<<<<<< Updated upstream
import GPy
=======
"""
This module enpowers Matt Nicholl's Superbol algorithm
by enabling the users to use the Gaussian Process
to perform light curves interpolation. Enjoy!
"""

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, DotProduct
import matplotlib.pyplot as plt
import traceback
>>>>>>> Stashed changes
import numpy as np

def apply_gaussian_process_gpy(times, luminosities, new_times, kernel_choices=None, kernel_params_list=None, correction=None):
    """
<<<<<<< Updated upstream
    Estrapola e corregge le curve bolometriche usando Gaussian Process con uno o più kernel di GPy.

=======
    Returns a dictionary of available kernels
    We can make this list as long as we want the rest will update automatically
    """
    if launch.gpy=="y" and gpy_available:
        kernels = {
        #"matern 1/2": lambda input_dim, **params: GPy.kern.Exponential(input_dim=input_dim, **params),
        "Matern32": GPy.kern.Matern32(input_dim=1),
        "Matern52": GPy.kern.Matern52(input_dim=1),
        "RBF": GPy.kern.RBF(input_dim=1),
        "Exponential": GPy.kern.Exponential(input_dim=1),
        "RationalQuadratic": GPy.kern.RatQuad(input_dim=1),
        "Linear":GPy.kern.Linear(input_dim=1),
        "Bias":GPy.kern.Bias(input_dim=1),
    }
    else: # use scikit learn
        kernels = {
        "Matern12": Matern(nu=1/2), 
        "Matern32": Matern(nu=3/2),
        "Matern52": Matern(nu=5/2),
        "RBF": RBF(),
        "White": WhiteKernel(),
        "Constant": ConstantKernel(),
        "DotProduct": DotProduct(),
        "RationalQuadratic": RationalQuadratic(),
        }
    return kernels


def select_kernels(launch):
    """
    This function enables to select the kernels interactively
    """
    print("\n  Choose one or more kernels for Gaussian Process fit")
    kernels = get_kernels(launch)
    default_kernel = list(kernels.keys())  # [list(kernels.keys())[0]]
    possible_ids = [str(ker_id + 1) for ker_id in range(len(kernels))]
    # Print all the possibilities
    for ker_id, k in enumerate(kernels.keys()):
        print("    " + str(ker_id + 1) + ":", k)
    # Parse your choices
    request = "  Enter kernel indices separated by commas (i.e. [2,3]): "
    choices_str = launch.kernel
    if choices_str == "all":
        print(f"    Using default kernel: all kernels")
        return default_kernel
    choices_str = input(request).split(",")
    if len(choices_str) == 0:
        default_kernel= ["Matern32", "Matern52"]
        return default_kernel
    valid_ids = [ker_id for ker_id in choices_str if ker_id in possible_ids]
    # Gets the kernel names from ids
    kernel_choices = [list(kernels.keys())[int(ker_id) - 1] for ker_id in valid_ids]
    if len(kernel_choices) == 0:
        default_kernel = ["Matern32", "Matern52"]
        print("  Warning: your selection [" + str(choices_str) + "] is invalid.")
        print(f"    Using default kernel:{default_kernel}")
        return default_kernel
    
    print(" Using choosen kernels" , kernel_choices)
    return kernel_choices

def select_params(kernel_choices, launch):
    """
    This function enables to tune the kernel parameters for each kernel
    parameters are taken automatically from the available ones for each specific kernel
    """
    kernels = get_kernels(launch)
    kernels_custom = []
    go_with_default = launch.kerpar
    if go_with_default != "y":
        go_with_default = input("\n  Use Default Kernel parameters? [y] ") or "y"
    for kernel in [kernels[k] for k in kernel_choices]:
        if go_with_default != "y":
            print("\n  Changing default parameters for kernel", kernel)
            for param, val in kernel.__dict__.items():
                cust_v = (input("    parameter " + param + " [" + str(val) + "] ") or val)
                setattr(kernel, param, type(val)(cust_v))
        kernels_custom.append(kernel)
    return kernels_custom


def apply_gaussian_process(times, luminosities, new_times, kernels_custom, launch):
    """
>>>>>>> Stashed changes
    Args:
        times (array-like): Tempi osservati (1D).
        luminosities (array-like): Luminosità associate ai tempi (1D).
        new_times (array-like): Nuovi tempi per l'estrapolazione (1D).
        kernel_choices (list of str, optional): Lista dei kernel scelti (es. ["Matern32", "Matern52"]).
                                                Predefiniti: ["Matern32", "Matern52"].
        kernel_params_list (list of dict, optional): Lista di parametri per ciascun kernel. Se None, usa parametri predefiniti.
        correction (array-like, optional): Trend della banda di riferimento (1D) da usare come correzione sulle predizioni.

    Returns:
        tuple: Predizioni corrette, predizioni originali, e incertezze (std).
    """
    # Assicura numpy array in forma corretta
    times = np.array(times).reshape(-1, 1)
    luminosities = np.array(luminosities).reshape(-1, 1)
    new_times = np.array(new_times).reshape(-1, 1)

    # Dizionario dei kernel supportati
    kernels = {
        "Matern12": lambda input_dim, **params: GPy.kern.Exponential(input_dim=input_dim, **params),
        "Matern32": GPy.kern.Matern32,
        "Matern52": GPy.kern.Matern52,
        "RBF": GPy.kern.RBF,
        "Exponential": GPy.kern.Exponential,
        "RationalQuadratic": GPy.kern.RatQuad,
        "Periodic": GPy.kern.PeriodicExponential,
        "Brownian": lambda input_dim, **params: GPy.kern.Bias(input_dim=input_dim, **params)
    }

    if kernel_choices is None:
        kernel_choices = ["Matern32", "Matern52"]

    if kernel_params_list is None:
        kernel_params_list = [{"lengthscale": 10.0, "variance": 1.0} for _ in kernel_choices]


    combined_kernel = None
    for choice, params in zip(kernel_choices, kernel_params_list):
        if choice not in kernels:
            raise ValueError(f"Kernel '{choice}' not supported. Choose between {', '.join(kernels.keys())}")
        current_kernel = kernels[choice](input_dim=1, **params)
        combined_kernel = current_kernel if combined_kernel is None else combined_kernel + current_kernel

    model = GPy.models.GPRegression(times, luminosities, combined_kernel)

    model.optimize(messages=False)

    predictions, variances = model.predict(new_times)
    uncertainties = np.sqrt(variances)


<<<<<<< Updated upstream
    if correction is not None:
        correction = np.array(correction).flatten()
        if len(correction) != len(new_times):
            raise ValueError("Correction array must have the same length as new_times.")
        corrected_predictions = predictions.flatten() + correction
    else:
        corrected_predictions = predictions.flatten()

    return corrected_predictions, predictions.flatten(), uncertainties.flatten()
=======
def gp_interpolate(lc, lc_int, ref_stack, i, cols, launch):
    """
    This is a wrapper around the GP process that binds with superbol code
    """
    try:
        # Pre filters data by eliminating nan values
        valid = ~np.isnan(lc[i][:, 1])
        times = lc[i][valid, 0]
        lc_val = lc[i][valid, 1]
        new_times = ref_stack[:, 0]
        # Core of the Gaussian Process
        preds, errs = select_do_gp(times, lc_val, new_times, launch)
        # Put values where is needed and plots them
        lc_int[i] = np.column_stack((new_times, preds, errs))
        # Plots all teh other bands
        for j in lc.keys():
            plt.plot(lc[j][:, 0], lc[j][:, 1], "o", color=cols[j], label=j)
        # plots the fitted band with error bars and variance area
        plt.errorbar( new_times, preds, errs,
            fmt="x", color=cols[i], label=i + " GP fit" )
        plt.gca().invert_yaxis()
    except Exception as exc:  # pylint: disable=broad-except
        traceback.print_exc() 
        print(f"  Error in Gaussian Process fit: {exc}")

>>>>>>> Stashed changes
