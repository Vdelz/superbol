import GPy
import numpy as np

def apply_gaussian_process_gpy(times, luminosities, new_times, kernel_choices=None, kernel_params_list=None, correction=None):
    """
    Estrapola e corregge le curve bolometriche usando Gaussian Process con uno o più kernel di GPy.

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


    if correction is not None:
        correction = np.array(correction).flatten()
        if len(correction) != len(new_times):
            raise ValueError("Correction array must have the same length as new_times.")
        corrected_predictions = predictions.flatten() + correction
    else:
        corrected_predictions = predictions.flatten()

    return corrected_predictions, predictions.flatten(), uncertainties.flatten()
