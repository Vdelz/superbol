"""
This module enpowers Matt Nicholl's Superbol algorithm
by enabling the users to use the Gaussian Process
to perform light curves interpolation. Enjoy!
"""

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, DotProduct
import matplotlib.pyplot as plt
import numpy as np



GPY_AVAILABLE = False
#try:
#    import GPy
#    GPY_AVAILABLE = True
#except ImportError as err:
#    pass


def get_kernels(launch):
    """
    Returns a dictionary of available kernels
    We can make this list as long as we want the rest will update automatically
    """
    if launch.gpy=="y" and GPY_AVAILABLE:
        kernels = {
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
        "Matern (v = 3/2)": Matern(nu=3/2),
        "Matern (v = 5/2)": Matern(nu=5/2),
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
        return default_kernel
    choices_str = input(request).split(",")
    if len(choices_str) == 0:
        return default_kernel
    valid_ids = [ker_id for ker_id in choices_str if ker_id in possible_ids]
    # Gets the kernel names from ids
    kernel_choices = [list(kernels.keys())[int(ker_id) - 1] for ker_id in valid_ids]
    if len(kernel_choices) == 0:
        print("  Warning: your selection [" + str(choices_str) + "] is invalid.")
        print("    Using default kernel")
        return default_kernel
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


def apply_gp(times, luminosities, new_times, kernels_custom, launch):
    """
    Args:
        times (array-like): dates of the in-band observation in MJD.
        luminosities (array-like): magnitudes measured on each date.
        new_times (array-like): dates of the reference band in MJD.
        kernels_custom (list) list of customized kernels:.
    Returns:
        tuple: LC predicted magnitudes, dev std of LC predictions.
    """
    # reshapes the inputs just in case
    times = np.array(times).reshape(-1, 1)
    luminosities = np.array(luminosities).reshape(-1, 1)
    new_times = np.array(new_times).reshape(-1, 1)
    t_plot = np.arange(np.min(new_times),np.max(new_times),1).reshape(-1, 1)
    # initialize, trains the regressor and interpolates
    if launch.gpy=="y" and GPY_AVAILABLE:
        combo_kernel = GPy.kern.src.add.Add(kernels_custom)
        gpr = GPy.models.GPRegression(times, luminosities, combo_kernel)
        gpr.optimize(messages=False)
        lc_predictions, lc_pred_var = gpr.predict(new_times)
        lc_pred_err = np.sqrt(lc_pred_var)
        lc_plot, lc_plot_var = gpr.predict(t_plot)
        lc_plot_err = np.sqrt(lc_plot_var)
    else: # use scikit learn
        combo_kernel = None
        for ker in kernels_custom:
            combo_kernel = ker if combo_kernel is None else combo_kernel + ker
        gpr = GaussianProcessRegressor(kernel=combo_kernel, random_state=0)
        gpr.fit(times, luminosities)
        lc_predictions, lc_pred_err = gpr.predict(new_times, return_std=True)
        lc_plot, lc_plot_err = gpr.predict(t_plot, return_std=True)
    lc_plot, lc_plot_err = lc_plot.flatten(), lc_plot_err.flatten()
    plt.fill_between( t_plot.flatten(), lc_plot - lc_plot_err, lc_plot + lc_plot_err,
        color="k", alpha=0.1, label="GP err", )
    return lc_predictions.flatten(), lc_pred_err.flatten()


def estimate_error_gp(times, lc_val, kernels_custom, launch):
    """
    This function enables to estimate the Gaussian process error
    by performing a 2fold cross validation.
    The 2 subsets of points are the even and the odds instead of random
    """
    # reshapes the inputs just in case
    times = np.array(times).reshape(-1)
    lc_val = np.array(lc_val).reshape(-1)

    twofold_cross_valid_err = 0
    mean_estimated_err = 0
    max_twofold_err = 0
    max_estimated_err = 0
    argmax_est = np.nan
    argmax_eff = np.nan

    for block in range(2):
        t_train,t_val = times[block::2],times[1-block::2]
        l_train,l_val_true = lc_val[block::2],lc_val[1-block::2]
        l_val_pred, est_errs = apply_gp(t_train, l_train, t_val, kernels_custom, launch)
        effective_errs = np.sqrt((l_val_pred-l_val_true)**2)

        twofold_cross_valid_err += np.sum(effective_errs)
        mean_estimated_err += np.sum(est_errs)
        max_twofold_err = max(max_twofold_err,np.max(effective_errs))
        max_estimated_err = max(max_estimated_err,np.max(est_errs))
        if max_twofold_err == np.max(effective_errs):
            argmax_eff = t_val[np.argmax(effective_errs)]
        if max_estimated_err == np.max(est_errs):
            argmax_est = t_val[np.argmax(est_errs)]

    twofold_cross_valid_err = twofold_cross_valid_err/len(times)
    mean_estimated_err = mean_estimated_err/len(times)
    print("\n  Interpolation error estimates performing half decimation")
    print("    Mean 2fold_cross_valid_error = ",twofold_cross_valid_err,"magnitudes")
    print("    Max  2fold_cross_valid_error = ",max_twofold_err,"magnitudes","at",argmax_eff)
    print("    Mean gp estimated_error      = ",mean_estimated_err,"magnitudes")
    print("    Max  gp estimated_error      = ",max_estimated_err,"magnitudes","at",argmax_est)


def select_do_gp(times, lc_val, new_times, launch):
    """
    This function is the GP "main function caller".
    Enables to select customize the kernels and run the GP
    """
    kernel_choices = select_kernels(launch)
    kernels_custom = select_params(kernel_choices,launch)
    preds, errs = apply_gp(times, lc_val, new_times, kernels_custom, launch)
    estimate_error_gp(times, lc_val, kernels_custom, launch)
    return preds, errs


def gp_interpolate(lc_orig, lc_int, ref_stack, i, cols, launch):
    """
    This is a wrapper around the GP process that binds with superbol code
    """
    try:
        # Pre filters data by eliminating nan values
        valid = ~np.isnan(lc_orig[i][:, 1])
        times = lc_orig[i][valid, 0]
        lc_val = lc_orig[i][valid, 1]
        new_times = ref_stack[:, 0]
        # Core of the Gaussian Process
        preds, errs = select_do_gp(times, lc_val, new_times, launch)
        # Put values where is needed and plots them
        lc_int[i] = np.column_stack((new_times, preds, errs))
        # Plots all teh other bands
        for j in lc_orig.keys():
            plt.plot(lc_orig[j][:, 0], lc_orig[j][:, 1], "o",
                    color=cols[j], label=j)
        # plots the fitted band with error bars and variance area
        plt.errorbar( new_times, preds, errs,
            fmt="x", color=cols[i], label=i + " GP fit" )
        plt.gca().invert_yaxis()
    except Exception as exc:  # pylint: disable=broad-except
        print(f"  Error in Gaussian Process fit: {exc}")
