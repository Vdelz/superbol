"""
This module allows to skip the interrupting questions
along the original Superbol code by loading the parameters externally at launch
"""
import argparse

def get_params():
    """
    This function allows to load the answers to the questions
    by parsing the parameters given externally at launch
    """
    args = argparse.ArgumentParser(description="example usage: superbol.py -sn SN1987A")
    args.add_argument("-sn", "--sn", type=str, help="Name of the Supernova REQUIRED", required=True)
    args.add_argument("-ff", "--findfiles", type=str, default="y", help="Find inputs automatically")
    args.add_argument("-u", "--use", type=str, default="n", help="Use interpolated LC")
    args.add_argument("-l", "--limitMJDs", type=str, default="n", help="Limit time range to use")
    args.add_argument("-b", "--bands", type=str, default="", help="Bands to use (blue to red)")
    args.add_argument("-ref", "--ref", type=str, default="", help="Ref band for sampling epochs")
    args.add_argument("-fm", "--findmax", type=str, default="n", help="Interactively find maximum")
    args.add_argument("-z", "--z", type=float, default=None, help="SN redshift or distance modulus")
    args.add_argument("-i", "--ilc", type=str, default="y", help="Interpolate LCs interactively")
    args.add_argument("-a", "--algo", type=str, default="ask", help="Chose algorithm to fit")
    args.add_argument("-gpy", "--gpy", type=str, default="y", help="Use y for GPy or n for Sklearn")
    args.add_argument("-k", "--kernel", type=str, default="all", help="Chose kernels")
    args.add_argument("-kp", "--kerpar", type=str, default="y", help="Use default Kernel Params")
    args.add_argument("-happy", "--happy", type=str, default="ask", help="Happy with fit")
    args.add_argument("-ord", "--order", type=int, default=4, help="Order of polynomial fit")
    args.add_argument("-ete", "--ete", type=str, default="p", help="Early-time extrapolation")
    args.add_argument("-lte", "--lte", type=str, default="c", help="Late-time extrapolation")
    args.add_argument("-ebv", "--ebv", type=float, default=None, help="Extinction correction")
    args.add_argument("-ds", "--defsys", type=str, default="y", help="All bands in default systems")
    args.add_argument("-luv", "--luv", type=str, default="c", help="Blackbody absorption L_uv")
    args.add_argument("-t0", "--t0", type=float, default=10000, help="Initial guess for T in K")
    args.add_argument("-r0", "--r0", type=float, default=1.0e15, help="Initial guess for R in cm")

    return args.parse_args()


def input_param(query,param="default"):
    """
    This function replaces all the input() functions in the original Superbol code
    by getting the input from the arguments list loaded above
    """
    if param == "default":
        if "]" not in query or "[" not in query:
            param = "ask"
        else:
            param = query.split("[")[-1].split("]")[0]
    if param in ("ask" , 0):
        return input(query)
    print(query,param)
    return param
