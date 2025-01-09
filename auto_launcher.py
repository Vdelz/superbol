import argparse

def get_params():
    parser = argparse.ArgumentParser(description="example usage: superbol.py -sn SN1987A")
    parser.add_argument("-sn", "--sn", type=str, help="Name of the Supernova REQUIRED", required=True)
    parser.add_argument("-ff", "--findfiles", type=str, default="y", help="Find input files automatically")
    parser.add_argument("-u", "--use", type=str, default="n", help="Use interpolated LC")
    parser.add_argument("-l", "--limitMJDs", type=str, default="n", help="Limit time range to use")
    parser.add_argument("-b", "--bands", type=str, default="", help="Enter bands to use (blue to red)")
    parser.add_argument("-ref", "--ref", type=str, default="", help="Choose reference band(s) for sampling epochs")
    parser.add_argument("-fm", "--findmax", type=str, default="n", help="Interactively find maximum")
    parser.add_argument("-z", "--z", type=float, default=None, help="enter SN redshift or distance modulus") # from file
    parser.add_argument("-i", "--ilc", type=str, default="ask", help="Interpolate light curves interactively")
    parser.add_argument("-a", "--algo", type=str, default="ask", help="Chose type of algorithm to fit")
    parser.add_argument("-gpy", "--gpy", type=str, default="y", help="y: Use GPy or n: Use Sklearn")
    parser.add_argument("-k", "--kernel", type=str, default="all", help="Chose kernels")
    parser.add_argument("-kp", "--kerpar", type=str, default="y", help="Go with default Kernel Params")
    parser.add_argument("-happy", "--happy", type=str, default="ask", help="Chose if you are happy with fit by default")
    parser.add_argument("-ord", "--order", type=int, default=4, help="Order of polynomial to fit")
    parser.add_argument("-ete", "--ete", type=str, default="p", help="Early-time extrapolation")
    parser.add_argument("-lte", "--lte", type=str, default="c", help="Late-time extrapolation")
    parser.add_argument("-ebv", "--ebv", type=float, default=None, help="Extinction correction") # from file
    parser.add_argument("-ds", "--defsys", type=str, default="y", help="Are all bands in their default systems")
    parser.add_argument("-luv", "--luv", type=str, default="c", help="Apply blackbody absorption L_uv(lam)")
    parser.add_argument("-t0", "--t0", type=float, default=10000, help="Initial guess for temperature in K")
    parser.add_argument("-r0", "--r0", type=float, default=1.0e15, help="Initial guess for starting radius in cm")

    return parser.parse_args()


def input_param(query,param="default"):
    if param == "default":
        if "]" not in query or "[" not in query:
            param = "ask"
        else:
            param = query.split("[")[-1].split("]")[0]
    if param == "ask":
        return input(query)
    else:
        print(query,param)
        return param
