import pandas as pd

file = r"info_SNe_minimal.txt"

info = pd.read_csv(file,sep="\t", index_col=0)

def get_names():
    return list(info.index)


def find_in_names(name):
    if name in get_names():
        return name
    # here handling partial matches
    partial_matches = [n for n in get_names() if name in n or n in name]
    if len(partial_matches) == 1:
        return partial_matches[0]
    else:
        return name


def get_property(name, property_key, default_value):
    found_name = find_in_names(name)
    if found_name in get_names():
        value = info[property_key][found_name]
        print(f"  {property_key} from info", value)
        return value
    else:
        print(f"  {name} not available from info")
        return default_value


def get_z_red(name):
    return get_property(name, "z_red", 10)


def get_E_BV(name):
    return get_property(name, "E_BV", 0)
