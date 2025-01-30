"""
This module enables the automatic loading of redshift and ebv from external table
"""
import pandas as pd

INFO_FILE = r"info_SNe_minimal.txt"

info = pd.read_csv(INFO_FILE,sep="\t", index_col=0)

def get_names():
    """
    get available SN names in the file
    """
    return list(info.index)


def find_in_names(name):
    """
    find the name by partial matching it
    """
    if name in get_names():
        return name
    # here handling partial matches
    partial_matches = [n for n in get_names() if name in n or n in name]
    if len(partial_matches) == 1:
        return partial_matches[0]

    return name


def get_property(name, property_key, default_value):
    """
    read a generic property in case we may add more
    """
    found_name = find_in_names(name)
    if found_name in get_names():
        value = info[property_key][found_name]
        print(f"  {property_key} from info", value)
        return value

    print(f"  {name} not available from info")
    return default_value


def get_z_red(name):
    """
    get redshift
    """
    return get_property(name, "z_red", 10)


def get_E_BV(name):
    """
    get ebv correction
    """
    return get_property(name, "E_BV", 0)
