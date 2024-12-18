import pandas as pd

file = r"info_SNe_minimal.txt"

info = pd.read_csv(file,sep="\t", index_col=0)

def get_names():
    return list(info.index)

def get_z_red(name):
    return info.z_red[name]

def get_E_BV(name):
    return info.E_BV[name]
