import os
import numpy as np
import matplotlib.pyplot as plt
import importlib
import pandas as pd

def get_version_info(version):
    vinfo = {}
    if version == "20240506":
        vinfo["Nsites"] = 4
    else:
        raise RuntimeError(f"Version {version} not recognized")
    return vinfo

def read_combine_multiple_csvs(filename_template, version):
    vinfo = get_version_info(version)
    df_combined = []
    for f in 1 + np.arange(vinfo["Nsites"]):
        
        # Read site's .csv
        filename = filename_template % f
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File not found: {filename}")
        df = pd.read_csv(filename)
        
        # Add column for site
        df = df.assign(site=f)
        
        # Append to edgeareas df
        df_combined.append(df)

    df_combined = pd.concat(df_combined)
    return df_combined