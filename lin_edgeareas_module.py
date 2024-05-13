import os
import numpy as np
import matplotlib.pyplot as plt
import importlib
import pandas as pd

def get_site_lc_area(lc, totalareas, landcovers):
    lc_area = landcovers[landcovers["is_" + lc]].groupby(["Year", "site"]).sum()
    lc_area = lc_area.rename(columns={"sumarea": lc})
    totalareas = totalareas.join(lc_area[lc])
    totalareas = totalareas.fillna(value=0)
    return totalareas

def get_version_info(version):
    vinfo = {}
    if version == "20240506":
        vinfo["Nsites"] = 4
    else:
        raise RuntimeError(f"Version {version} not recognized")
    return vinfo

def import_landcovers(this_dir, version):
    # Import legend
    landcovers_legend = pd.read_csv(os.path.join(this_dir, "MAPBIOMAS_Col6_Legenda_Cores.simple.csv"))
    
    # Import landcovers
    filename_template = os.path.join(this_dir, "inout", version, f"Landcover_clean_%d.csv")
    landcovers = read_combine_multiple_csvs(filename_template, version)
    landcovers = landcovers.rename(columns={"landcover": "landcover_num"})
    
    # Add labels
    landcovers = landcovers.assign(tmp = landcovers.landcover_num)
    landcovers = landcovers.set_index("tmp").join(landcovers_legend.set_index("landcover_num"))
    
    # Regenerate index
    index = np.arange(len(landcovers.index))
    landcovers = landcovers.set_index(index)
    
    # Add legend-based info
    is_water = []
    is_forest = []
    is_fforest = []
    is_crop = []
    is_pasture = []
    is_croppast = []
    is_agri = []
    is_unvegd = []
    is_vegd = []
    is_unobs = []
    for i, num in enumerate(landcovers.landcover_num):
        
        # Get landcover string for this landcover code
        matching_landcovers = landcovers_legend[landcovers_legend.landcover_num == num]
        Nmatches = matching_landcovers.shape[0]
        if Nmatches != 1:
            raise RuntimeError(f"Expected 1 landcover matching {num}; found {Nmatches}")
        landcover_str = matching_landcovers.landcover_str.values[0]
        
        # Classify based on landcover string
        is_water.append("#5" in landcover_str)
        is_forest.append("#1" in landcover_str)
        is_fforest.append("#1.1" in landcover_str)
        is_crop.append("#3.2" in landcover_str)
        is_pasture.append("#3.1" in landcover_str)
        is_croppast.append(is_crop[i] or is_pasture[i] or "3.4" in landcover_str)
        is_agri.append("#3" in landcover_str)
        is_unvegd.append("#4" in landcover_str)
        is_vegd.append(not is_water[i] and not is_unvegd[i])
        is_unobs.append("#6" in landcover_str)

    landcovers = landcovers.assign(
        is_water=is_water,
        is_forest=is_forest,
        is_fforest=is_fforest,
        is_crop=is_crop,
        is_pasture=is_pasture,
        is_croppast=is_croppast,
        is_agri=is_agri,
        is_unvegd=is_unvegd,
        is_vegd=is_vegd,
        is_unobs=is_unobs,
        )
    
    return landcovers


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