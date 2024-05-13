# %% Setup

import pandas as pd
import importlib
import lin_edgeareas_module as lem

this_dir = "/Users/samrabin/Library/CloudStorage/Dropbox/2023_NCAR/FATES escaped fire/Lin_edgeareas"
version = "20240506"


# %% Import data

importlib.reload(lem)

# Import edge areas
filename_template = os.path.join(this_dir, "inout", version, f"Edgearea_clean_%d.csv")
edgeareas = lem.read_combine_multiple_csvs(filename_template, version)

# Import land covers
filename_template = os.path.join(this_dir, "inout", version, f"Landcover_clean_%d.csv")
landcovers = lem.read_combine_multiple_csvs(filename_template, version)
