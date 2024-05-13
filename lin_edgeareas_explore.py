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
landcovers = lem.import_landcovers(this_dir, version)


# %% Get derived information
importlib.reload(lem)

# Total forest area (from Lin's edgeareas files)
totalareas = edgeareas.groupby(["Year", "site"]).sum().drop(columns="edge").rename(columns={"sumarea": "forest_from_ea"})

# Total forest area (from landcovers)
totalareas = lem.get_site_lc_area("forest", totalareas, landcovers)

# Total area
site_area = landcovers.groupby(["Year", "site"]).sum()
totalareas = totalareas.assign(site=site_area.sumarea)
