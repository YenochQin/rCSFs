# %% cell 1
import graspkit as gk
import numpy as np
import polars as pl

from rcsfs import (
    convert_csfs_parallel,
    export_descriptors_with_polars_parallel,
    generate_descriptors_from_parquet_parallel,
    read_peel_subshells,
)

# %%
raw_csfs_path = "/Users/yiqin/Documents/PythonProjects/as3_odd4/cv4odd1as3_odd4_1.c"
# %%
raw_csfs = gk.GraspFileLoad.from_filepath(raw_csfs_path, "CSF").get_csfs_data()

# %%
rout_path = "/Users/yiqin/Documents/PythonProjects/as3_odd4/cv4odd1as3_odd4_1.parquet"
rust_load = convert_csfs_parallel(raw_csfs_path, rout_path)

# %%
rout_path = "/Users/yiqin/Documents/PythonProjects/as3_odd4/cv4odd1as3_odd4_1.parquet"
# convert_csfs = pl.read_parquet(rout_path)

# %%
header_file_path = (
    "/Users/yiqin/Documents/PythonProjects/as3_odd4/cv4odd1as3_odd4_1_header.toml"
)
peel_subshell = read_peel_subshells(header_file_path)

rout_desc_path = (
    "/Users/yiqin/Documents/PythonProjects/as3_odd4/cv4odd1as3_odd4_1_desc.parquet"
)
gen_result = export_descriptors_with_polars_parallel(
    rout_path, rout_desc_path, peel_subshell
)
# %%
#
gen_result

# %%
py_desc = np.load(
    "/Users/yiqin/Documents/PythonProjects/as3_odd4/cv4odd1as3_odd4_1_desc.npy"
)
# %%
rcsfs_desc = pl.read_parquet(rout_desc_path)
# %%
rcsfs_desc_np = rcsfs_desc.to_numpy()
np.array_equal(py_desc, rcsfs_desc_np)
