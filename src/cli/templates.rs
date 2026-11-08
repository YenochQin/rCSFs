//! Commented examples; these do not activate a calculation until edited.
pub(super) fn template(command: &str) -> Option<&'static str> {
    Some(match command {
        "csfsgenerate" => {
            r###"# [csfsgenerate]
# as = 6
# orbital_order = "*"
# inactive_core = 0
# reference_configuration = ["1s(2,*)"]
# active_space = "2s"
# j_min = 0
# j_max = 0
# excitations = 0
# rcsfs_out = "generated.c"
# generate_parquet = false
# generate_descriptors = false
# rcsfs_parquet = "generated.parquet"
# descriptor = "generated_descriptors.parquet"  # Optional: defaults to {conf}_desc.parquet with conf.
# threads = 8
# memory_budget_mib = 1024
# estimate_only = false
# allow_unchecked_space = false
# generation_storage = "disk"
# continue_lists = false
"###
        }
        "gen-descriptors" => {
            r###"# [gen-descriptors]
# input_parquet = "generated.parquet"
# output_parquet = "generated_descriptors.parquet"
# header = "generated_header.toml"
# num_workers = 8
# descriptor_version = 2
# compression = "zstd"
# json = false
"###
        }
        "zero-first" => {
            r###"# [zero-first]
# zero_csf = "zero.c"
# full_csf = "generated.c"
# output_csf = "zero_first.c"
# num_workers = 8
# max_line_len = 256
# json = false
"###
        }
        "csfs-split" => {
            r###"# [csfs-split]
# split_csfs_parquet = "generated.parquet"
# csfs_header = "generated_header.toml"
# json = false
# [csfs-split.active_spaces]
# AS0 = "1s"
# AS1 = "2s"
# AS2 = "3s"
"###
        }
        "jsplit" => {
            r###"# [jsplit]
# input_csf = "input.c"
# prefix = "split"
# copy_w = true
# overwrite = true
# json = false
"###
        }
        "interacting" => {
            r###"# [interacting]
# reference = "reference.c"
# candidates = "generated.c"
# output = "interacting.c"
# hamiltonian = "dc"
# method = "structural-upper-bound"
# num_workers = 8
# json = false
"###
        }
        "restore-csfs" => {
            r###"# [restore-csfs]
# descriptors = "generated_descriptors.parquet"
# header = "generated_header.toml"
# output = "restored.c"
# indices = [0, 1]
# json = false
"###
        }
        "convert" => {
            r###"# [convert]
# input_csf = "input.c"
# output_parquet = "output.parquet"
# num_workers = 8
# json = false
"###
        }
        "info" => {
            r###"# [info]
# input_parquet = "output.parquet"
# json = false
"###
        }
        "csfggenerate" => {
            r###"# [csfggenerate]
# inactive_core = 0
# reference_configuration = ["1s(2,*)"]
# labeling_space = "1s"
# active_space = "2s"
# j_min = 0
# j_max = 0
# excitations = 2
# rcsfg_out = "rcsfg.out"
# rlabel_out = "rlabel.out"
# threads = 8
# overwrite = false
"###
        }
        _ => return None,
    })
}
