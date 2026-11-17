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
# output = "generated_descriptors.parquet"  # With conf/as, inferred automatically.
# threads = 8
# memory_budget_mib = 1024
# estimate_only = false
# allow_unchecked_space = false
# generation_storage = "disk"
# continue_lists = false
"###
        }
        "zero-first" => {
            r###"# [zero-first]
# zero_csf = "zero_descriptors.parquet"
# full_csf = "generated_descriptors.parquet"
# output_csf = "zero_first_descriptors.parquet"
# json = false
"###
        }
        "csfs-split" => {
            r###"# [csfs-split]
# split_csfs_parquet = "generated_descriptors.parquet"
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
# input_csf = "input_descriptors.parquet"
# prefix = "split"
# copy_w = true
# overwrite = true
# json = false
"###
        }
        "interacting" => {
            r###"# [interacting]
# reference = "reference_descriptors.parquet"
# candidates = "generated_descriptors.parquet"
# output = "interacting_descriptors.parquet"
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
# output_parquet = "output_descriptors.parquet"
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
