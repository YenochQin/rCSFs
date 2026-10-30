# Small CSFG compatibility fixture

`input` was run with GRASPG e157d9f, as integrated in grasp-atomic-suite
50ca581. `rcsfg.txt` and `rlabel.txt` are its unchanged outputs. The model
is 1s², labeling space 1s, active space 3s/3p, SD, 2J=0..4. It contains
all five CSFG types, same-kappa distinct-radial pairs and same-subshell
pairs. Maintained tests read these small fixtures without requiring an
external Fortran program. File names use `.txt` so source distributions
and Git ignore rules do not mistake them for run outputs.
