ds_home = './datasets'

ascadf_files = {
    "opoi": (f"{ds_home}/ASCAD-fixed/ASCAD.h5", 700),
    "opoi+desync50": (f"{ds_home}/ASCAD-fixed/ASCAD_desync50.h5", 700),
    "opoi+desync100": (f"{ds_home}/ASCAD-fixed/ASCAD_desync100.h5", 700),
    "nopoi": (f"{ds_home}/ASCAD-fixed/ASCAD_nopoi_window_20.h5", 10000),
    "nopoi+desync50": (f"{ds_home}/ASCAD-fixed/ascad_fixed_nopoi_window_40_desync50.h5", 5000),
    "nopoi+desync100": (f"{ds_home}/ASCAD-fixed/ascad_fixed_nopoi_window_40_desync100.h5", 5000),
    "nopoi+desync200": (f"{ds_home}/ASCAD-fixed/ascad_fixed_nopoi_window_40_desync200.h5", 5000),
}

ascadr_files = {
    "opoi": (f"{ds_home}/ASCAD-random/ascad-variable.h5", 1400),
    "opoi+desync50": (f"{ds_home}/ASCAD-random/ascad-variable-desync50.h5", 1400),
    "opoi+desync100": (f"{ds_home}/ASCAD-random/ascad-variable-desync100.h5", 1400),
    "nopoi": (f"{ds_home}/ASCAD-random/ascad-variable_nopoi_window_20.h5", 25000),
    "nopoi+desync50": (f"{ds_home}/ASCAD-random/ascad_random_nopoi_window_40_desync50.h5", 12500),
    "nopoi+desync100": (f"{ds_home}/ASCAD-random/ascad_random_nopoi_window_40_desync100.h5", 12500),
    "nopoi+desync200": (f"{ds_home}/ASCAD-random/ascad_random_nopoi_window_40_desync200.h5", 12500),
}

ascadf = {
    "n_profiling": 45000,
    "n_attack": 5000,
    "n_attack_evo": 5000,
    "n_validation": 5000,
    "n_attack_ge": 3000,
    "n_validation_ge": 3000,
    'files': ascadf_files,
}

ascadr = {
    "n_profiling": 195000,
    "n_attack": 5000,
    "n_attack_evo": 5000,
    "n_validation": 5000,
    "n_attack_ge": 3000,
    "n_validation_ge": 3000,
    'files': ascadr_files,
}


chesctf_files = {
    "nopoi": (f"{ds_home}/chesctf/ches_ctf_nopoi_window_40.h5", 7500),
    "nopoi+desync50": (f"{ds_home}/chesctf/ches_ctf_nopoi_window_40_desync50.h5", 7500),
    "nopoi+desync100": (f"{ds_home}/chesctf/ches_ctf_nopoi_window_40_desync100.h5", 7500),
    "nopoi+desync200": (f"{ds_home}/chesctf/ches_ctf_nopoi_window_40_desync200.h5", 7500),
}

chesctf = {
    "n_profiling": 25000,
    "n_attack": 5000,
    "n_attack_evo": 5000,
    "n_validation": 5000,
    'files': chesctf_files,
}

eshard_files = {
    "opoi": (f"{ds_home}/eshard/eshard.h5", 1400),
}

eshard = {
    "n_profiling": 85000,
    "n_attack": 5000,
    "n_attack_evo": 5000,
    "n_validation": 5000,
    'files': eshard_files,
}
