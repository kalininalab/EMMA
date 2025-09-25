# threshold for Ki, Kd, IC50  and EC50
# https://pubchem.ncbi.nlm.nih.gov/bioassay/1795532#section=Description
EnInh_KiIcMic_nM = 1000
EnNi_KiECKdIC_nM = 10000
EnInh_inh_perc = 70
EnNi_inh_perc = 10
############
# threshold for Km
# https://bionumbers.hms.harvard.edu/bionumber.aspx?s=n&v=3&id=105085
# https://bionumbers.hms.harvard.edu/bionumber.aspx?s=n&v=1&id=111413
EnNi_km_nM = 100000000  # 0.1 M, 100 mM,100000 uM
EnSub_km_nM = 10000  # 10 uM


R_params = {
    "training": {"batch_size": 64, "initial_lr": 1e-6, "weight_decay": 4e-3, "max_epochs": 100},
    "loss": {"reduction": "mean",
             "interaction_gamma": 3.0, "subclass_gamma": 2.0,
             "interaction_alpha": 0.3, "subclass_alpha": 0.3}
}

C1f_params = {
    "training": {"batch_size": 64, "initial_lr": 1e-6, "weight_decay": 4e-3, "max_epochs": 100},
    "loss": {"reduction": "mean",
             "interaction_gamma": 3.0, "subclass_gamma": 2.0,
             "interaction_alpha": 0.3, "subclass_alpha": 0.3}
}

C1e_params = {
    "training": {"batch_size": 64, "initial_lr": 1e-6, "weight_decay": 4e-3, "max_epochs": 100},
    "loss": {"reduction": "mean",
             "interaction_gamma": 3.0, "subclass_gamma": 2.0,
             "interaction_alpha": 0.5, "subclass_alpha": 0.5}
}

C1_params = {
    "training": {"batch_size": 64, "initial_lr": 1e-6, "weight_decay": 4e-3, "max_epochs": 100},
    "loss": {"reduction": "mean",
             "interaction_gamma": 3.0, "subclass_gamma": 2.0,
             "interaction_alpha": 0.5, "subclass_alpha": 0.5}
}

C2_params = {
    "training": {"batch_size": 32, "initial_lr": 1e-6, "weight_decay": 8e-3, "max_epochs": 100},
    "loss": {"reduction": "mean",
             "interaction_gamma": 3.0, "subclass_gamma": 2.0,
             "interaction_alpha": 0.3, "subclass_alpha": 0.3}
}



