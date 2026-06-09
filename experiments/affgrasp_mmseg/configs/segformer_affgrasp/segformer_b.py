EXPERIMENT = {
    "model_name": "segformer",
    "experiment_type": "segformer_b",
    "backbone": "mit_b0",
    "pretrained": False,
    "freeze_mode": "partial",
    "use_lora": False,
    "use_adapters": False,
    "epochs": 15,
    "batch_size": 8,
    "num_workers": 2,
    "resize_size": 476,
    "crop_size": 448,
    "lr": 1e-3,
    "backbone_lr": 1e-4,
    "weight_decay": 1e-4,
    "focal_alpha": 1.0,
}

