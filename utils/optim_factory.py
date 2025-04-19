import torch

# Optimizer registry
OPTIMIZER_REGISTRY = {
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "SGD": torch.optim.SGD,
    "RMSprop": torch.optim.RMSprop,
    "Adagrad": torch.optim.Adagrad,
    # Add more or custom ones here
}

# Scheduler registry
SCHEDULER_REGISTRY = {
    "StepLR": torch.optim.lr_scheduler.StepLR,
    "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
    "ExponentialLR": torch.optim.lr_scheduler.ExponentialLR,
    "MultiStepLR": torch.optim.lr_scheduler.MultiStepLR,
    # Add custom schedulers too
}
