import torch.nn as nn
from . import DummyLoss
def get_loss_fn(cfg):
    """
    Get the loss function class based on the configuration.
    
    Args:
        loss_name (str): Config for the loss function class.
        
    Returns:
        loss_fn: An instance of the loss function class.
    """
    if cfg.name == "L1":
        return nn.L1Loss()
    elif cfg.name == "MSE":
        return nn.MSELoss()
    elif cfg.name == "DummyLoss":
        return DummyLoss.DummyLoss()
    else:
        raise ValueError(f"Loss {cfg.name} not implemented")
    

