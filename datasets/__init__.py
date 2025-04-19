import torch
from . import DummyDataset
DATASET_REGISTRY = {
    
    "DummyDataset": DummyDataset.DummyDataset,
}

def get_dataloader(cfg, batch_size=32):
    """
    Get the dataloader class from the registry based on the configuration.
    
    Args:
        cfg (str): The config file for the dataloader class.
        
    Returns:
        dataloader: An instance of the dataloader class.
    """
    if cfg.name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset {cfg.name} not found in registry.")
    
    dataset_cls = DATASET_REGISTRY[cfg.name]
    train_set = dataset_cls(cfg.train_path)
    val_set = dataset_cls(cfg.val_path)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
