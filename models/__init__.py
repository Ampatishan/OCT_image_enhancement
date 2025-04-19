from . import DummyModel

MODEL_REGISTRY = {
    "DummyModel": DummyModel.DummyModel,
    
}# Add your model classes to this dictionary


def get_model(cfg):
    """
    Get the model class from the registry based on the configuration.
    
    Args:
        cfg (str): The name of the model class.
        
    Returns:
        model: An instance of the model class.
    """
    if cfg['name'] not in MODEL_REGISTRY:
        raise ValueError(f"Model {cfg.name} not found in registry.")
    
    return MODEL_REGISTRY[cfg.name](**cfg.config.param)