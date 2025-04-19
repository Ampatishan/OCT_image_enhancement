from .optim_factory import OPTIMIZER_REGISTRY, SCHEDULER_REGISTRY

def get_optimizer(cfg, model_params):
    opt_class = OPTIMIZER_REGISTRY.get(cfg.name)
    if not opt_class:
        raise ValueError(f"Optimizer '{cfg.name}' not found in registry.")
    return opt_class(model_params, **cfg.params)

def get_scheduler(cfg, optimizer):
    sched_class = SCHEDULER_REGISTRY.get(cfg.name)
    if not sched_class:
        raise ValueError(f"Scheduler '{cfg.name}' not found in registry.")
    return sched_class(optimizer, **cfg.params)

