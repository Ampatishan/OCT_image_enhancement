# train.py
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from  configs.logger import setup_logger
from models import get_model
from datasets import get_dataloader
from losses import get_loss_fn
from utils.optim_utils import get_optimizer, get_scheduler
from trainer.base_trainer import Trainer
import os


@hydra.main(config_path="configs", config_name="dummy_config.yaml")
def main(cfg: DictConfig):
    
    # Setup logger
    log = setup_logger(cfg)
    
    print("\n🚀 Starting training with config:")
    print(OmegaConf.to_yaml(cfg))

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # Load model
    log.info(f"Initializing model: {cfg.model.name}")
    model = get_model(cfg.model).to(device)

    # Load dataset
    log.info("Preparing dataloaders...")
    train_loader, val_loader = get_dataloader(cfg.dataset, cfg.training.batch_size)

    # Loss function
    loss_fn = get_loss_fn(cfg.loss).to(device)

    # Optimizer and scheduler
    optimizer = get_optimizer(cfg.optimizer, model.parameters())
    scheduler = get_scheduler(cfg.scheduler, optimizer)

    # Init Trainer
    log.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        device=device,
        log=log
    )

    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
