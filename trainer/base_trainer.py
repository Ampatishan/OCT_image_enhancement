# filepath: /home/ampatishan/BaseClass/trainer.py
import torch
from configs.logger import setup_logger
from tqdm import tqdm


class Trainer:
    def __init__(self, model, train_loader, val_loader, loss_fn, optimizer, scheduler, cfg, device, log):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.device = device
        self.log = log

    def train(self):
        self.log.info("Starting training...")
        for epoch in range(self.cfg.training.epochs):
            self.log.info(f"Epoch {epoch + 1}/{self.cfg.training.epochs}")
            train_loss = self._train_one_epoch()
            val_loss = self._validate()

            self.log.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if self.scheduler:
                self.scheduler.step()

    def _train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            inputs, targets = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def _validate(self):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                running_loss += loss.item()

        return running_loss / len(self.val_loader)