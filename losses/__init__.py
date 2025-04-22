import torch
import torch.nn as nn
import torchmetrics

class LossFactory:
    """
    A factory for combining multiple loss functions into one.
    
    :param loss_weights: Dict of loss weights (e.g., {'ssim': 1.0, 'l1': 0.5, 'l2': 0.25}).
    :param reduction: How to reduce the final loss ('mean' or 'sum').
    """
    def __init__(self, loss_weights=None, reduction='mean',device='cuda'):
        self.loss_weights = loss_weights or {'ssim': 1.0, 'l1': 1.0, 'l2': 1.0}
        self.reduction = reduction
        
        # Loss functions
        self.ssim_loss = torchmetrics.StructuralSimilarityIndexMeasure()
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss(reduction='mean')
        self.device = device

    def __call__(self, x, y):
        """
        Calculate the combined loss.
        :param x: Predicted tensor.
        :param y: Ground truth tensor.
        :return: Combined weighted loss.
        """
        x = x.to(self.device)
        y = y.to(self.device)
        total_loss = torch.tensor(0.0, device=self.device)
        if 'ssim' in self.loss_weights:
            total_loss += self.loss_weights['ssim'] * self.ssim_loss(x, y)
        if 'l1' in self.loss_weights:
            total_loss += self.loss_weights['l1'] * self.l1_loss(x, y)
        if 'l2' in self.loss_weights:
            total_loss += self.loss_weights['l2'] * self.l2_loss(x, y)
        return total_loss
    

