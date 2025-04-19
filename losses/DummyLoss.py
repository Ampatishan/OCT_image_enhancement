import torch.nn as nn

class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, outputs, targets):
        return self.loss(outputs, targets)
