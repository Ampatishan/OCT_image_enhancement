import torch.nn as nn

class DummyModel(nn.Module):
    def __init__(self, input_dim=10, output_dim=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )
    
    def forward(self, x):
        return self.fc(x)
