import torch.nn as nn

class TreatmentResponseModel(nn.Module):
    def __init__(self, in_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 2 classes: 0=Not respond, 1=Respond
        )

    def forward(self, x):
        return self.net(x)
