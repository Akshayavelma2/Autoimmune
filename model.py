import torch
import torch.nn as nn


# -------------------------------------------------
# Model A (Recommended): Wide + Deep + BatchNorm + Dropout
# Best for tabular data with many 0/1 + numeric features
# -------------------------------------------------
class DiseaseModel_WideDeepBN(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()

        # Wide part (memorizes direct rules)
        self.wide = nn.Linear(in_dim, num_classes)

        # Deep part (learns complex interactions)
        self.deep = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.wide(x) + self.deep(x)


# -------------------------------------------------
# Model B: Improved MLP (stable baseline)
# -------------------------------------------------
class DiseaseModel_BN(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# -------------------------------------------------
# Model C: Residual model (good when many features correlate)
# -------------------------------------------------
class DiseaseModel_Residual(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.bn = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.out = nn.Linear(256, num_classes)

    def forward(self, x):
        h = self.relu(self.fc1(x))
        h2 = self.relu(self.bn(self.fc2(h)))
        h = h + h2
        return self.out(h)


# -------------------------------------------------
# IMPORTANT: train.py imports DiseaseModel
# Choose which model you want to run by changing this one line.
# -------------------------------------------------
DiseaseModel = DiseaseModel_WideDeepBN
