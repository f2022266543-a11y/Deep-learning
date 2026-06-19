"""
Custom CNN architecture for plant disease classification.
5-block convolutional network with batch normalization and dropout.
"""

import torch.nn as nn


class CustomCNN(nn.Module):
    """
    A custom 5-layer CNN for plant disease classification.

    Architecture:
        - Block 1: Conv(3→32) → BN → ReLU → MaxPool
        - Block 2: Conv(32→64) → BN → ReLU → MaxPool
        - Block 3: Conv(64→128) → BN → ReLU → MaxPool
        - Block 4: Conv(128→256) → BN → ReLU → MaxPool
        - Block 5: Conv(256→512) → BN → ReLU → AdaptiveAvgPool
        - Classifier: FC(512→512) → ReLU → Dropout → FC(512→num_classes)
    """

    def __init__(self, num_classes, dropout=0.5):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 224x224 → 112x112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 112x112 → 56x56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 56x56 → 28x28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 28x28 → 14x14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 5: 14x14 → 7x7
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Adaptive pooling makes classifier input-size agnostic
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x
