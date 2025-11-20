import torch
from torch import nn

# Define the custom neural network
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()

        # Convolutional sequence (Conv -> ReLU -> MaxPool)
        self.features = nn.Sequential(
            # Block 1: B x 3 x 224 x 224 -> B x 64 x 112 x 112
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: B x 64 x 112 x 112 -> B x 128 x 56 x 56
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Calculate the dimensions after the two MaxPools: 224 -> 112 -> 56
        # Channels: 128
        # Flattened size: 128 * 56 * 56 = 401408 (for batch_size=32)
        self.classifier = nn.Sequential(
            nn.Flatten(),   # Flatten the tensor that means B x 128 x 56 x 56 -> B x (128*56*56)
            nn.Linear(128 * 56 * 56, 512), # Fully connected layer means B x (128*56*56) -> B x 512
            nn.Linear(512, 200) # 200 is the number of classes in TinyImageNet
        )

    # Forward pass : defines how the data flows through the network
    def forward(self, x):
        x = self.features(x)
        logits = self.classifier(x)
        return logits   # in logits we have the raw scores for each class)