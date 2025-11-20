import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=200):
        super(AlexNet, self).__init__()

        # Input: B x 3 x 224 x 224
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            # After Conv(11, stride=4, padding=2) + ReLU: B x 64 x 55 x 55
            nn.MaxPool2d(kernel_size=3, stride=2),
            # After MaxPool(k=3, stride=2): B x 64 x 27 x 27

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # After Conv(5, padding=2) + ReLU: B x 192 x 27 x 27
            nn.MaxPool2d(kernel_size=3, stride=2),
            # After MaxPool(k=3, stride=2): B x 192 x 13 x 13

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # After Conv(3, padding=1) + ReLU: B x 384 x 13 x 13

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # After Conv(3, padding=1) + ReLU: B x 256 x 13 x 13

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # After Conv(3, padding=1) + ReLU: B x 256 x 13 x 13
            nn.MaxPool2d(kernel_size=3, stride=2),
            # After final MaxPool(k=3, stride=2): B x 256 x 6 x 6
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # B x (256*6*6) = B x 9216
        x = self.classifier(x)
        return x  # raw scores for each class