import torch.nn as nn


class AlexNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class CountingFeatureExtractor(nn.Module):
    def __init__(self, feat_extr, n_elems=1000):
        super().__init__()

        self.feat_extr = feat_extr

        self.flatten = nn.Flatten(start_dim=1, end_dim=3)
        self.fc6 = nn.Linear(256 * 5 * 5, 4096)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, n_elems)

    def forward(self, x):
        x = self.feat_extr(x)

        x = self.flatten(x)
        x = self.dropout(x)

        # x = nn.Linear(x.shape[1], 4096)(x) # fc6
        x = self.fc6(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc7(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc8(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x
