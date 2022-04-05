import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 15, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(15, 20, 5)
        self.fc1 = nn.Linear(20*5*5, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 200)
        self.fc4 = nn.Linear(200, 100)
        self.fc5 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.drop(self.conv1(x))))
        x = self.pool(F.relu(self.drop(self.conv2(x))))
# flatten all dimensions except batch
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
