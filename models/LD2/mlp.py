import torch.nn as nn

class mlp(nn.Module):
    
    def __init__(self, in_channels, num_classes):
        super(mlp, self).__init__()
        self.fc1 = nn.Linear(in_channels, 100)
        self.fc2 = nn.Linear(100, 100)
        self.out = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.out(x)

        return x