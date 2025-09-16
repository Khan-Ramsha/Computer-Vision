import torch
import torch.nn as nn
import torch.nn.functional as F

class Alexnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride = 4, padding = 2),
            #output dim: 55*55*96
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            #27 × 27 × 96
            nn.Conv2d(96, 256, 5, padding = 2),
            #127 × 27 × 256
            nn.MaxPool2d(3, 2),
            # 13 × 13 × 256
            #now back to back 3 convolution layers 
            nn.Conv2d(256, 384, 3, padding = 1),
            #13-3+2/1+1=13
            nn.Conv2d(384, 384, 3, padding = 1),
            #13*13
            nn.Conv2d(384, 256, 3, padding = 1),
            #13*13
            nn.MaxPool2d(3, 2),
            #6*6*256
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p = 0.5),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p = 0.5),

            nn.Linear(4096, 1000)
            
        )
    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

model = Alexnet()
inputs = torch.randn((16, 3, 227, 227))
print(model(inputs).shape)