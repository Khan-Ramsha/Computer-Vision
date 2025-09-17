"""
vgg16 and vgg19 architecture

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

VGG_types = {
    "vgg16": [64, 64, "MaxPool", 128, 128, "MaxPool", 256, 256, 256, "MaxPool", 512, 512, 512, "MaxPool", 512, 512, 512, "MaxPool"],
    "vgg19": [64, 64, "MaxPool", 128, 128, "MaxPool", 256, 256, 256, 256, "MaxPool", 512, 512, 512, 512, "MaxPool", 512, 512, 512, 512, "MaxPool"],
}

class VGG(nn.Module):
    def __init__(self, in_channels, num_classes = 10):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.layers = self.create_layers(VGG_types['vgg19'])
        self.fully_connected_layers = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)

        )
    def forward(self, x):
        x = self.layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fully_connected_layers(x)
        return x
    
    def create_layers(self, config):
        layers = []
        for i in config:
            if type(i) == int:
                output_channels = i
                layers += [
                    nn.Conv2d(self.in_channels, output_channels, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                ]
                self.in_channels = output_channels
            elif i == "MaxPool":
                layers += [nn.MaxPool2d(2,2)]
        return nn.Sequential(*layers)
    
model = VGG(in_channels=3, num_classes=10)
x = torch.randn((1, 3, 224, 224))
prediction = model(x)
print(prediction.shape)
print(prediction)