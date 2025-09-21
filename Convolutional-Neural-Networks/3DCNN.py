""""
Introduction to 3D CNNs

3D CNNs are used when working with volumetric data. In 2D cnns, filters slide over height and width to extract features from flat images. 
In 3D CNNs, you add a depth dimension so filters operate over height width and depth capturing spatial relationships over slices,
crucial for medical imaging where there are multiple slices forming a 3D volume (e.g a brain scan contains multiple DICOM slices, those stack of slices are now 3D volumes)

Extensions of 2D for 3D:
1) replace nn.Conv2d(input channels, output channels, kernel_size = (depth, height, width)) to nn.Conv3d(). 
similarly for pooling (MaxPool3d) and batch norm (Batchnorm3d). 
2) Input shape for 3D (batchsize, channels, depth, height, width)

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        #2 conv layers
        self.conv1 = nn.Conv3d(3, 16, 3, 1) # input channels, output channels, kernel size = 3*3*3, stride = 1
        self.conv2 = nn.Conv3d(16, 32, 3, 1)
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.conv3 = nn.Conv3d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64*1*252*252, 1020) # 1 is depth and height and width = 252
        self.fc2 = nn.Linear(1020, 10) # considering 10 class as the output
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = F.relu(self.conv3(x))
        x = x.reshape(x.shape[0], -1)
        x = self.dropout2(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    
model = Network()
inputs = torch.randn(1, 3, 10, 512, 512)
output = model(inputs)  
print(output.shape)