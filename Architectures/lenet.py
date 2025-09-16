import torch
import torch.nn as nn
import torch.nn.functional as F
class LeNet(nn.Module):
  def __init__(self):
    super().__init__()
    # input_channels = 1 # greyscale images
    # output_channels = 6 # number of kernels
    # kernel_size = (5, 5) #size of kernel (height* width)
    #layer 1
    self.conv1 = nn.Conv2d(1, 6, 5) #default stride = 1, padding = 0
    #output dim = 28*28*6
    self.pool = nn.AvgPool2d(2, 2) # kernel_size = 2, stride = 2
    #outputdim = 14*14*6
    self.conv2 = nn.Conv2d(6, 16, 5)
    #output dim = 10*10*16
    # after avg pooling 5*5*16
    self.conv3 = nn.Conv2d(16, 120, 5)
    #after conv operation 5-5/1+1 = 1

    self.fn1 = nn.Linear(120, 84)
    self.fn2 = nn.Linear(84, 10)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = F.relu(self.conv3(x))
    x = x.reshape(x.shape[0], -1) # x.shape[0] is batchsize and -1 gives 120*1*1
    x = F.relu(self.fn1(x))
    x = self.fn2(x)
    return x

model = LeNet()
inputs = torch.randn((16, 1, 32, 32)) # 16 batch of images, 1 channel, imgsize = 32*32
print(model(inputs).shape)