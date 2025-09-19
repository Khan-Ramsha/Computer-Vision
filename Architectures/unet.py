import torch
import torch.nn as nn
import torch.nn.functional as F
def double_conv(in_ch, out_ch):
    # 2 convolution layers of size 3*3
    conv = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3),
        nn.ReLU(),
        nn.Conv2d(out_ch, out_ch, kernel_size=3),
        nn.ReLU()
    )
    return conv

def crop_image(tensor, target):
    #reduce the tensor to target size
    target_size = target.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2
    return tensor[:, :, delta: tensor_size - delta, delta: tensor_size- delta]

class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_1 = double_conv(1, 64)
        self.conv_2 = double_conv(64, 128)
        self.conv_3 = double_conv(128, 256)       
        self.conv_4 = double_conv(256, 512)
        self.conv_5 = double_conv(512, 1024)

        self.up_transpose1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_conv1 = double_conv(1024, 512)
        
        self.up_transpose2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_conv2 = double_conv(512, 256)

        self.up_transpose3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_conv3 = double_conv(256, 128)

        self.up_transpose4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up_conv4 = double_conv(128, 64)

        self.output = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)
    
    def forward(self, image):
        #encoder
        x1 = self.conv_1(image)
        x2 = self.maxpool(x1)
        x3 = self.conv_2(x2)
        x4 = self.maxpool(x3)
        x5 = self.conv_3(x4)
        x6 = self.maxpool(x5)
        x7 = self.conv_4(x6)
        x8 = self.maxpool(x7)
        x9 = self.conv_5(x8)

        x = self.up_transpose1(x9)  
        #we are croping to make the dims compatible in order to add as given in the architecture
        y = crop_image(x7, x)
        x = self.up_conv1(torch.cat([x, y], 1))

        x = self.up_transpose2(x)  
        y = crop_image(x5, x)
        x = self.up_conv2(torch.cat([x, y], 1))

        x = self.up_transpose3(x)  
        y = crop_image(x3, x)
        x = self.up_conv3(torch.cat([x, y], 1))

        x = self.up_transpose4(x)  
        #we are croping to make the dims compatible in order to add as given in the architecture
        y = crop_image(x1, x)
        x = self.up_conv4(torch.cat([x, y], 1))
        
        x = self.output(x)
        print(x.size())
       

model = Unet()
#batchsize , channel , height, width
image = torch.rand((1, 1, 572, 572))
print(model(image))