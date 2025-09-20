""""
This is implementation of SA_Unetv2 that uses both encoder and decoder feature for attention not just add attention in the bottleneck as done in SA-Unet

You can find original paper here -> https://arxiv.org/html/2509.11774v1

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Average across channels
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Max across channels
        combined = torch.cat([avg_out, max_out], dim=1)  # Concat: (B, 2, H, W)
        attention_map = self.sigmoid(self.conv(combined))  # (B, 1, H, W)
        return x * attention_map
        
class CrossScaleSpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # as mentioned in the paper: apply avgpooling on feature maps of encoder and decoder separately then pass it to 7*7 convolution followed by sigmoid
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding = 3)

    def forward(self, encoder_feature, decoder_feature):
        if encoder_feature.shape[2:] != decoder_feature.shape[2:]:
            decoder_feature = F.interpolate(decoder_feature, size=encoder_feature.shape[2:], mode='bilinear', align_corners=False)
        en_f = torch.mean(encoder_feature, dim=1, keepdim=True)
        de_f = torch.mean(decoder_feature, dim=1, keepdim=True)
        #now concatenate
        features = torch.cat([en_f, de_f], dim = 1)
        attention = self.sigmoid(self.conv(features))

        return encoder_feature * attention

def double_conv(in_ch, out_ch):
    # 2 convolution layers of size 3*3
    conv = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding = 1),
        nn.Dropout(p = 0.15), #add some probability
        nn.GroupNorm(8, out_ch),
        nn.SiLU(),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding = 1),
        nn.Dropout(p = 0.15),
        nn.GroupNorm(8, out_ch),
        nn.SiLU()
    )
    return conv

# def crop_image(tensor, target):
#     #reduce the tensor to target size
#     target_size = target.size()[2]
#     tensor_size = tensor.size()[2]
#     delta = tensor_size - target_size
#     delta = delta // 2
#     return tensor[:, :, delta: tensor_size - delta, delta: tensor_size- delta]

class SAUnetv2(nn.Module):
    def __init__(self, input_channel = 3, output_channel = 1):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_1 = double_conv(input_channel, 16)
        self.conv_2 = double_conv(16, 32)
        self.conv_3 = double_conv(32, 48)       
        self.conv_4 = double_conv(48, 64)

        self.bottleneck = double_conv(64, 64) #bottleneck
        #applying spatial attention to bottleneck
        self.spatial_attention = SpatialAttention(kernel_size = 7)
        self.csa4 = CrossScaleSpatialAttention()
        self.csa3 = CrossScaleSpatialAttention()
        self.csa2 = CrossScaleSpatialAttention()
        self.csa1 = CrossScaleSpatialAttention()

        self.up4 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2)
        self.up_conv4 = double_conv(128, 64)
        
        self.up3 = nn.ConvTranspose2d(in_channels=64, out_channels=48, kernel_size=2, stride=2)
        self.up_conv3 = double_conv(96, 48)

        self.up2 = nn.ConvTranspose2d(in_channels=48, out_channels=32, kernel_size=2, stride=2)
        self.up_conv2 = double_conv(64, 32)

        self.up1 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2)
        self.up_conv1 = double_conv(32, 16)

        self.output = nn.Conv2d(in_channels=16, out_channels=output_channel, kernel_size=1)
    
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
        x9 = self.bottleneck(x8)
        bottleneck = self.spatial_attention(x9)

        x = self.up4(bottleneck)  
        print(x.shape)
        #we are croping to make the dims compatible in order to add as given in the architecture
        d4_attention = self.csa4(x7, x)
        x = self.up_conv4(torch.cat([x, d4_attention], 1))

        x = self.up3(x)  
        print(x.shape)
        d3_attention = self.csa3(x5, x)
        x = self.up_conv3(torch.cat([x, d3_attention], 1))

        x = self.up2(x) 
        print(x.shape)
 
        d2_attention= self.csa2(x3, x)
        x = self.up_conv2(torch.cat([x, d2_attention], 1))

        x = self.up1(x)  
        print(x.shape)
        d1_attention= self.csa1(x1, x)
        x = self.up_conv1(torch.cat([x, d1_attention], 1))
        
        x = torch.sigmoid(self.output(x))
        print(x.size())
       
model = SAUnetv2()
#batchsize , channel , height, width
image = torch.rand((1, 3, 592, 592))
print(model(image))

## loss function

"""
Implement compound loss -> SAUnetv2 adopts weighted Binary cross entropy plus Matthews Correlation Coefficient (MCC) loss to improve robustness to class imbalance.
"""
def calculate_mcc(probab, target):
    eps = 1e-7
    tp = torch.sum(probab * target)
    tn = torch.sum((1 - target) * (1 - probab))
    fp = torch.sum((1-target) * probab)
    fn = torch.sum(target * (1 - probab))
    numerator = (tp * tn) - (fp * fn)
    denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + eps)
    mcc = numerator / denominator
    return 1 - mcc
    
def compound_loss(logits, target):
    lambda1, lambda2 = 0.5, 0.5
    if(logits.size() != target.size()):
        print("Size mismatch for logits and target")
    else:
        bce = nn.BCEWithLogitsLoss()(logits, target)
        #MCC needs probability
        probab = torch.sigmoid(logits)
        mcc = calculate_mcc(probab, target)

    return (lambda1 * bce + lambda2 * mcc)