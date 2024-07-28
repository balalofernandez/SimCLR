import torch.nn as nn
import torch
from torchvision import models
#Need two distinct models: 
#   ResNet to encode images, f() in the paper
#   MLP with one layer and ReLU to map representations to contrastive loss space, g() in the paper

class ProjectionHead(nn.Module):

    def __init__(self, input_dims=512, output_dims=128):
        """
        In figure 8 of paper:
        Input dimensionality is always 512
        Output dimensionality appears to not change model performance past 128
        """
        super().__init__()

        #First and only layer with non-linearity
        self.fc1 = nn.Sequential(nn.Linear(input_dims, int(input_dims/2)),
                                 nn.BatchNorm1d(int(input_dims/2)),
                                 nn.ReLU())

        #Output layer
        self.output = nn.Sequential(nn.Linear(int(input_dims/2), output_dims),
                                 nn.BatchNorm1d(int(output_dims))
                                 )

    def forward(self, h):
        """
        h being the encoded representation of the image via resnet
        z being the representation of h in contrastive loss space
        """
        z = self.output(self.fc1(h.squeeze()))
        return z

class ResNet(nn.Module):

    def __init__(self,model="resnet18",change_conv=True):
        super().__init__()

        #Pretrained RESNET
        if model == "resnet34":
            self.resnet = models.resnet34()
        elif model == "resnet18":
            self.resnet = models.resnet18()
        else:
            raise "You must use either resnet18 or resnet34"
        
        #Change first convolution layer to recognise 32*32 images (CIFAR10)
        if change_conv:
           self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), bias=False)
           self.resnet.maxpool = nn.Identity()

        #I remove the avgpool and the fully connected network:
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

        for parameter in self.resnet.parameters():
            parameter.requires_grad = True
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.projection_head = ProjectionHead()

    def forward(self, batch_of_images):
        h = self.resnet(batch_of_images)
        pooled_h = self.avgpool(h)
        z = self.projection_head(pooled_h)
        return z
    
class FullNet(nn.Module):
    
    def __init__(self, encoder:ResNet, out_shape=torch.Size):
        super().__init__()
        
        self.encoder = encoder
        """for p in self.encoder.parameters():
            p.requires_grad = False"""
        
        input_shape = 512 # 2048

        self.upsampler = nn.Sequential(nn.Conv2d(input_shape, input_shape//2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
              nn.BatchNorm2d(input_shape//2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
              nn.ReLU(),
              nn.Dropout(p=0.1, inplace=False),
              nn.UpsamplingBilinear2d(size=tuple(x // 2 for x in out_shape)),
              nn.Conv2d(input_shape//2, 3, kernel_size=(1, 1), stride=(1, 1)),
              nn.UpsamplingBilinear2d(size=out_shape)
              )

    def forward(self, img):
        encoded_img = self.encoder.resnet(img)
        # result = self.upsampler(encoded_img)
        result = self.upsampler(encoded_img)
        # result = nn.functional.interpolate(result, size=(32,32), mode="bilinear", align_corners=False)
        return result