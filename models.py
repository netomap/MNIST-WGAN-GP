import torch
from torch import nn

class Discriminator(nn.Module):

    def __init__(self, channels_img, img_size, features):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=channels_img, out_channels=features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            self._block(in_channels=features, out_channels=features*2, kernel_size=4, stride=2, padding=1),
            self._block(in_channels=features*2, out_channels=features*4, kernel_size=4, stride=2, padding=1),
            self._block(in_channels=features*4, out_channels=features*8, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(in_channels=features*8, out_channels=1, kernel_size=4, stride=2, padding=0)
        )
    

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding, bias=False),
            nn.InstanceNorm2d(num_features=out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.2)
        )
    
    def forward(self, x):
        return self.net(x)

class Generator(nn.Module):
    
    def __init__(self, channel_noise, channel_img, features):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            self._block(in_channels=channel_noise, out_channels=features*16, kernel_size=4, stride=1, padding=0),
            self._block(in_channels=features*16, out_channels=features*8, kernel_size=4, stride=2, padding=1),
            self._block(in_channels=features*8, out_channels=features*4, kernel_size=4, stride=2, padding=1),
            self._block(in_channels=features*4, out_channels=features*2, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(in_channels=features*2, out_channels=channel_img, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.net(x)

def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

