import torch 
import torch.nn as nn 


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample_flag=True, dropout_flag=False):
        super(ConvBlock, self).__init__()
        self.downsample_flag = downsample_flag
        self.dropout_flag = dropout_flag
        self.down_operation = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
        self.up_operation = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()   
        )
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        if self.downsample_flag: 
            return self.down_operation(x)
        elif self.dropout_flag:
            return self.dropout(self.up_operation(x))
        else:
            return self.up_operation(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2)
        )
        
        current_channels = 64
        self.down_layers = [self.initial]
        for _ in range(3):
            self.down_layers.append(ConvBlock(current_channels, out_channels=current_channels*2))
            current_channels = current_channels*2
        for _ in range(3):
            self.down_layers.append(ConvBlock(current_channels, current_channels))
            
        self.bottleneck = nn.Sequential(
            nn.Conv2d(current_channels, current_channels, 4, 2, 1), 
            nn.ReLU()
        )
     
        self.down_layers.append(self.bottleneck)
        self.up_layers = [
        ConvBlock(current_channels , current_channels, downsample_flag=False, dropout_flag=True),
        ConvBlock(current_channels * 2, current_channels, downsample_flag=False, dropout_flag=True),
        ConvBlock(current_channels * 2, current_channels, downsample_flag=False, dropout_flag=True),
        ConvBlock(current_channels * 2, current_channels, downsample_flag=False, dropout_flag=False),
        ConvBlock(current_channels * 2, current_channels // 2, downsample_flag=False, dropout_flag=False),
        ConvBlock(current_channels , current_channels // 4, downsample_flag=False, dropout_flag=False),
        ConvBlock(current_channels // 2, current_channels//8, downsample_flag=False, dropout_flag=False),
        nn.Sequential(
            nn.ConvTranspose2d(current_channels // 4, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    ]
        

    def forward(self, x):
            # Down-sampling layers
            d1 = self.down_layers[0](x)
            d2 = self.down_layers[1](d1)
            d3 = self.down_layers[2](d2)
            d4 = self.down_layers[3](d3)
            d5 = self.down_layers[4](d4)
            d6 = self.down_layers[5](d5)
            d7 = self.down_layers[6](d6)

            # Bottleneck layer
            bottleneck = self.down_layers[7](d7)
            
            # Up-sampling layers
            up1 = self.up_layers[0](bottleneck)
            up2 = self.up_layers[1](torch.cat([up1, d7], 1))
            up3 = self.up_layers[2](torch.cat([up2, d6], 1))
            up4 = self.up_layers[3](torch.cat([up3, d5], 1))
            up5 = self.up_layers[4](torch.cat([up4, d4], 1))
            up6 = self.up_layers[5](torch.cat([up5, d3], 1))
            up7 = self.up_layers[6](torch.cat([up6, d2], 1))

            # Final layer
            return self.up_layers[7](torch.cat([up7, d1], 1))
        

def test():
    x = torch.randn((1, 3, 256, 256))
    model = Generator()
    preds = model(x)
    print(preds.shape)


if __name__ == "__main__":
    test()