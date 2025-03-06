import torch
import torch.nn as nn
import torch.nn.functional as F

# Базовый сверточный блок: Conv -> BN -> ReLU6
class ConvBNReLU6(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBNReLU6, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)

# Блок depthwise-separable свертки:
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True)
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# Блок апсемплинга для декодера:
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(UpBlock, self).__init__()
        self.scale_factor = scale_factor
        self.conv = ConvBNReLU6(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        x = self.conv(x)
        return x

# Основная сеть MobileDepth с encoder-decoder архитектурой
class MobileDepth(nn.Module):
    def __init__(self, num_classes=1):
        super(MobileDepth, self).__init__()
        # Кодировщик (Encoder)
        self.initial_conv = ConvBNReLU6(3, 32, kernel_size=3, stride=2, padding=1)  # 640x640 -> 320x320, 32 каналов
        self.dsconv1 = DepthwiseSeparableConv(32, 64, stride=1)   # 320x320, 64 каналов
        self.dsconv2 = DepthwiseSeparableConv(64, 128, stride=2)   # 320x320 -> 160x160, 128 каналов
        self.dsconv3 = DepthwiseSeparableConv(128, 128, stride=1)  # 160x160, 128 каналов
        self.dsconv4 = DepthwiseSeparableConv(128, 256, stride=2)  # 160x160 -> 80x80, 256 каналов
        self.dsconv5 = DepthwiseSeparableConv(256, 256, stride=1)  # 80x80, 256 каналов
        self.dsconv6 = DepthwiseSeparableConv(256, 512, stride=2)  # 80x80 -> 40x40, 512 каналов
        
        # Сохранение skip connections
        # skip0: после initial_conv (32 каналов, 320x320)
        # skip1: после dsconv1 (64 каналов, 320x320)
        # skip2: после dsconv3 (128 каналов, 160x160)
        # skip3: после dsconv5 (256 каналов, 80x80)
        
        # Декодировщик (Decoder)
        self.up1 = UpBlock(512, 256, scale_factor=2)   # 40x40 -> 80x80, 256 каналов
        self.up2 = UpBlock(256 + 256, 128, scale_factor=2)  # объединение с skip3 -> 80x80 -> 160x160
        self.up3 = UpBlock(128 + 128, 64, scale_factor=2)   # объединение с skip2 -> 160x160 -> 320x320
        self.up4 = UpBlock(64 + 32, 32, scale_factor=2)    # объединение с skip1 -> 320x320 -> 640x640
        
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)
    
    # def forward(self, x):
    #     skip0 = self.initial_conv(x)       # [B, 32, 320, 320]
    #     skip1 = self.dsconv1(skip0)        # [B, 64, 320, 320]
    #     x1 = self.dsconv2(skip1)           # [B, 128, 160, 160]
    #     skip2 = self.dsconv3(x1)           # [B, 128, 160, 160]
    #     x2 = self.dsconv4(skip2)           # [B, 256, 80, 80]
    #     skip3 = self.dsconv5(x2)           # [B, 256, 80, 80]
    #     x3 = self.dsconv6(skip3)           # [B, 512, 40, 40]
        
    #     d1 = self.up1(x3)                # [B, 256, 80, 80]
    #     d1 = torch.cat([d1, skip3], dim=1) # [B, 512, 80, 80]
    #     d2 = self.up2(d1)                # [B, 128, 160, 160]
    #     d2 = torch.cat([d2, skip2], dim=1) # [B, 256, 160, 160]
    #     d3 = self.up3(d2)                # [B, 64, 320, 320]
    #     d3 = torch.cat([d3, skip1], dim=1) # [B, 128, 320, 320]
    #     d4 = self.up4(d3)                # [B, 32, 640, 640]
        
    #     out = self.final_conv(d4)        # [B, 1, 640, 640]
    #     return out
    
    def forward(self, x):
        # Кодировщик
        skip0 = self.initial_conv(x)       # [B, 32, 320, 320]
        skip1 = self.dsconv1(skip0)        # [B, 64, 320, 320]
        x1 = self.dsconv2(skip1)           # [B, 128, 160, 160]
        skip2 = self.dsconv3(x1)           # [B, 128, 160, 160]
        x2 = self.dsconv4(skip2)           # [B, 256, 80, 80]
        skip3 = self.dsconv5(x2)           # [B, 256, 80, 80]
        x3 = self.dsconv6(skip3)           # [B, 512, 40, 40]
        
        # Декодировщик с объединением (skip connections)
        d1 = self.up1(x3)                # [B, 256, 80, 80]
        d1 = torch.cat([d1, skip3], dim=1) # [B, 256+256=512, 80, 80]
        d2 = self.up2(d1)                # [B, 128, 160, 160]
        d2 = torch.cat([d2, skip2], dim=1) # [B, 128+128=256, 160, 160]
        d3 = self.up3(d2)                # [B, 64, 320, 320]
        d3 = torch.cat([d3, skip0], dim=1) # [B, 64+32=96, 320, 320]  <-- используем skip0 здесь
        d4 = self.up4(d3)                # [B, 32, 640, 640]
        
        out = self.final_conv(d4)        # [B, 1, 640, 640]
        return out


if __name__ == "__main__":
    model = MobileDepth(num_classes=1)
    x = torch.randn(1, 3, 640, 640)
    y = model(x)
    print("Output shape:", y.shape)  # Ожидаем: [1, 1, 640, 640]
