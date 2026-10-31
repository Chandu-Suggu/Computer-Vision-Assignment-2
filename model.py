"""
CSL7360 Assignment 2 - Track B: Fully Convolutional Network
Model Architecture: Lightweight Depthwise-Separable U-Net (DS-UNet)
All components built strictly from scratch using torch.nn primitives.
No external pretrained models or segmentation libraries used.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution block:
    3x3 Depthwise Conv -> BatchNorm -> ReLU -> 1x1 Pointwise Conv -> BatchNorm -> ReLU
    Drastically reduces FLOPs and parameter count compared to standard 3x3 conv.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride,
            padding=1, groups=in_channels, bias=False
        )
        self.bn_dw = nn.BatchNorm2d(in_channels)
        self.act_dw = nn.ReLU(inplace=True)
        
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1,
            padding=0, bias=False
        )
        self.bn_pw = nn.BatchNorm2d(out_channels)
        self.act_pw = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.act_dw(self.bn_dw(self.depthwise(x)))
        x = self.act_pw(self.bn_pw(self.pointwise(x)))
        return x

class DSUNet(nn.Module):
    """
    Lightweight U-Net architecture for cost-efficient salient object segmentation.
    - Encoder: 4 stages shrinking resolution by 2x at each stage via strided depthwise-separable convs.
    - Decoder: Bilinear upsampling with skip connections concatenated from encoder.
    - Head: 1x1 conv predicting binary segmentation logits.
    """
    def __init__(self, in_channels=3, base_channels=12):
        super().__init__()
        self.base_channels = base_channels
        c1 = base_channels       # 12
        c2 = base_channels * 2   # 24
        c3 = base_channels * 4   # 48
        c4 = base_channels * 8   # 96

        # Encoder Stage 1 (full resolution)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(c1, c1, stride=1)
        )
        # Encoder Stage 2 (downsampled 2x)
        self.enc2 = nn.Sequential(
            DepthwiseSeparableConv(c1, c2, stride=2),
            DepthwiseSeparableConv(c2, c2, stride=1)
        )
        # Encoder Stage 3 (downsampled 4x)
        self.enc3 = nn.Sequential(
            DepthwiseSeparableConv(c2, c3, stride=2),
            DepthwiseSeparableConv(c3, c3, stride=1)
        )
        # Encoder Stage 4 / Bottleneck (downsampled 8x)
        self.enc4 = nn.Sequential(
            DepthwiseSeparableConv(c3, c4, stride=2),
            DepthwiseSeparableConv(c4, c4, stride=1)
        )

        # Decoder Stage 3 (upsample to Stage 3 resolution + concatenate skip from enc3)
        self.dec3 = DepthwiseSeparableConv(c4 + c3, c3, stride=1)

        # Decoder Stage 2 (upsample to Stage 2 resolution + concatenate skip from enc2)
        self.dec2 = DepthwiseSeparableConv(c3 + c2, c2, stride=1)

        # Decoder Stage 1 (upsample to Stage 1 resolution + concatenate skip from enc1)
        self.dec1 = DepthwiseSeparableConv(c2 + c1, c1, stride=1)

        # Final prediction head (1x1 conv to 1 logit map)
        self.head = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)  # H, W
        e2 = self.enc2(e1) # H/2, W/2
        e3 = self.enc3(e2) # H/4, W/4
        e4 = self.enc4(e3) # H/8, W/8

        # Decoder with bilinear upsampling + skip connections
        d3 = F.interpolate(e4, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        logits = self.head(d1)
        return logits

def get_model(base_channels=12):
    return DSUNet(in_channels=3, base_channels=base_channels)

if __name__ == "__main__":
    for bc in [12, 16]:
        m = get_model(base_channels=bc)
        params = sum(p.numel() for p in m.parameters())
        print(f"Base channels: {bc} -> Total parameters: {params:,}")
    inp = torch.randn(1, 3, 128, 128)
    out = m(inp)
    print(f"Input shape: {inp.shape} -> Output shape: {out.shape}")
