import torch
import torch.nn as nn
from .decoder_back import RSU4F, RSU5, RSU6, RSU7, RSU4
import torch.nn.functional as F


class UpConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpConvBlock, self).__init__()
        self.blk = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self, x):
        return self.blk(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.conv = nn.Conv2d(64, 48, kernel_size=3, stride=1, padding=1)

        self.upconv3 = UpConvBlock(512, 256)
        self.upconv2 = UpConvBlock(512, 128)
        self.upconv1 = UpConvBlock(256, 64)
        self.upconv0 = UpConvBlock(128, 48)
        self.upconv2mask = UpConvBlock(96, 48)

        self.final_conv = nn.Conv2d(48, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, encoder_output, concat_features):
        # concat_features = [level0, level1, level2, level3]
        f0, f1, f2, f3 = concat_features

        # 512 x 8 x 8 -> 512 x 16 x 16
        x_up3 = self.upconv3(encoder_output)
        x_up3 = torch.cat([x_up3, f3], dim=1)

        # 512 x 16 x 16 -> 256 x 32 x 32
        x_up2 = self.upconv2(x_up3)
        x_up2 = torch.cat([x_up2, f2], dim=1)

        # 256 x 32 x 32 -> 128 x 64 x 64
        x_up1 = self.upconv1(x_up2)
        x_up1 = torch.cat([x_up1, f1], dim=1)

        # 128 x 64 x 64 -> 96 x 128 x 128
        x_up0 = self.upconv0(x_up1)
        f0 = self.conv(f0)
        x_up2mask = torch.cat([x_up0, f0], dim=1)

        # 96 x 128 x 128 -> 48 x 256 x 256
        x_mask = self.upconv2mask(x_up2mask)

        # 48 x 256 x 256 -> 1 x 256 x 256
        x_mask = self.final_conv(x_mask)

        return x_mask


def _upsample_like(src, tar):
    src = F.upsample(src, size=tar.shape[2:], mode='bilinear')

    return src


class myDecoder(nn.Module):

    def __init__(self):
        super(myDecoder, self).__init__()

        self.upconv4 = RSU4F(512, 128, 256)
        self.upconv3 = RSU4(512, 64, 128)
        self.upconv2 = RSU4(256, 32, 64)
        self.upconv1 = RSU4(128, 32, 64)
        self.upconv0 = RSU4(128, 16, 64)
        self.upconv = RSU4(64, 16, 16)

        # self.side = nn.Conv2d(64, 1, 3, padding=1)
        # self.side0 = nn.Conv2d(64, 1, 3, padding=1)
        # self.side1 = nn.Conv2d(64, 1, 3, padding=1)
        # self.side2 = nn.Conv2d(64, 1, 3, padding=1)
        # self.side3 = nn.Conv2d(128, 1, 3, padding=1)
        # self.side4 = nn.Conv2d(256, 1, 3, padding=1)

        self.outconv = nn.Conv2d(16 , 2, kernel_size=3, stride=1, padding=1)

    def forward(self, encoder_output, concat_features):
        f0, f1, f2, f3 = concat_features

        x_up4d = self.upconv4(encoder_output)
        x_up4_up = _upsample_like(x_up4d, f3)

        x_up3d = self.upconv3(torch.cat((x_up4_up, f3), 1))
        x_up3_up = _upsample_like(x_up3d, f2)

        x_up2d = self.upconv2(torch.cat((x_up3_up, f2), 1))
        x_up2_up = _upsample_like(x_up2d, f1)

        x_up1d = self.upconv1(torch.cat((x_up2_up, f1), 1))
        x_up1_up = _upsample_like(x_up1d, f0)

        x_up0d = self.upconv0(torch.cat((x_up1_up, f0), 1))
        x_up0_up = F.upsample(x_up0d, (256, 256), mode='bilinear')

        x_upd = self.upconv(x_up0_up)

#        side_d = self.side(x_upd)

#        side_0d = self.side0(x_up0d)
#        side_0d = _upsample_like(side_0d, side_d)

#        side_1d = self.side1(x_up1d)
#        side_1d = _upsample_like(side_1d, side_d)

#        side_2d = self.side2(x_up2d)
#        side_2d = _upsample_like(side_2d, side_d)

#        side_3d = self.side3(x_up3d)
#        side_3d = _upsample_like(side_3d, side_d)

#        side_4d = self.side4(x_up4d)
#        side_4d = _upsample_like(side_4d, side_d)

#        out = self.outconv(torch.cat((side_d, side_0d, side_1d, side_2d, side_3d, side_4d), 1))
        out = self.outconv(x_upd)




        return out
