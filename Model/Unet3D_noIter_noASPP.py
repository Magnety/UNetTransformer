# 3D-UNet model.
# x: 128x128 resolution for 128 frames.      x: 320*320 resolution for 32frames.
"""
@author: liuyiyao
@time: 2020/10/15 下午
"""

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        rate_list = (1, rate, rate)
        self.atrous_convolution = nn.Conv3d(inplanes, planes, kernel_size=3,
                                            stride=1, padding=rate_list, dilation=rate_list)
        self.group_norm = nn.GroupNorm(16, planes)
        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.group_norm(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class conv_block(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(16, out_ch),
            nn.PReLU(),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(16, out_ch),
            nn.PReLU())

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_block2(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block2, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(16, out_ch),
            nn.PReLU(),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(16, out_ch),
            nn.PReLU())

    def forward(self, x):
        x = self.conv(x)
        return x
class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            #nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(16, out_ch),
            nn.PReLU(),
        )

    def forward(self, x):
        x = self.up(x)
        return x






class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(16, F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(16, F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(1, 1),
            nn.Sigmoid()
        )

        self.PReLU = nn.PReLU()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.PReLU(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out

class AttU_Net(nn.Module):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """
    def __init__(self, img_ch=1, output_ch=1):
        super(AttU_Net, self).__init__()

        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.AvgPool3d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.AvgPool3d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.AvgPool3d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.AvgPool3d(kernel_size=2, stride=2)

        self.Conv1 = conv_block2(img_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Recons_up_conv1 = up_conv(filters[1], filters[0])
        self.Recons_up_conv2 = up_conv(filters[2],filters[1])


        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv3d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

        self.extra_feature1 = nn.Sequential(nn.AdaptiveAvgPool3d((1,1,1)))
        self.extra_feature2 = nn.Sequential(nn.AdaptiveAvgPool3d((1,1,1)),nn.Dropout(0.5))
        self.extra_feature3 = nn.Sequential(nn.AdaptiveAvgPool3d((1,1,1)))
        self.extra_feature4 = nn.Sequential(nn.AdaptiveAvgPool3d((1,1,1)))
        self.extra_feature5 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)),nn.Dropout(0.5))
        self.extra_feature6 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)))
        self.extra_feature7 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)),nn.Dropout(0.5))
        self.extra_feature8 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)))
        self.fc1 = nn.Sequential(nn.Linear(688, 384),nn.Dropout(0.5))
        self.fc2 = nn.Sequential(nn.Linear(384, 128),nn.Dropout(0.5))
        self.fc3 = nn.Sequential(nn.Linear(128, 2), nn.LogSoftmax())

        rates = (1, 6, 12, 18)

        self.aspp1 = ASPP_module(16, 16, rate=rates[0])
        self.aspp2 = ASPP_module(16, 16, rate=rates[1])
        self.aspp3 = ASPP_module(16, 16, rate=rates[2])
        self.aspp4 = ASPP_module(16, 16, rate=rates[3])

        self.aspp_conv = nn.Conv3d(64, 64, 1)
        self.aspp_gn = nn.GroupNorm(32, 64)
        self.Out = nn.Conv3d(64, output_ch, kernel_size=1, stride=1, padding=0)
        #self.active = torch.nn.Sigmoid()


    def forward(self, x):
        print("x shape",x.shape)
        down1 = self.Conv1(x)
        print("down1 shape",down1.shape)

        pool1 = self.Maxpool1(down1)
        print("pool1 shape",pool1.shape)
        down2 = self.Conv2(pool1)
        recons_up1 = F.upsample(down2, size=down1.size()[2:], mode='trilinear')
        recons1 = self.Recons_up_conv1(recons_up1)
        print("down2 shape",down2.shape)

        pool2 = self.Maxpool2(down2)
        print("pool2 shape",pool2.shape)
        down3 = self.Conv3(pool2)
        feature3 = self.extra_feature3(down3)
        recons_up2 = F.upsample(down3, size=down2.size()[2:], mode='trilinear')
        recons2 = self.Recons_up_conv2(recons_up2)
        print("down3 shape",down3.shape)
        pool3 = self.Maxpool3(down3)
        print("pool3 shape",pool3.shape)
        down4 = self.Conv4(pool3)

        feature4 = self.extra_feature4(down4)

        print("down4 shape",down4.shape)
        pool4 = self.Maxpool4(down4)
        print("pool4 shape",pool4.shape)
        down5 = self.Conv5(pool4)
        print("down5 shape",down5.shape)

        feature5 = self.extra_feature5(down5)

        #print(x5.shape)
        down5_sample = F.upsample(down5, size=down4.size()[2:], mode='trilinear')
        up5 = self.Up5(down5_sample)
        print("up5 shape",up5.shape)
        #print(d5.shape)
        att4 = self.Att5(g=up5, x=down4)
        print("att4 shape",att4.shape)
        up5_cat = torch.cat((att4, up5), dim=1)
        print("up5_cat shape",up5_cat.shape)
        up5_conv = self.Up_conv5(up5_cat)

        feature6 = self.extra_feature6(up5_conv)

        print("up5_conv shape",up5_conv.shape)
        up5_sample = F.upsample(up5_conv, size=down3.size()[2:], mode='trilinear')
        up4 = self.Up4(up5_sample)
        print("up4 shape",up4.shape)
        att3 = self.Att4(g=up4, x=down3)
        print("att3 shape",att3.shape)
        up4_cat = torch.cat((att3, up4), dim=1)
        print("up4_cat shape",up4_cat.shape)
        up4_conv = self.Up_conv4(up4_cat)

        feature7 = self.extra_feature7(up4_conv)

        print("up4_conv shape",up4_conv.shape)
        up4_sample = F.upsample(up4_conv, size=down2.size()[2:], mode='trilinear')
        up3 = self.Up3(up4_sample)
        print("up3 shape",up3.shape)
        att2 = self.Att3(g=up3, x=down2)
        print("att2 shape",att2.shape)
        up3_cat = torch.cat((att2, up3), dim=1)
        print("up3_cat shape",up3_cat.shape)
        up3_conv = self.Up_conv3(up3_cat)
        feature1= self.extra_feature1(up3_conv)

        print("up3_conv shape",up3_conv.shape)
        up3_sample = F.upsample(up3_conv, size=down1.size()[2:], mode='trilinear')
        up2 = self.Up2(up3_sample)
        print("up2 shape",up2.shape)
        att1 = self.Att2(g=up2, x=down1)
        print("att1 shape",att1.shape)
        up2_cat = torch.cat((att1, up2), dim=1)
        print("up2_cat shape",up2_cat.shape)
        up2_conv = self.Up_conv2(up2_cat)
        feature2 = self.extra_feature2(up2_conv)

        print("up2_conv.shape",up2_conv.shape)
        out1 = self.Conv(up2_conv)
        print("out1 shape",out1.shape)


        feature = torch.cat((feature1,feature2,feature3, feature4,feature5,feature6,feature7), 1)
        feature = feature.view(-1)
        print("feature shape",feature.shape)

        cls_out = self.fc1(feature)
        cls_out = self.fc2(cls_out)
        cls_out = self.fc3(cls_out)

      #  out = self.active(out)

        return down1,recons1,down2,recons2,out1,out1,cls_out

