# 3D-UNet model.
# x: 128x128 resolution for 128 frames.      x: 320*320 resolution for 32frames.
"""
@author: liuyiyao
@time: 2020/10/15 下午
"""
'''import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data.DataOperate import MySet1
import torch.nn.functional as F
def conv_block_3d(in_dim,out_dim,activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.GroupNorm(32,out_dim),
        activation,)


def conv_trans_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.GroupNorm(32, out_dim),
        activation,)



def max_pooling_3d():
    return nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)



def conv_block_2_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        conv_block_3d(in_dim, out_dim, activation),
        nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.GroupNorm(32, out_dim),)
class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        rate_list = (1, rate, rate)
        self.atrous_convolution = nn.Conv3d(inplanes, planes, kernel_size=3,
                                            stride=1, padding=rate_list, dilation=rate_list)
        self.group_norm = nn.GroupNorm(32, planes)
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

class UNet(nn.Module):
    def __init__(self, in_dim, out_dim, num_filters):
        super(UNet, self).__init__()

        self.in_dim = in_dim
        self.out_dim = 64
        self.num_filters = num_filters
        activation = nn.LeakyReLU(0.2, inplace=True)
        #1,1,32,384,384
        # Down sampling
        self.layer_1 = nn.Sequential(
        nn.Conv3d(1, 64, kernel_size=7, stride=(1,2,2), padding=(3,3,3)),
        nn.GroupNorm(32, 64),
        activation,)  #1,64,32,384,384
        self.pool_1 = max_pooling_3d()    #1,64,32,192,192
        self.layer_2 = nn.Sequential(
        nn.Conv3d(64, 128, kernel_size=1, stride=1, padding=0),
        activation,)   #1,128,16,192,192
        self.gt_2 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=1, stride=(1, 2, 2), padding=0),
            nn.GroupNorm(32, 128),
            activation, )
        self.layer_3 = nn.Sequential(
        nn.Conv3d(128, 256, kernel_size=1, stride=(1,2,2), padding=0),
        nn.GroupNorm(32, 256),
        activation,) #1,256,8,96,96
        #self.pool_3 = max_pooling_3d()    #1,256,16,48,48


        self.layer_4 = nn.Sequential(
        nn.Conv3d( 256,512, kernel_size=1, stride=1, padding=0),
        nn.GroupNorm(32, 512),
        activation,)    #1,512,4,48,48
       # self.pool_4 = max_pooling_3d()    #1,512,16,48,48


        self.layer_5 =  nn.Sequential(
        nn.Conv3d( 512,1024,kernel_size=1, stride=1, padding=0),
        nn.GroupNorm(32, 1024),
        activation,)  #1,1024,2,24,24
        #self.pool_5 = max_pooling_3d()    #1,1024,16,48,48


        self.down5 = nn.Sequential(
            nn.Conv3d(1024, 128, kernel_size=1),
            nn.GroupNorm(32, 128),
            nn.PReLU(),
        )

        self.down4 = nn.Sequential(
            nn.Conv3d(512, 128, kernel_size=1),
            nn.GroupNorm(32, 128),
            nn.PReLU(),
        )
        self.down3 = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=1),
            nn.GroupNorm(32, 128),
            nn.PReLU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=1),
            nn.GroupNorm(32, 128),
            nn.PReLU(),
        )
        self.fusdown1 = nn.Sequential(
            nn.Conv3d(512, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU()
        )
        rates = (1, 6, 12, 18)
        self.aspp1 = ASPP_module(64, 64, rate=rates[0])
        self.aspp2 = ASPP_module(64, 64, rate=rates[1])
        self.aspp3 = ASPP_module(64, 64, rate=rates[2])
        self.aspp4 = ASPP_module(64, 64, rate=rates[3])

        self.aspp_conv = nn.Conv3d(256, 64, 1)
        self.aspp_gn = nn.GroupNorm(32, 64)
        self.attention4 = nn.Sequential(
            nn.Conv3d(192, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.Sigmoid()
        )

        self.attention3 = nn.Sequential(
            nn.Conv3d(192, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.Sigmoid()
        )
        self.attention2 = nn.Sequential(
            nn.Conv3d(192, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.Sigmoid()
        )
        self.attention1 = nn.Sequential(
            nn.Conv3d(192, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.Sigmoid()
        )

        self.refine4 = nn.Sequential(
            nn.Conv3d(192, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU()

        )
        self.refine3 = nn.Sequential(
            nn.Conv3d(192, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
        )
        self.refindown2 = nn.Sequential(
            nn.Conv3d(192, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU()
        )
        self.refindown1 = nn.Sequential(
            nn.Conv3d(192, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU()
        )
        self.refine = nn.Sequential(nn.Conv3d(256, 64, kernel_size=1),
                                    nn.GroupNorm(32, 64),
                                    nn.PReLU(), )
        # Bridge
        self.bridge = conv_block_2_3d(self.num_filters * 16, self.num_filters * 32, activation)

        # Up sampling
        self.trans_1 = conv_trans_block_3d(self.num_filters * 32, self.num_filters * 32, activation)
        self.up_1 = conv_block_2_3d(192, 64, activation)
        self.trans_2 = conv_trans_block_3d(self.num_filters * 16, self.num_filters * 16, activation)
        self.up_2 = conv_block_2_3d(192, 64, activation)
        self.trans_3 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 8, activation)
        self.up_3 = conv_block_2_3d(192, 64, activation)
        self.trans_4 = conv_trans_block_3d(self.num_filters * 4, self.num_filters * 4, activation)
        self.up_4 = conv_block_2_3d(192, 64, activation)
        self.trans_5 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 2, activation)
        self.up_5 = conv_block_2_3d(192, 64, activation)

        # Output
        self.predict = nn.Conv3d(64, 1, kernel_size=1)

    def forward(self, x):
        # Down sampling
        print("x shape")
        print(x.shape)

        layer_1 = self.layer_1(x) # -> [1, 4, 128, 128, 128]   [1, 4, 32, 320, 320]
        #down_1 = self.gn1(down_1)
        pool_1 = self.pool_1(layer_1) # -> [1, 4, 64, 64, 64]  [1, 4, 16, 160, 160]
        print("down1 shape")
        print(layer_1.shape)
        print("pool1 shape")
        print(pool_1.shape)
        layer_2 = self.layer_2(pool_1) # -> [1, 8, 64, 64, 64]  [1, 8, 16, 160, 160]
        #down_2 = self.gn2(down_2)
        #pool_2 = self.pool_2(down_2) # -> [1, 8, 32, 32, 32] [1, 8, 8, 80, 80]
        print("down2 shape")
        print(layer_2.shape)
        print("pool2 shape")
        #print(pool_2.shape)
        layer_3 = self.layer_3(layer_2) # -> [1, 16, 32, 32, 32] [1, 16, 8, 80, 80]
        #down_3 = self.gn3(down_3)
        #pool_3 = self.pool_3(down_3) # -> [1, 16, 16, 16, 16] [1, 16, 4, 40, 40]
        print("down3 shape")
        print(layer_3.shape)
        print("pool3 shape")
        #print(pool_3.shape)
        layer_4 = self.layer_4(layer_3) # -> [1, 32, 16, 16, 16]  [1, 32, 4, 40, 40]
        #down_4 = self.gn4(down_4)
        #pool_4 = self.pool_4(down_4) # -> [1, 32, 8, 8, 8]   [1, 32, 2, 20, 20]
        print("down4 shape")
        print(layer_4.shape)
        print("pool4 shape")
        #print(pool_4.shape)
        layer_5 = self.layer_5(layer_4) # -> [1, 64, 8, 8, 8]   [1, 64, 2, 20, 20]
       # down_5 = self.gn5(down_5)
        #pool_5 = self.pool_5(down_5) # -> [1, 64, 4, 4, 4]   [1, 64, 1, 10, 10]
        print("down5 shape")
        print(layer_5.shape)
        print("pool5 shape")
       # print(pool_5.shape)
        # Bridge
        gt_5 = self.down5(layer_5)
        print("gt5 shape")
        print(gt_5.shape)

        gt_4 = self.down4(layer_4)
        print("gt4 shape")
        print(gt_4.shape)
        gt_3 = self.down3(layer_3)
        print("gt3 shape")
        print(gt_3.shape)
        gt_2 = self.down2(layer_2)
        print("gt2 shape")
        print(gt_2.shape)
        #gt_2 = self.gt_2(gt_2)
        #print("gt2 shape")
       # print(gt_2.shape)


        sum_4 = gt_5
        print("sum4 shape")
        print(sum_4.shape)
        sum_3 = torch.add(F.upsample(sum_4, size=gt_4.size()[2:], mode='trilinear'),
            gt_4)
        print("sum3 shape")
        print(sum_3.shape)
        sum_2 = torch.add(F.upsample(sum_3, size=gt_3.size()[2:], mode='trilinear'),
            gt_3)
        print("sum2 shape")
        print(sum_2.shape)
        sum_1 = torch.add(F.upsample(sum_2, size=gt_2.size()[2:], mode='trilinear'),
            gt_2)
        print("sum1 shape")
        print(sum_1.shape)
        sum_4 = F.upsample(sum_4, size=sum_1.size()[2:], mode='trilinear')
        sum_3 = F.upsample(sum_3, size=sum_1.size()[2:], mode='trilinear')
        sum_2 = F.upsample(sum_2, size=sum_1.size()[2:], mode='trilinear')
        fuse = torch.cat((sum_1,sum_2,sum_3,sum_4),1)
        print("fuse shape")
        print(fuse.shape)
        fuse = self.fusdown1(fuse)
        print("fuse shape")
        print(fuse.shape)
        aspp1 = self.aspp1(fuse)
        print("aspp1 shape")
        print(aspp1.shape)
        aspp2 = self.aspp2(fuse)
        print("aspp2 shape")
        print(aspp2.shape)
        aspp3 = self.aspp3(fuse)
        print("aspp3 shape")
        print(aspp3.shape)
        aspp4 = self.aspp4(fuse)
        print("aspp4 shape")
        print(aspp4.shape)

        aspp = torch.cat((aspp1,aspp2,aspp3,aspp4),1)
        print("aspp shape")
        print(aspp.shape)
        aspp = self.aspp_conv(aspp)
        print("aspp shape")
        print(aspp.shape)
        aspp = self.aspp_gn(aspp)
        print("aspp shape")
        print(aspp.shape)
        attention4 = self.attention4(torch.cat((aspp,sum_4),1))
        print("attention4 shape")
        print(attention4.shape)
        attention3 = self.attention3(torch.cat((aspp,sum_3),1))
        print("attention3 shape")
        print(attention3.shape)
        attention2 = self.attention2(torch.cat((aspp,sum_2),1))
        print("attention2 shape")
        print(attention2.shape)
        attention1 = self.attention1(torch.cat((aspp,sum_1),1))
        print("attention1 shape")
        print(attention1.shape)

        refine4 = self.refine4(torch.cat((sum_4,attention4*fuse),1))
        print("refine4 shape")
        print(refine4.shape)
        refine3 = self.refine3(torch.cat((sum_3,attention3*fuse),1))
        print("refine3 shape")
        print(refine3.shape)
        refindown2 = self.refindown2(torch.cat((sum_2,attention2*fuse),1))
        print("refindown2 shape")
        print(refindown2.shape)
        refindown1 = self.refine4(torch.cat((sum_1,attention1*fuse),1))
        print("refindown1 shape")
        print(refindown1.shape)
        refine = self.refine(torch.cat((refine4,refine3,refindown2,refindown1),1))
        print("refine shape")
        print(refine.shape)
        predict1 = self.predict(refine)
        print("preedict1 shape")
        print(predict1.shape)

        # Up sampling

        concat_1 = torch.cat((refine, sum_4),1) # -> [1, 192, 8, 8, 8]  [1, 192, 2, 20, 20]
        up_1 = self.up_1(concat_1) # -> [1, 64, 8, 8, 8]   [1, 64, 2, 20, 20]
        print("concat_1 shape")
        print(concat_1.shape)
        print("up_1 shape")
        print(up_1.shape)
        concat_2 = torch.cat((up_1, sum_3),1) # -> [1, 96, 16, 16, 16]  [1, 96, 4, 40, 40]
        up_2 = self.up_2(concat_2) # -> [1, 32, 16, 16, 16]  [1, 32, 4, 40, 40]
        print("concat_2 shape")
        print(concat_2.shape)
        print("up_2 shape")
        print(up_2.shape)

        concat_3 = torch.cat((up_2, sum_2),1) # -> [1, 48, 32, 32, 32]   [1, 48, 8, 80, 80]
        up_3 = self.up_3(concat_3) # -> [1, 16, 32, 32, 32]  [1, 16, 8, 80, 80]

        print("concat_3 shape")
        print(concat_3.shape)
        print("up_3 shape")
        print(up_3.shape)
        concat_4 = torch.cat((up_3,sum_1), dim=1) # -> [1, 24, 64, 64, 64]   [1, 24, 16, 160, 160]
        up_4 = self.up_4(concat_4) # -> [1, 8, 64, 64, 64]  [1, 8, 16, 160, 160]
        print("concat_4 shape")
        print(concat_4.shape)
        print("up_4 shape")
        print(up_4.shape)

        predict2 = self.predict(up_4)
        print("preedict2 shape")
        print(predict2.shape)
        predict3 = torch.add(predict1*0.5,predict2*0.5)
        print("preedict3 shape")
        print(predict3.shape)
        predict1 = F.upsample(predict1, size=x.size()[2:], mode='trilinear')
        predict2 = F.upsample(predict2, size=x.size()[2:], mode='trilinear')
        predict3 = F.upsample(predict3, size=x.size()[2:], mode='trilinear')
        print("preedict3 shape")
        print(predict3.shape)
        # Output

        return predict1,predict2,predict3'''

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
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(16, out_ch),
            nn.ReLU(inplace=True))

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
            nn.GroupNorm(1, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(1, out_ch),
            nn.ReLU(inplace=True))

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
            # nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(16, out_ch),
            nn.ReLU(inplace=True),
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

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
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

        self.Maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv1 = conv_block2(img_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])
        self.do1 = nn.Dropout3d(0.5)

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

        self.extra_feature1 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)))
        self.extra_feature2 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)))
        self.extra_feature3 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)))
        self.extra_feature4 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)))
        self.extra_feature5 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)))
        self.extra_feature6 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)))
        self.extra_feature7 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)))
        self.extra_feature8 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)))
        self.fc1 = nn.Sequential(nn.Linear(688, 256), nn.Dropout(0.5))
        self.fc2 = nn.Sequential(nn.Linear(256, 2), nn.LogSoftmax())

        rates = (1, 6, 12, 18)

        self.aspp1 = ASPP_module(16, 16, rate=rates[0])
        self.aspp2 = ASPP_module(16, 16, rate=rates[1])
        self.aspp3 = ASPP_module(16, 16, rate=rates[2])
        self.aspp4 = ASPP_module(16, 16, rate=rates[3])

        self.aspp_conv = nn.Conv3d(64, 64, 1)
        self.aspp_gn = nn.GroupNorm(32, 64)
        self.Out = nn.Conv3d(64, output_ch, kernel_size=1, stride=1, padding=0)
        # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        print("x shape", x.shape)
        down1 = self.Conv1(x)
        print("down1 shape", down1.shape)
        feature5 = self.extra_feature1(down1)

        pool1 = self.Maxpool1(down1)
        print("pool1 shape", pool1.shape)
        down2 = self.Conv2(pool1)
        feature6 = self.extra_feature1(down2)

        print("down2 shape", down2.shape)

        pool2 = self.Maxpool2(down2)
        print("pool2 shape", pool2.shape)
        down3 = self.Conv3(pool2)
        feature7 = self.extra_feature1(down3)

        print("down3 shape", down3.shape)
        pool3 = self.Maxpool3(down3)
        print("pool3 shape", pool3.shape)
        down4 = self.Conv4(pool3)

        feature1 = self.extra_feature1(down4)

        print("down4 shape", down4.shape)
        pool4 = self.Maxpool4(down4)
        print("pool4 shape", pool4.shape)
        down5 = self.Conv5(pool4)
        print("down5 shape", down5.shape)

        feature2 = self.extra_feature2(down5)
        down5 = self.do1(down5)
        # print(x5.shape)
        down5_sample = F.upsample(down5, size=down4.size()[2:], mode='trilinear')
        up5 = self.Up5(down5_sample)
        print("up5 shape", up5.shape)
        # print(d5.shape)
        att4 = self.Att5(g=up5, x=down4)
        print("att4 shape", att4.shape)
        up5_cat = torch.cat((att4, up5), dim=1)
        print("up5_cat shape", up5_cat.shape)
        up5_conv = self.Up_conv5(up5_cat)

        feature3 = self.extra_feature3(up5_conv)

        print("up5_conv shape", up5_conv.shape)
        up5_sample = F.upsample(up5_conv, size=down3.size()[2:], mode='trilinear')
        up4 = self.Up4(up5_sample)
        print("up4 shape", up4.shape)
        att3 = self.Att4(g=up4, x=down3)
        print("att3 shape", att3.shape)
        up4_cat = torch.cat((att3, up4), dim=1)
        print("up4_cat shape", up4_cat.shape)
        up4_conv = self.Up_conv4(up4_cat)

        feature4 = self.extra_feature4(up4_conv)

        print("up4_conv shape", up4_conv.shape)
        up4_sample = F.upsample(up4_conv, size=down2.size()[2:], mode='trilinear')
        up3 = self.Up3(up4_sample)
        print("up3 shape", up3.shape)
        att2 = self.Att3(g=up3, x=down2)
        print("att2 shape", att2.shape)
        up3_cat = torch.cat((att2, up3), dim=1)
        print("up3_cat shape", up3_cat.shape)
        up3_conv = self.Up_conv3(up3_cat)
        print("up3_conv shape", up3_conv.shape)
        up3_sample = F.upsample(up3_conv, size=down1.size()[2:], mode='trilinear')
        up2 = self.Up2(up3_sample)
        print("up2 shape", up2.shape)
        att1 = self.Att2(g=up2, x=down1)
        print("att1 shape", att1.shape)
        up2_cat = torch.cat((att1, up2), dim=1)
        print("up2_cat shape", up2_cat.shape)
        up2_conv = self.Up_conv2(up2_cat)
        print("up2_conv.shape", up2_conv.shape)
        out1 = self.Conv(up2_conv)


        feature = torch.cat((feature1, feature2, feature3, feature4, feature5, feature6, feature7), 1)
        feature = feature.view(-1)
        print("feature shape", feature.shape)

        cls_out = self.fc1(feature)
        cls_out = self.fc2(cls_out)

        #  out = self.active(out)

        return out1, out1, cls_out