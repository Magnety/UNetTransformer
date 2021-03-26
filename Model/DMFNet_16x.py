import torch.nn as nn
import torch.nn.functional as F
import torch

try:
    from .sync_batchnorm import SynchronizedBatchNorm3d
except:
    pass

def normalization(planes, norm='bn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(4, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    elif norm == 'sync_bn':
        m = SynchronizedBatchNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m

class Conv3d_Block(nn.Module):
    def __init__(self,num_in,num_out,kernel_size=1,stride=1,g=1,padding=None,norm=None):
        super(Conv3d_Block, self).__init__()
        if padding == None:
            padding = (kernel_size - 1) // 2
        self.bn = normalization(num_in,norm=norm)
        self.act_fn = nn.PReLU()
        self.conv = nn.Conv3d(num_in, num_out, kernel_size=kernel_size, padding=padding,stride=stride, groups=g, bias=False)

    def forward(self, x): # BN + Relu + Conv
        h = self.act_fn(self.bn(x))
        h = self.conv(h)
        return h


class DilatedConv3DBlock(nn.Module):
    def __init__(self, num_in, num_out, kernel_size=(1,1,1), stride=1, g=1, d=(1,1,1), norm=None):
        super(DilatedConv3DBlock, self).__init__()
        assert isinstance(kernel_size,tuple) and isinstance(d,tuple)

        padding = tuple(
            [(ks-1)//2 *dd for ks, dd in zip(kernel_size, d)]
        )

        self.bn = normalization(num_in, norm=norm)
        self.act_fn = nn.PReLU()
        self.conv = nn.Conv3d(num_in,num_out,kernel_size=kernel_size,padding=padding,stride=stride,groups=g,dilation=d,bias=False)

    def forward(self, x):
        h = self.act_fn(self.bn(x))
        h = self.conv(h)
        return h


class MFunit(nn.Module):
    def __init__(self, num_in, num_out, g=1, stride=1, d=(1,1),norm=None):
        """  The second 3x3x1 group conv is replaced by 3x3x3.
        :param num_in: number of input channels
        :param num_out: number of output channels
        :param g: groups of group conv.
        :param stride: 1 or 2
        :param d: tuple, d[0] for the first 3x3x3 conv while d[1] for the 3x3x1 conv
        :param norm: Batch Normalization
        """
        super(MFunit, self).__init__()
        num_mid = num_in if num_in <= num_out else num_out
        self.conv1x1x1_in1 = Conv3d_Block(num_in,num_in//4,kernel_size=1,stride=1,norm=norm)
        self.conv1x1x1_in2 = Conv3d_Block(num_in//4,num_mid,kernel_size=1,stride=1,norm=norm)
        self.conv3x3x3_m1 = DilatedConv3DBlock(num_mid,num_out,kernel_size=(3,3,3),stride=stride,g=g,d=(d[0],d[0],d[0]),norm=norm) # dilated
        self.conv3x3x3_m2 = DilatedConv3DBlock(num_out,num_out,kernel_size=(3,3,1),stride=1,g=g,d=(d[1],d[1],1),norm=norm)
        # self.conv3x3x3_m2 = DilatedConv3DBlock(num_out,num_out,kernel_size=(1,3,3),stride=1,g=g,d=(1,d[1],d[1]),norm=norm)

        # skip connection
        if num_in != num_out or stride != 1:
            if stride == 1:
                self.conv1x1x1_shortcut = Conv3d_Block(num_in, num_out, kernel_size=1, stride=1, padding=0,norm=norm)
            if stride == 2:
                # if MF block with stride=2, 2x2x2
                self.conv2x2x2_shortcut = Conv3d_Block(num_in, num_out, kernel_size=2, stride=2,padding=0, norm=norm) # params

    def forward(self, x):
        x1 = self.conv1x1x1_in1(x)
        x2 = self.conv1x1x1_in2(x1)
        x3 = self.conv3x3x3_m1(x2)
        x4 = self.conv3x3x3_m2(x3)

        shortcut = x

        if hasattr(self,'conv1x1x1_shortcut'):
            shortcut = self.conv1x1x1_shortcut(shortcut)
        if hasattr(self,'conv2x2x2_shortcut'):
            shortcut = self.conv2x2x2_shortcut(shortcut)

        return x4 + shortcut
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
            nn.PReLU(),
            nn.Dropout3d(0.3)
        )

    def forward(self, x):
        x = self.up(x)
        return x
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
            nn.Dropout3d(0.3),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(16, out_ch),
            nn.PReLU())

    def forward(self, x):
        x = self.conv(x)
        return x
class DMFUnit(nn.Module):
    # weighred add
    def __init__(self, num_in, num_out, g=1, stride=1,norm=None,dilation=None):
        super(DMFUnit, self).__init__()
        self.weight1 = nn.Parameter(torch.ones(1))
        self.weight2 = nn.Parameter(torch.ones(1))
        self.weight3 = nn.Parameter(torch.ones(1))

        num_mid = num_in if num_in <= num_out else num_out

        self.conv1x1x1_in1 = Conv3d_Block(num_in, num_in // 4, kernel_size=1, stride=1, norm=norm)
        self.conv1x1x1_in2 = Conv3d_Block(num_in // 4,num_mid,kernel_size=1, stride=1, norm=norm)

        self.conv3x3x3_m1 = nn.ModuleList()
        if dilation == None:
            dilation = [1,2,3]
        for i in range(3):
            self.conv3x3x3_m1.append(
                DilatedConv3DBlock(num_mid,num_out, kernel_size=(3, 3, 3), stride=stride, g=g, d=(dilation[i],dilation[i], dilation[i]),norm=norm)
            )

        # It has not Dilated operation
        #self.conv3x3x3_m2 = DilatedConv3DBlock(num_out, num_out, kernel_size=(3, 3, 1), stride=(1,1,1), g=g,d=(1,1,1), norm=norm)
        self.conv3x3x3_m2 = DilatedConv3DBlock(num_out, num_out, kernel_size=(1, 3, 3), stride=(1,1,1), g=g,d=(1,1,1), norm=norm)

        # skip connection
        if num_in != num_out or stride != 1:
            if stride == 1:
                self.conv1x1x1_shortcut = Conv3d_Block(num_in, num_out, kernel_size=1, stride=1, padding=0, norm=norm)
            if stride == 2:
                self.conv2x2x2_shortcut = Conv3d_Block(num_in, num_out, kernel_size=2, stride=2, padding=0, norm=norm)


    def forward(self, x):
        print("111111111111111111111111111")
        print(x.shape)

        x1 = self.conv1x1x1_in1(x)
        print(x1.shape)
        x2 = self.conv1x1x1_in2(x1)
        print(x2.shape)

        x3 = self.weight1*self.conv3x3x3_m1[0](x2) + self.weight2*self.conv3x3x3_m1[1](x2) + self.weight3*self.conv3x3x3_m1[2](x2)
        print(x3.shape)

        x4 = self.conv3x3x3_m2(x3)
        print(x4.shape)

        shortcut = x
        if hasattr(self, 'conv1x1x1_shortcut'):
            shortcut = self.conv1x1x1_shortcut(shortcut)
        if hasattr(self, 'conv2x2x2_shortcut'):
            shortcut = self.conv2x2x2_shortcut(shortcut)
        print(shortcut.shape)
        shortcut = F.upsample(shortcut,size=x4.size()[2:], mode='trilinear')
        print(shortcut.shape)

        return x4 + shortcut
class New_DMFNet(nn.Module):
    def __init__(self,in_channel=1,mid_channel=32,channels=32,groups=16,norm='gn',out_classes=2):
        super(New_DMFNet, self).__init__()
        self.in_conv = nn.Conv3d(in_channel,mid_channel,kernel_size=3,padding=1,stride=2,bias=False)
        self.encoder_block1 = nn.Sequential(
            DMFUnit(mid_channel, channels, g=groups, stride=2, norm=norm, dilation=[1, 2, 3]),
            DMFUnit(channels, channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),
            DMFUnit(channels, channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),
        )
        self.encoder_block2 = nn.Sequential(
            DMFUnit(channels, 2 * channels, g=groups, stride=2, norm=norm, dilation=[1, 2, 3]),
            DMFUnit(2 * channels, 2 * channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),
            DMFUnit(2 * channels, 2 * channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),
        )
        self.encoder_block3 = nn.Sequential(
            DMFUnit(2 * channels, 3 * channels, g=groups, stride=2, norm=norm, dilation=[1, 2, 3]),
            DMFUnit(3 * channels, 3 * channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),
            DMFUnit(3 * channels, 3 * channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),
        )


        self.recons_mfunit1 = MFunit(channels, mid_channel, g=groups, stride=1, norm=norm)
        self.recons_mfunit2 = MFunit(2 * channels, channels, g=groups, stride=1, norm=norm)




        self.decoder_mfunit1 = MFunit(channels * 3, channels * 2, g=groups, stride=1, norm=norm)
        self.decoder_mfunit2 = MFunit(channels *2+channels * 2, channels , g=groups, stride=1, norm=norm)
        self.decoder_mfunit3 = MFunit(channels +channels , channels, g=groups, stride=1, norm=norm)
        self.decoder_mfunit4 = MFunit(mid_channel+channels, mid_channel, g=groups, stride=1, norm=norm)
        self.out_conv = nn.Sequential(nn.Conv3d(mid_channel, 1, kernel_size=1, padding=0, stride=1, bias=False))

        self.extra_feature1 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)))
        self.extra_feature2 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)))
        self.extra_feature3 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)))
        self.extra_feature4 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)))
        self.extra_feature5 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)))

        self.fc1 = nn.Sequential(nn.Linear(384, 256), nn.Dropout(0.5))
        self.fc2 = nn.Sequential(nn.Linear(256, 128), nn.Dropout(0.5))
        self.fc3 = nn.Sequential(nn.Linear(128, 2), nn.LogSoftmax())

    def forward(self, x):
        print(111111111111111111111111111)
        print(x.shape)
        encoder1 = self.in_conv(x)
        print(encoder1.shape)
        encoder2 = self.encoder_block1(encoder1)
        feature1 = self.extra_feature1(encoder2)

        encoder3 = self.encoder_block2(encoder2)
        feature2 = self.extra_feature2(encoder3)

        encoder4 = self.encoder_block3(encoder3)
        feature3 = self.extra_feature3(encoder4)

        recons_up1 = F.upsample(encoder2, size=encoder1.size()[2:], mode='trilinear')
        recons1 = self.recons_mfunit1(recons_up1)

        recons_up2 = F.upsample(encoder3, size=encoder2.size()[2:], mode='trilinear')
        recons2 = self.recons_mfunit2(recons_up2)

        decoder_up1 = F.upsample(encoder4, size=encoder3.size()[2:], mode='trilinear')
        decoder_mfunit1 = self.decoder_mfunit1(decoder_up1)
        decoder_cat1 = torch.cat((decoder_mfunit1, encoder3), dim=1)

        feature4 = self.extra_feature4(decoder_cat1)

        decoder_up2 = F.upsample(decoder_cat1, size=encoder2.size()[2:], mode='trilinear')
        print("22222222222222")
        print(decoder_up2.shape)
        print(encoder2.shape)

        decoder_mfunit2 = self.decoder_mfunit2(decoder_up2)
        decoder_cat2 = torch.cat((decoder_mfunit2, encoder2), dim=1)

        feature5 = self.extra_feature5(decoder_cat2)

        decoder_up3 = F.upsample(decoder_cat2, size=encoder1.size()[2:], mode='trilinear')
        decoder_mfunit3 = self.decoder_mfunit3(decoder_up3)
        decoder_cat3 = torch.cat((decoder_mfunit3, encoder1), dim=1)

        decoder_up4 = F.upsample(decoder_cat3, size=x.size()[2:], mode='trilinear')
        decoder_mfunit4 =self.decoder_mfunit4(decoder_up4)

        features = torch.cat((feature1, feature2, feature3, feature4, feature5), 1)
        features = features.view(-1)

        print(features.shape)
        cls_out = self.fc1(features)
        cls_out = self.fc2(cls_out)
        cls_out = self.fc3(cls_out)

        seg_out = self.out_conv(decoder_mfunit4)

        return encoder1, recons1, encoder2, recons2, seg_out, seg_out, cls_out
class New_Att_DMFNet(nn.Module):
    def __init__(self,in_channel=1,mid_channel=32,channels=64,groups=16,norm='gn',out_classes=2):
        super(New_Att_DMFNet, self).__init__()
        self.in_conv = nn.Conv3d(in_channel,mid_channel,kernel_size=3,padding=1,stride=2,bias=False)

        self.encoder_block1 = nn.Sequential(
            DMFUnit(mid_channel, channels, g=groups, stride=2, norm=norm, dilation=[1, 2, 3]),
            DMFUnit(channels, channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),
            DMFUnit(channels, channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),
        )
        self.encoder_block2 = nn.Sequential(
            DMFUnit(channels, 2 * channels, g=groups, stride=2, norm=norm, dilation=[1, 2, 3]),
            DMFUnit(2 * channels, 2 * channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),
            DMFUnit(2 * channels, 2 * channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),
        )
        self.encoder_block3 = nn.Sequential(
            DMFUnit(2 * channels, 3 * channels, g=groups, stride=2, norm=norm, dilation=[1, 2, 3]),
            DMFUnit(3 * channels, 3 * channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),
            DMFUnit(3 * channels, 3 * channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),
        )
        self.recons_mfunit1 = MFunit(channels, mid_channel, g=groups, stride=1, norm=norm)
        self.recons_mfunit2 = MFunit(2 * channels, channels, g=groups, stride=1, norm=norm)
        self.Up1 = up_conv(3*channels,2*channels)
        self.att1 = Attention_block(F_g=channels*2,F_l=channels*2,F_int=channels*2)
        self.Up2 = up_conv(2*channels,1*channels)
        self.att2 = Attention_block(F_g=channels,F_l=channels,F_int=channels)
        self.Up3 = up_conv(channels, mid_channel)
        self.att3 = Attention_block(F_g=mid_channel, F_l=mid_channel, F_int=mid_channel)
        self.decoder_mfunit1 = MFunit(channels * 2+channels * 2, channels * 2, g=groups, stride=1, norm=norm)
        self.decoder_mfunit2 = MFunit(channels +channels , channels , g=groups, stride=1, norm=norm)
        self.decoder_mfunit3 = MFunit(mid_channel +mid_channel, mid_channel, g=groups, stride=1, norm=norm)
        self.decoder_mfunit4 = MFunit(mid_channel+mid_channel, mid_channel, g=groups, stride=1, norm=norm)
        self.out_conv = nn.Sequential(nn.Conv3d(mid_channel, 1, kernel_size=1, padding=0, stride=1, bias=False))

        self.extra_feature1 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)))
        self.extra_feature2 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)))
        self.extra_feature3 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)))
        self.extra_feature4 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)))
        self.extra_feature5 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)))

        self.fc1 = nn.Sequential(nn.Linear(768, 256), nn.Dropout(0.5))
        self.fc2 = nn.Sequential(nn.Linear(256, 128), nn.Dropout(0.5))
        self.fc3 = nn.Sequential(nn.Linear(128, 2), nn.LogSoftmax())

    def forward(self, x):
        print(111111111111111111111111111)
        print(x.shape)
        encoder1 = self.in_conv(x)
        print(encoder1.shape)
        encoder2 = self.encoder_block1(encoder1)
        feature1 = self.extra_feature1(encoder2)
        encoder3 = self.encoder_block2(encoder2)
        feature2 = self.extra_feature2(encoder3)
        encoder4 = self.encoder_block3(encoder3)
        feature3 = self.extra_feature3(encoder4)
        recons_up1 = F.upsample(encoder2, size=encoder1.size()[2:], mode='trilinear')
        recons1 = self.recons_mfunit1(recons_up1)
        recons_up2 = F.upsample(encoder3, size=encoder2.size()[2:], mode='trilinear')
        recons2 = self.recons_mfunit2(recons_up2)
        decoder_up1 = F.upsample(encoder4, size=encoder3.size()[2:], mode='trilinear')
        up1 = self.Up1(decoder_up1)
        att1 = self.att1(g=up1,x=encoder3)
        decoder_cat1 = torch.cat((up1, att1), dim=1)
        decoder_mfunit1 = self.decoder_mfunit1(decoder_cat1)
        feature4 = self.extra_feature4(decoder_cat1)
        decoder_up2 = F.upsample(decoder_mfunit1, size=encoder2.size()[2:], mode='trilinear')
        print("22222222222222")
        print(decoder_up2.shape)
        print(encoder2.shape)
        up2 = self.Up2(decoder_up2)
        att2 = self.att2(g=up2, x=encoder2)
        decoder_cat2 = torch.cat((up2, att2), dim=1)
        decoder_mfunit2 = self.decoder_mfunit2(decoder_cat2)
        feature5 = self.extra_feature5(decoder_cat2)
        decoder_up3 = F.upsample(decoder_mfunit2, size=encoder1.size()[2:], mode='trilinear')
        up3 = self.Up3(decoder_up3)
        att3 = self.att3(g=up3, x=encoder1)
        decoder_cat3 = torch.cat((up3, att3), dim=1)
        decoder_mfunit3 = self.decoder_mfunit3(decoder_cat3)
        decoder_up4 = F.upsample(decoder_mfunit3, size=x.size()[2:], mode='trilinear')
        features = torch.cat((feature1, feature2, feature3, feature4, feature5), 1)
        features = features.view(-1)
        print(features.shape)
        cls_out = self.fc1(features)
        cls_out = self.fc2(cls_out)
        cls_out = self.fc3(cls_out)
        seg_out = self.out_conv(decoder_up4)
        return encoder1, recons1, encoder2, recons2, seg_out, seg_out, cls_out




class New_DMFNet_NewRecons(nn.Module):
    def __init__(self,in_channel=1,mid_channel=32,channels=32,groups=16,norm='gn',out_classes=2):
        super(New_DMFNet_NewRecons, self).__init__()
        self.in_conv = nn.Conv3d(in_channel,mid_channel,kernel_size=3,padding=1,stride=2,bias=False)
        self.encoder_block1 = nn.Sequential(
            DMFUnit(mid_channel, channels, g=groups, stride=2, norm=norm, dilation=[1, 2, 3]),
            DMFUnit(channels, channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),
            DMFUnit(channels, channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),
        )
        self.encoder_block2 = nn.Sequential(
            DMFUnit(channels, 2 * channels, g=groups, stride=2, norm=norm, dilation=[1, 2, 3]),
            DMFUnit(2 * channels, 2 * channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),
            DMFUnit(2 * channels, 2 * channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),
        )
        self.encoder_block3 = nn.Sequential(
            DMFUnit(2 * channels, 3 * channels, g=groups, stride=2, norm=norm, dilation=[1, 2, 3]),
            DMFUnit(3 * channels, 3 * channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),
            DMFUnit(3 * channels, 3 * channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),
        )

        self.recons_mfunit1 = MFunit(channels * 3, channels * 2, g=groups, stride=1, norm=norm)
        self.recons_mfunit2 = MFunit(channels * 2 , channels, g=groups, stride=1, norm=norm)
        self.recons_mfunit3 = MFunit(channels ,mid_channel, g=groups, stride=1, norm=norm)
        self.recons_mfunit4 = nn.Sequential(nn.Conv3d(mid_channel, 1, kernel_size=1, padding=0, stride=1, bias=False))




        self.decoder_mfunit1 = MFunit(channels * 3, channels * 2, g=groups, stride=1, norm=norm)
        self.decoder_mfunit2 = MFunit(channels *2+channels * 2, channels , g=groups, stride=1, norm=norm)
        self.decoder_mfunit3 = MFunit(channels +channels , channels, g=groups, stride=1, norm=norm)
        self.decoder_mfunit4 = MFunit(mid_channel+channels, mid_channel, g=groups, stride=1, norm=norm)
        self.out_conv = nn.Sequential(nn.Conv3d(mid_channel, 1, kernel_size=1, padding=0, stride=1, bias=False))

        self.extra_feature1 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)))
        self.extra_feature2 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)))
        self.extra_feature3 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)))
        self.extra_feature4 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)))
        self.extra_feature5 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)))

        self.fc1 = nn.Sequential(nn.Linear(384, 256), nn.Dropout(0.5))
        self.fc2 = nn.Sequential(nn.Linear(256, 128), nn.Dropout(0.5))
        self.fc3 = nn.Sequential(nn.Linear(128, 2), nn.LogSoftmax())

    def forward(self, x):
        print(111111111111111111111111111)
        print(x.shape)
        encoder1 = self.in_conv(x)
        print(encoder1.shape)
        encoder2 = self.encoder_block1(encoder1)
        feature1 = self.extra_feature1(encoder2)

        encoder3 = self.encoder_block2(encoder2)
        feature2 = self.extra_feature2(encoder3)

        encoder4 = self.encoder_block3(encoder3)
        feature3 = self.extra_feature3(encoder4)

        recons_up1 = F.upsample(encoder4, size=encoder3.size()[2:], mode='trilinear')
        recons1 = self.recons_mfunit1(recons_up1)
        recons_up2 = F.upsample(recons1,size=encoder2.size()[2:], mode='trilinear')
        recons2 = self.recons_mfunit2(recons_up2)
        recons_up3 = F.upsample(recons2,size=encoder1.size()[2:], mode='trilinear')
        recons3 = self.recons_mfunit3(recons_up3)
        recons_up4 = F.upsample(recons3,size=x.size()[2:], mode='trilinear')
        recons4 = self.recons_mfunit4(recons_up4)

        decoder_up1 = F.upsample(encoder4, size=encoder3.size()[2:], mode='trilinear')
        decoder_mfunit1 = self.decoder_mfunit1(decoder_up1)
        decoder_cat1 = torch.cat((decoder_mfunit1, encoder3), dim=1)

        feature4 = self.extra_feature4(decoder_cat1)

        decoder_up2 = F.upsample(decoder_cat1, size=encoder2.size()[2:], mode='trilinear')
        print("22222222222222")
        print(decoder_up2.shape)
        print(encoder2.shape)

        decoder_mfunit2 = self.decoder_mfunit2(decoder_up2)
        decoder_cat2 = torch.cat((decoder_mfunit2, encoder2), dim=1)

        feature5 = self.extra_feature5(decoder_cat2)

        decoder_up3 = F.upsample(decoder_cat2, size=encoder1.size()[2:], mode='trilinear')
        decoder_mfunit3 = self.decoder_mfunit3(decoder_up3)
        decoder_cat3 = torch.cat((decoder_mfunit3, encoder1), dim=1)

        decoder_up4 = F.upsample(decoder_cat3, size=x.size()[2:], mode='trilinear')
        decoder_mfunit4 =self.decoder_mfunit4(decoder_up4)

        features = torch.cat((feature1, feature2, feature3, feature4, feature5), 1)
        features = features.view(-1)

        print(features.shape)
        cls_out = self.fc1(features)
        cls_out = self.fc2(cls_out)
        cls_out = self.fc3(cls_out)

        seg_out = self.out_conv(decoder_mfunit4)

        return recons4, seg_out, seg_out, cls_out
        

class My_DFMNet(nn.Module):
    def __init__(self,in_channel=1,mid_channel=32,channels=128,groups=16,norm='gn',out_classes=2):
        super(My_DFMNet,self).__init__()
        self.in_conv = nn.Conv3d(in_channel,mid_channel,kernel_size=3,padding=1,stride=2,bias=False)
        self.encoder_block1 = nn.Sequential(
            DMFUnit(mid_channel,channels,g=groups,stride=2,norm=norm,dilation=[1,2,3]),
            DMFUnit(channels,channels,g=groups,stride=1,norm=norm,dilation=[1,2,3]),
            DMFUnit(channels, channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),
        )
        self.encoder_block2 = nn.Sequential(
            DMFUnit(channels, 2*channels, g=groups, stride=2, norm=norm, dilation=[1, 2, 3]),
            DMFUnit(2*channels, 2*channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),
            DMFUnit(2*channels, 2*channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),
            nn.Dropout3d(0.3)
        )
        self.encoder_block3 = nn.Sequential(
            DMFUnit(2*channels, 3 * channels, g=groups, stride=2, norm=norm, dilation=[1, 2, 3]),
            DMFUnit(3 * channels, 3 * channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),
            DMFUnit(3 * channels, 3 * channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),
        )
        self.mfunit_block1 = MFunit(mid_channel,mid_channel,g=groups,stride=1,norm=norm)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False) # H//8
        self.mfunit_block2 = MFunit(channels,mid_channel,g=groups,stride=1,norm=norm)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False) # H//8

        self.decoder_up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.decoder_mfunit1 = MFunit(channels*3,channels*2,g=groups,stride=1,norm=norm)
        self.decoder_up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.decoder_mfunit2 = MFunit(channels*2+channels*2,channels*2,g=groups,stride=1,norm=norm)

        self.decoder_up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.decoder_mfunit3 = MFunit(channels*2+channels,channels*1,g=groups,stride=1,norm=norm)

        self.decoder_conv = nn.Conv3d(channels+mid_channel,mid_channel,kernel_size=1, padding=0,stride=1,bias=False)

        self.decoder_up4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        self.out_conv = nn.Sequential(nn.Conv3d(mid_channel,1,kernel_size=1, padding=0,stride=1,bias=False)
                                      )

        self.extra_feature1 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)))
        self.extra_feature2 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)))
        self.extra_feature3 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)))
        self.extra_feature4 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)))
        self.extra_feature5 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)))
        self.extra_feature6 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)))

        self.fc1 = nn.Sequential(nn.Linear(1824, 512), nn.Dropout(0.3))
        self.fc2 = nn.Sequential(nn.Linear(512, 256), nn.Dropout(0.2))

        self.fc3 = nn.Sequential(nn.Linear(256, 2), nn.LogSoftmax())

    def forward(self,x):
        in_conv = self.in_conv(x)
        encoder1 = self.encoder_block1(in_conv)
        encoder2 = self.encoder_block2(encoder1)
        encoder3 = self.encoder_block3(encoder2)
        feature1 = self.extra_feature2(encoder1)

        feature2 = self.extra_feature2(encoder2)
        feature3 = self.extra_feature3(encoder3)

        #decoder_up1 = self.decoder_up1(encoder3)
        decoder_up1 = F.upsample(encoder3, size=encoder2.size()[2:], mode='trilinear')
        decoder_mfunit1 = self.decoder_mfunit1(decoder_up1)
        decoder_cat1 = torch.cat((decoder_mfunit1,encoder2),dim=1)
        feature4 = self.extra_feature4(decoder_cat1)

        #decoder_up2 = self.decoder_up2(decoder_cat1)
        decoder_up2 = F.upsample(decoder_cat1, size=encoder1.size()[2:], mode='trilinear')
        decoder_mfunit2 = self.decoder_mfunit2(decoder_up2)


        decoder_cat2 = torch.cat((decoder_mfunit2,encoder1),dim=1)
        feature5 = self.extra_feature5(decoder_cat2)

        #decoder_up3 = self.decoder_up3(decoder_cat2)
        decoder_up3 = F.upsample(decoder_cat2, size=in_conv.size()[2:], mode='trilinear')

        decoder_mfunit3 = self.decoder_mfunit3(decoder_up3)
        decoder_cat3 = torch.cat((decoder_mfunit3,in_conv),dim=1)
        feature6 = self.extra_feature6(decoder_cat3)

        decoder_conv = self.decoder_conv(decoder_cat3)
        #decoder_up4 = self.decoder_up4(decoder_conv)
        decoder_up4 = F.upsample(decoder_conv, size=x.size()[2:], mode='trilinear')
        seg_out = self.out_conv(decoder_up4)
        features = torch.cat((feature1, feature2, feature3,feature4,feature5,feature6), 1)
        features = features.view(-1)
        print(features.shape)
        cls_out = self.fc1(features)
        cls_out = self.fc2(cls_out)
        cls_out = self.fc3(cls_out)

        return seg_out,seg_out,cls_out


class MFNet(nn.Module): #
    # [96]   Flops:  13.361G  &  Params: 1.81M
    # [112]  Flops:  16.759G  &  Params: 2.46M
    # [128]  Flops:  20.611G  &  Params: 3.19M
    def __init__(self, c=4,n=32,channels=128,groups = 16,norm='bn', num_classes=4):
        super(MFNet, self).__init__()

        # Entry flow
        self.encoder_block1 = nn.Conv3d( c, n, kernel_size=3, padding=1, stride=2, bias=False)# H//2
        self.encoder_block2 = nn.Sequential(
            MFunit(n, channels, g=groups, stride=2, norm=norm),# H//4 down
            MFunit(channels, channels, g=groups, stride=1, norm=norm),
            MFunit(channels, channels, g=groups, stride=1, norm=norm)
        )
        #
        self.encoder_block3 = nn.Sequential(
            MFunit(channels, channels*2, g=groups, stride=2, norm=norm), # H//8
            MFunit(channels * 2, channels * 2, g=groups, stride=1, norm=norm),
            MFunit(channels * 2, channels * 2, g=groups, stride=1, norm=norm)
        )

        self.encoder_block4 = nn.Sequential(# H//8,channels*4
            MFunit(channels*2, channels*3, g=groups, stride=2, norm=norm), # H//16
            MFunit(channels*3, channels*3, g=groups, stride=1, norm=norm),
            MFunit(channels*3, channels*2, g=groups, stride=1, norm=norm),
        )

        self.upsample1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False) # H//8
        self.decoder_block1 = MFunit(channels*2+channels*2, channels*2, g=groups, stride=1, norm=norm)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False) # H//4
        self.decoder_block2 = MFunit(channels*2 + channels, channels, g=groups, stride=1, norm=norm)

        self.upsample3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False) # H//2
        self.decoder_block3 = MFunit(channels + n, n, g=groups, stride=1, norm=norm)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False) # H
        self.seg = nn.Conv3d(n, num_classes, kernel_size=1, padding=0,stride=1,bias=False)

        self.softmax = nn.Softmax(dim=1)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.torch.nn.init.kaiming_normal_(m.weight) #
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm) or isinstance(m, SynchronizedBatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        x1 = self.encoder_block1(x)# H//2 down
        x2 = self.encoder_block2(x1)# H//4 down
        x3 = self.encoder_block3(x2)# H//8 down
        x4 = self.encoder_block4(x3) # H//16
        # Decoder
        y1 = self.upsample1(x4)# H//8
        y1 = torch.cat([x3,y1],dim=1)
        y1 = self.decoder_block1(y1)

        y2 = self.upsample2(y1)# H//4
        y2 = torch.cat([x2,y2],dim=1)
        y2 = self.decoder_block2(y2)

        y3 = self.upsample3(y2)# H//2
        y3 = torch.cat([x1,y3],dim=1)
        y3 = self.decoder_block3(y3)
        y4 = self.upsample4(y3)
        y4 = self.seg(y4)
        if hasattr(self,'softmax'):
            y4 = self.softmax(y4)
        return y4


class DMFNet(MFNet): # softmax
    # [128]  Flops:  27.045G  &  Params: 3.88M
    def __init__(self, c=4,n=32,channels=128, groups=16,norm='bn', num_classes=4):
        super(DMFNet, self).__init__(c,n,channels,groups, norm, num_classes)

        self.encoder_block2 = nn.Sequential(
            DMFUnit(n, channels, g=groups, stride=2, norm=norm,dilation=[1,2,3]),# H//4 down
            DMFUnit(channels, channels, g=groups, stride=1, norm=norm,dilation=[1,2,3]), # Dilated Conv 3
            DMFUnit(channels, channels, g=groups, stride=1, norm=norm,dilation=[1,2,3])
        )

        self.encoder_block3 = nn.Sequential(
            DMFUnit(channels, channels*2, g=groups, stride=2, norm=norm,dilation=[1,2,3]), # H//8
            DMFUnit(channels * 2, channels * 2, g=groups, stride=1, norm=norm,dilation=[1,2,3]),# Dilated Conv 3
            DMFUnit(channels * 2, channels * 2, g=groups, stride=1, norm=norm,dilation=[1,2,3])
        )


class PositionAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(PositionAttentionModule, self).__init__()
        self.first_branch_conv = nn.Conv3d(in_channels, int(in_channels / 8), kernel_size=1)
        self.second_branch_conv = nn.Conv3d(in_channels, int(in_channels / 8), kernel_size=1)
        self.third_branch_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.output_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)

    def forward(self, F):
        # first branch
        F1 = self.first_branch_conv(F)  # (C/8, W, H)
        F1 = F1.reshape((F1.size(0), F1.size(1), -1))  # (C/8, W*H)
        F1 = torch.transpose(F1, -2, -1)  # (W*H, C/8)
        # second branch
        F2 = self.second_branch_conv(F)  # (C/8, W, H)
        F2 = F2.reshape((F2.size(0), F2.size(1),-1))  # (C/8, W*H)
        F2 = nn.Softmax(dim=-1)(torch.matmul(F1, F2))  # (W*H, W*H)
        # third branch
        F3 = self.third_branch_conv(F)  # (C, W, H)
        F3 = F3.reshape((F3.size(0), F3.size(1), -1))  # (C, W*H)
        F3 = torch.matmul(F3, F2)  # (C, W*H)
        F3 = F3.reshape(F.shape)  # (C, W, H)
        return self.output_conv(F3 * F)


class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttentionModule, self).__init__()
        self.output_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)

    def forward(self, F):
        # first branch
        F1 = F.reshape((F.size(0), F.size(1), -1))  # (C, W*H)
        F1 = torch.transpose(F1, -2, -1)  # (W*H, C)
        # second branch
        F2 = F.reshape((F.size(0), F.size(1), -1))  # (C, W*H)
        F2 = nn.Softmax(dim=-1)(torch.matmul(F2, F1))  # (C, C)
        # third branch
        F3 = F.reshape((F.size(0), F.size(1), -1))  # (C, W*H)
        F3 = torch.matmul(F2, F3)  # (C, W*H)
        F3 = F3.reshape(F.shape)  # (C, W, H)
        return self.output_conv(F3 * F)


class GuidedAttentionModule(nn.Module):
    def __init__(self, in_channels_F, in_channels_Fms):
        super(GuidedAttentionModule, self).__init__()
        in_channels = in_channels_F + in_channels_Fms
        self.pam = PositionAttentionModule(in_channels)
        self.cam = ChannelAttentionModule(in_channels)
        self.attention_map_conv = nn.Sequential(nn.Conv3d(in_channels, in_channels_Fms, kernel_size=1),
                                                nn.BatchNorm3d(in_channels_Fms),
                                                nn.ReLU())
    def forward(self, F, F_ms):
        _F = torch.cat((F, F_ms), dim=1)  # concatenate the extracted feature map with the multi scale feature map
        F_pcam =  self.cam(_F)# sum the ouputs of the position and channel attention modules
        F_output = self.attention_map_conv(_F * F_pcam)
        F_output = F_output * F
        return F_output


class DMF_FPN(nn.Module):
    def __init__(self, in_channel=1, mid_channel=64, channels=64, groups=16, norm='gn', out_classes=2):
        super(DMF_FPN, self).__init__()
        self.in_conv = nn.Conv3d(in_channel, mid_channel, kernel_size=3, padding=1, stride=2, bias=False)

        self.encoder_block1 = nn.Sequential(
            DMFUnit(mid_channel, channels, g=groups, stride=2, norm=norm, dilation=[1, 2, 3]),
            DMFUnit(channels, channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),
            DMFUnit(channels, channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),
        )
        self.encoder_block2 = nn.Sequential(
            DMFUnit(channels, 2 * channels, g=groups, stride=2, norm=norm, dilation=[1, 2, 3]),
            DMFUnit(2 * channels, 2 * channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),
            DMFUnit(2 * channels, 2 * channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),
        )
        self.encoder_block3 = nn.Sequential(
            DMFUnit(2 * channels, 3 * channels, g=groups, stride=2, norm=norm, dilation=[1, 2, 3]),
            DMFUnit(3 * channels, 3 * channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),
            DMFUnit(3 * channels, 3 * channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),
        )
        self.encoder_block4 = nn.Sequential(
            DMFUnit(3 * channels, 4 * channels, g=groups, stride=2, norm=norm, dilation=[1, 2, 3]),
            DMFUnit(4 * channels, 4 * channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),
            DMFUnit(4 * channels, 4 * channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),
        )
        self.down1 = nn.Sequential(
            nn.Conv3d(1 * channels, channels, kernel_size=1),
            nn.GroupNorm(channels, channels),
            nn.PReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv3d(2 * channels, channels, kernel_size=1),
            nn.GroupNorm(channels, channels),
            nn.PReLU()
        )
        self.down3 = nn.Sequential(
            nn.Conv3d(3 * channels, channels, kernel_size=1),
            nn.GroupNorm(channels, channels),
            nn.PReLU()
        )
        self.down4 = nn.Sequential(
            nn.Conv3d(4 * channels, channels, kernel_size=1),
            nn.GroupNorm(channels, channels),
            nn.PReLU()
        )
        self.fms_conv = nn.Sequential(nn.Conv3d(4*channels, channels, kernel_size=1),
                                    nn.GroupNorm(channels, channels),
                                    nn.PReLU(),)

        self.ga1 = GuidedAttentionModule(channels,channels)
        self.ga2 = GuidedAttentionModule(channels,channels)
        self.ga3 = GuidedAttentionModule(channels,channels)
        self.ga4 = GuidedAttentionModule(channels,channels)

        self.recons_mfunit1 = MFunit(channels, mid_channel, g=groups, stride=1, norm=norm)
        self.recons_mfunit2 = MFunit(2 * channels, channels, g=groups, stride=1, norm=norm)

        self.decoder_mfunit1 = MFunit(2*channels, channels , g=groups, stride=1, norm=norm)
        self.decoder_mfunit2 = MFunit(2*channels, channels, g=groups, stride=1, norm=norm)
        self.decoder_mfunit3 = MFunit(2*channels, channels, g=groups, stride=1, norm=norm)
        self.decoder_mfunit4 = MFunit(2*channels, channels, g=groups, stride=1, norm=norm)
        self.out_conv = nn.Sequential(nn.Conv3d(4*channels, 1, kernel_size=1, padding=0, stride=1, bias=False))

        self.extra_feature1 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)))
        self.extra_feature2 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)))
        self.extra_feature3 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)))
        self.extra_feature4 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)))
        self.extra_feature5 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)))

        self.fc1 = nn.Sequential(nn.Linear(256, 128), nn.Dropout(0.5))
        self.fc2 = nn.Sequential(nn.Linear(128, 64), nn.Dropout(0.5))
        self.fc3 = nn.Sequential(nn.Linear(64, 2), nn.LogSoftmax())

    def forward(self, x):
        print(111111111111111111111111111)
        print(x.shape)
        inconv = self.in_conv(x)
        print(inconv.shape)
        encoder1 = self.encoder_block1(inconv)
        recons_up1 = F.upsample(encoder1, size=inconv.size()[2:], mode='trilinear')
        recons1 = self.recons_mfunit1(recons_up1)
        encoder2 = self.encoder_block2(encoder1)
        recons_up2 = F.upsample(encoder2, size=encoder1.size()[2:], mode='trilinear')
        recons2 = self.recons_mfunit2(recons_up2)
        encoder3 = self.encoder_block3(encoder2)
        encoder4 = self.encoder_block4(encoder3)
        down4_1 = self.down4(encoder4)
        feature1 = self.extra_feature1(down4_1)
        down3_1 = torch.add(
            F.upsample(down4_1, size=encoder3.size()[2:], mode='trilinear'),
            self.down3(encoder3)
        )
        feature2 = self.extra_feature2(down3_1)

        down2_1 = torch.add(
            F.upsample(down3_1, size=encoder2.size()[2:], mode='trilinear'),
            self.down2(encoder2)
        )
        feature3 = self.extra_feature3(down2_1)
        down1_1 = torch.add(
            F.upsample(down2_1, size=encoder1.size()[2:], mode='trilinear'),
            self.down1(encoder1)
        )
        feature4 = self.extra_feature4(down1_1)
        down1_2 =F.upsample(down1_1, size=encoder1.size()[2:], mode='trilinear')
        down2_2 =F.upsample(down2_1, size=encoder1.size()[2:], mode='trilinear')
        down3_2 =F.upsample(down3_1, size=encoder1.size()[2:], mode='trilinear')
        down4_2 =F.upsample(down4_1, size=encoder1.size()[2:], mode='trilinear')
        FMS = torch.cat((down1_2, down2_2, down3_2, down4_2), 1)
        FMS = self.fms_conv(FMS)
        """ga1 = self.ga3(down1_2,FMS)
        ga2 = self.ga4(down2_2,FMS)
        ga3 = self.ga3(down3_2,FMS)
        ga4 = self.ga4(down4_2,FMS)"""
        ga1 = torch.cat((down1_2*FMS,down1_2),1)
        ga2 = torch.cat((down2_2*FMS,down2_2),1)
        ga3 = torch.cat((down3_2*FMS,down3_2),1)
        ga4 = torch.cat((down4_2*FMS,down4_2),1)
        decoder1 = self.decoder_mfunit1(ga1)
        decoder2 = self.decoder_mfunit1(ga2)
        decoder3 = self.decoder_mfunit1(ga3)
        decoder4 = self.decoder_mfunit1(ga4)
        decoder = torch.cat((decoder1,decoder2,decoder3,decoder4),1)
        features = torch.cat((feature1, feature2, feature3, feature4), 1)
        features = features.view(-1)

        print(features.shape)
        cls_out = self.fc1(features)
        cls_out = self.fc2(cls_out)
        cls_out = self.fc3(cls_out)
        decoder = F.upsample(decoder, size=x.size()[2:], mode='trilinear')
        seg_out = self.out_conv(decoder)

        return inconv, recons1, encoder1, recons2, seg_out, seg_out, cls_out

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

class Deep_Attention_block(nn.Module):
    """
    Attention Block
    """
    def __init__(self, F_g, F_x, F_int):
        super(Deep_Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(16, F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_x, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(16, F_int),
        )
        self.fuse = nn.Sequential(
            nn.Conv3d(320, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, F_int, kernel_size=3, padding=1), nn.GroupNorm(32, F_int), nn.PReLU()
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(1, 1),
            nn.Sigmoid()
        )
        self.out_att = nn.Sequential(
            nn.Conv3d(2*F_int, F_int, kernel_size=1), nn.GroupNorm(32, F_int), nn.PReLU(),
            nn.Conv3d(F_int, F_int, kernel_size=3, padding=1), nn.GroupNorm(32, F_int), nn.PReLU(),
            nn.Conv3d(F_int, F_int, kernel_size=3, padding=1), nn.GroupNorm(32, F_int), nn.PReLU())

        self.PReLU = nn.PReLU()

    def forward(self, e1,e2,e3,e4,g,x):

        e1_resample = F.upsample(e1,size=x.size()[2:], mode='trilinear')
        e2_resample = F.upsample(e2,size=x.size()[2:], mode='trilinear')
        e3_resample = F.upsample(e3, size=x.size()[2:], mode='trilinear')
        e4_resample = F.upsample(e4, size=x.size()[2:], mode='trilinear')
        fms_concat = torch.cat((e1_resample,e2_resample,e3_resample,e4_resample),1)
        fms_fuse = self.fuse(fms_concat)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.PReLU(g1 + x1)
        psi = self.psi(psi)
        local_att = x * psi
        total_att = torch.cat((fms_fuse,local_att),1)
        out_att = self.out_att(total_att)
        return out_att


class Deep_Attentive_DMFNet(nn.Module):
    def __init__(self,in_channel=1,mid_channel=32,channels=32,groups=16,norm='gn',out_classes=2):
        super(Deep_Attentive_DMFNet, self).__init__()
        self.in_conv = nn.Conv3d(in_channel,mid_channel,kernel_size=3,padding=1,stride=2,bias=False)
        self.encoder_block1 = nn.Sequential(
            DMFUnit(mid_channel, channels, g=groups, stride=2, norm=norm, dilation=[1, 2, 3]),
            DMFUnit(channels, channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),
            DMFUnit(channels, channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),
        )
        self.encoder_block2 = nn.Sequential(
            DMFUnit(channels, 2 * channels, g=groups, stride=2, norm=norm, dilation=[1, 2, 3]),
            DMFUnit(2 * channels, 2 * channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),
            DMFUnit(2 * channels, 2 * channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),
        )
        self.encoder_block3 = nn.Sequential(
            DMFUnit(2 * channels, 3 * channels, g=groups, stride=2, norm=norm, dilation=[1, 2, 3]),
            DMFUnit(3 * channels, 3 * channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),
            DMFUnit(3 * channels, 3 * channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),
        )
        self.encoder_block4 = nn.Sequential(
            DMFUnit(3 * channels, 4 * channels, g=groups, stride=2, norm=norm, dilation=[1, 2, 3]),
            DMFUnit(4 * channels, 4 * channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),
            DMFUnit(4 * channels, 4 * channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),
        )
        self.att_block1 = Deep_Attention_block(4*channels,3*channels,3*channels)
        self.att_block2 = Deep_Attention_block(2*channels,2*channels,2*channels)
        self.att_block3 = Deep_Attention_block(1*channels,1*channels,1*channels)

        self.recons_mfunit1 = MFunit(channels, mid_channel, g=groups, stride=1, norm=norm)
        self.recons_mfunit2 = MFunit(2 * channels, channels, g=groups, stride=1, norm=norm)

        self.decoder_block1 = MFunit(channels * 7, channels * 2, g=groups, stride=1, norm=norm)
        self.decoder_block2 = MFunit(channels * 4, channels , g=groups, stride=1, norm=norm)
        self.decoder_block3 = MFunit(channels * 2 ,channels, g=groups, stride=1, norm=norm)
        self.out_conv = nn.Sequential(nn.Conv3d(channels, 1, kernel_size=1, padding=0, stride=1, bias=False))

        self.extra_feature1 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)))
        self.extra_feature2 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)))
        self.extra_feature3 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)))
        self.extra_feature4 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)))

        self.fc1 = nn.Sequential(nn.Linear(320, 128), nn.Dropout(0.5))
        self.fc2 = nn.Sequential(nn.Linear(128, 2), nn.LogSoftmax())

    def forward(self, x):
        print("-------------------------------------------")
        print("input shape:")
        print(x.shape)
        inconv = self.in_conv(x)
        print("inconv shape:")
        print(inconv.shape) #mid channel
        #encoder part
        encoder1 = self.encoder_block1(inconv)   #1 channels
        encoder2 = self.encoder_block2(encoder1) #2 channels
        encoder3 = self.encoder_block3(encoder2) #3 channels
        encoder4 = self.encoder_block4(encoder3) #4 channels

        feature1 = self.extra_feature1(encoder1)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(feature1.shape)
        feature2 = self.extra_feature2(encoder2)
        feature3 = self.extra_feature3(encoder3)
        feature4 = self.extra_feature4(encoder4)

        #attentive
        up_decoder1 = F.upsample(encoder4,size=encoder3.size()[2:], mode='trilinear')
        att1 = self.att_block1(encoder1,encoder2,encoder3,encoder4,up_decoder1,encoder3)
        decoder1 = self.decoder_block1(torch.cat((att1,up_decoder1),1))
        up_decoder2 = F.upsample(decoder1,size=encoder2.size()[2:], mode='trilinear')
        att2 = self.att_block2(encoder1,encoder2,encoder3,encoder4,up_decoder2,encoder2)
        decoder2 = self.decoder_block2(torch.cat((att2,up_decoder2),1))
        up_decoder3 = F.upsample(decoder2, size=encoder1.size()[2:], mode='trilinear')
        att3 = self.att_block3(encoder1, encoder2, encoder3, encoder4, up_decoder3, encoder1)
        decoder3 = self.decoder_block3(torch.cat((att3, up_decoder3), 1))
        up_decoder4 = F.upsample(decoder3, size=inconv.size()[2:], mode='trilinear')

        #recons
        recons_up1 = F.upsample(encoder1, size=inconv.size()[2:], mode='trilinear')
        recons1 = self.recons_mfunit1(recons_up1)
        recons_up2 = F.upsample(encoder2, size=encoder1.size()[2:], mode='trilinear')
        recons2 = self.recons_mfunit2(recons_up2)

        #classification
        features = torch.cat((feature1, feature2, feature3, feature4), 1)
        print("::::::::::::::::::::::::::::::::::::::")
        print(features.shape)
        features = features.view(-1)
        print(features.shape)
        cls_out = self.fc1(features)
        cls_out = self.fc2(cls_out)
        #segmentation
        seg_out = self.out_conv(up_decoder4)
        seg_out = F.upsample(seg_out,size=x.size()[2:], mode='trilinear')
        return inconv, recons1, encoder1, recons2, seg_out, seg_out, cls_out

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda:0')
    x = torch.rand((1,4,128,128,128),device=device) # [bsize,channels,Height,Width,Depth]
    model = DMFNet(c=4, groups=16, norm='sync_bn', num_classes=4)
    model.cuda(device)
    y = model(x)
    print(y.shape)
