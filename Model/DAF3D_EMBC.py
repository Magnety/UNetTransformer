from torch import nn
import torch
import torch.nn.functional as F
from .BackBone3D import BackBone3D


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


class DAF3D(nn.Module):
    def __init__(self):
        super(DAF3D, self).__init__()
        self.backbone = BackBone3D()

        self.down4 = nn.Sequential(
            nn.Conv3d(1024, 128, kernel_size=1),
            nn.GroupNorm(32, 128),
            nn.PReLU(),
        )
        self.down3 = nn.Sequential(
            nn.Conv3d(512, 128, kernel_size=1),
            nn.GroupNorm(32, 128),
            nn.PReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=1),
            nn.GroupNorm(32, 128),
            nn.PReLU()
        )
        self.down1 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=1),
            nn.GroupNorm(32, 128),
            nn.PReLU()
        )

        self.fuse1 = nn.Sequential(
            nn.Conv3d(512, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU()
        )
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
        self.refine2 = nn.Sequential(
            nn.Conv3d(192, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU()
        )
        self.refine1 = nn.Sequential(
            nn.Conv3d(192, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU()
        )
        self.refine = nn.Sequential(nn.Conv3d(256, 64, kernel_size=1),
                                    nn.GroupNorm(32, 64),
                                    nn.PReLU(),)

        rates = (1, 6, 12, 18)
        self.aspp1 = ASPP_module(64, 64, rate=rates[0])
        self.aspp2 = ASPP_module(64, 64, rate=rates[1])
        self.aspp3 = ASPP_module(64, 64, rate=rates[2])
        self.aspp4 = ASPP_module(64, 64, rate=rates[3])

        self.aspp_conv = nn.Conv3d(256, 64, 1)
        self.aspp_gn = nn.GroupNorm(32, 64)
        self.extra_feature1 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)))
        self.extra_feature2 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)))
        self.extra_feature3 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)))
        self.extra_feature4 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)))
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Sequential(nn.Linear(256, 2), nn.LogSoftmax())
        self.predict1_4 = nn.Conv3d(128, 1, kernel_size=1)
        self.predict1_3 = nn.Conv3d(128, 1, kernel_size=1)
        self.predict1_2 = nn.Conv3d(128, 1, kernel_size=1)
        self.predict1_1 = nn.Conv3d(128, 1, kernel_size=1)

        self.predict2_4 = nn.Conv3d(64, 1, kernel_size=1)
        self.predict2_3 = nn.Conv3d(64, 1, kernel_size=1)
        self.predict2_2 = nn.Conv3d(64, 1, kernel_size=1)
        self.predict2_1 = nn.Conv3d(64, 1, kernel_size=1)

        self.predict = nn.Conv3d(64, 1, kernel_size=1)

    def forward(self, x):
        print("input shape")
        print(x.shape)
        layer0 = self.backbone.layer0(x)
        print("layer0 shape")
        print(layer0.shape)
        layer1 = self.backbone.layer1(layer0)
        print("layer1 shape")
        print(layer1.shape)
        layer2 = self.backbone.layer2(layer1)
        print("layer2 shape")
        print(layer2.shape)
        layer3 = self.backbone.layer3(layer2)
        print("layer3 shape")
        print(layer3.shape)

        layer4 = self.backbone.layer4(layer3)
        print("layer4 shape")
        print(layer4.shape)

        # Top-down
        down4 = self.down4(layer4)

        print("down4 shape")
        print(down4.shape)
        print("layer3.size")
        print(layer3.size()[2:])
        down3 = torch.add(
            F.upsample(down4, size=layer3.size()[2:], mode='trilinear'),
            self.down3(layer3)
        )


        print("down3 shape")
        print(down3.shape)
        down2 = torch.add(
            F.upsample(down3, size=layer2.size()[2:], mode='trilinear'),
            self.down2(layer2)
        )
        print("down2 shape")
        print(down2.shape)
        down1 = torch.add(
            F.upsample(down2, size=layer1.size()[2:], mode='trilinear'),
            self.down1(layer1)
        )
        print("down1 shape")
        print(down1.shape)
        down4 = F.upsample(down4, size=layer1.size()[2:], mode='trilinear')
        print("down4 shape")
        print(down4.shape)

        down3 = F.upsample(down3, size=layer1.size()[2:], mode='trilinear')
        print("down3 shape")
        print(down3.shape)
        down2 = F.upsample(down2, size=layer1.size()[2:], mode='trilinear')
        print("down2 shape")
        print(down2.shape)

        predict1_4 = self.predict1_4(down4)
        print("predict1_4 shape")
        print(predict1_4.shape)
        predict1_3 = self.predict1_3(down3)
        print("predict1_3 shape")
        print(predict1_3.shape)
        predict1_2 = self.predict1_2(down2)
        print("predict1_2 shape")
        print(predict1_2.shape)
        predict1_1 = self.predict1_1(down1)
        print("predict1_1 shape")
        print(predict1_1.shape)

        fuse1 = self.fuse1(torch.cat((down4, down3, down2, down1), 1))
        print("fuse1 shape")
        print(fuse1.shape)
        attention4 = self.attention4(torch.cat((down4, fuse1), 1))
        print("attention4 shape")
        print(attention4.shape)
        attention3 = self.attention3(torch.cat((down3, fuse1), 1))
        print("attention3 shape")
        print(attention3.shape)
        attention2 = self.attention2(torch.cat((down2, fuse1), 1))
        print("attention2 shape")
        print(attention2.shape)
        attention1 = self.attention1(torch.cat((down1, fuse1), 1))
        print("attention1 shape")
        print(attention1.shape)

        refine4 = self.refine4(torch.cat((down4, attention4 * fuse1), 1))
        print("refine4 shape")
        print(refine4.shape)
        refine3 = self.refine3(torch.cat((down3, attention3 * fuse1), 1))
        print("refine3 shape")
        print(refine3.shape)
        refine2 = self.refine2(torch.cat((down2, attention2 * fuse1), 1))
        print("refine2 shape")
        print(refine2.shape)
        refine1 = self.refine1(torch.cat((down1, attention1 * fuse1), 1))
        print("refine1 shape")
        print(refine1.shape)

        refine = self.refine(torch.cat((refine1, refine2, refine3, refine4), 1))
        print("refine shape")
        print(refine.shape)
        predict2_4 = self.predict2_4(refine4)
        print("predict2_4 shape")
        print(predict2_4.shape)
        predict2_3 = self.predict2_3(refine3)
        print("predict2_3 shape")
        print(predict2_3.shape)
        predict2_2 = self.predict2_2(refine2)
        print("predict2_2 shape")
        print(predict2_2.shape)
        predict2_1 = self.predict2_1(refine1)
        print("predict2_1 shape")
        print(predict2_1.shape)

        aspp1 = self.aspp1(refine)
        print("aspp1 shape")
        print(aspp1.shape)
        aspp2 = self.aspp2(refine)
        print("aspp2 shape")
        print(aspp2.shape)
        aspp3 = self.aspp3(refine)
        print("aspp3 shape")
        print(aspp3.shape)
        aspp4 = self.aspp4(refine)
        print("aspp4 shape")
        print(aspp4.shape)

        aspp = torch.cat((aspp1, aspp2, aspp3, aspp4), dim=1)
        print("aspp shape")
        print(aspp.shape)
        aspp = self.aspp_gn(self.aspp_conv(aspp))
        print("aspp shape")
        print(aspp.shape)
        predict = self.predict(aspp)
        print("predict shape")
        print(predict.shape)
        predict1_1 = F.upsample(predict1_1, size=x.size()[2:], mode='trilinear')
        print("predict1_1 shape")
        print(predict1_1.shape)
        predict1_2 = F.upsample(predict1_2, size=x.size()[2:], mode='trilinear')
        print("predict1_2 shape")
        print(predict1_2.shape)
        predict1_3 = F.upsample(predict1_3, size=x.size()[2:], mode='trilinear')
        print("predict1_3 shape")
        print(predict1_3.shape)
        predict1_4 = F.upsample(predict1_4, size=x.size()[2:], mode='trilinear')
        print("predict1_4 shape")
        print(predict1_4.shape)
        predict2_1 = F.upsample(predict2_1, size=x.size()[2:], mode='trilinear')
        print("predict2_1 shape")
        print(predict2_1.shape)
        predict2_2 = F.upsample(predict2_2, size=x.size()[2:], mode='trilinear')
        print("predict2_2 shape")
        print(predict2_2.shape)
        predict2_3 = F.upsample(predict2_3, size=x.size()[2:], mode='trilinear')
        print("predict2_3 shape")
        print(predict2_3.shape)
        predict2_4 = F.upsample(predict2_4, size=x.size()[2:], mode='trilinear')
        print("predict2_4 shape")
        print(predict2_4.shape)

        predict = F.upsample(predict, size=x.size()[2:], mode='trilinear')
        print("predict shape")
        print(predict.shape)

        return predict

