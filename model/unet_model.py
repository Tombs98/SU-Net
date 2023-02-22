""" Full assembly of the parts to form the complete network """
import torch
from sklearn.tree import DecisionTreeRegressor
from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.id = DoubleConv(n_channels, 1)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.lr = Regression(360*360, 1)
        # self.svr = Svr(1,360*360,1)
        # self.lr2 = Regression(128*128, 64*64)
        # self.lr3 = Regression(64*64, 32*32)
        # self.lr4 = Regression(32*32, 1)
        # self.lr0 = Regression(360*360*3, 1)

    def forward(self, x):
        # y = x.view(2, 360*360*3)
        # # print("x", y.shape)
        # x_r = self.lr0(y)
        # print("x_r",x_r.shape)
        x0 = self.id(x)
        # print("x0",x0.shape)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        # print('~~~~~~~~~~~~~~~~~~~')
        # print("lo",logits.shape)
        logits = torch.squeeze(logits)
        x0 = torch.squeeze(x0)
        # print(x0.shape)
        x0 = x0.view(2, 360*360)
        # print(x0.shape)
        # print("lo", logits.shape)
        logits = logits.view(2, 360*360)
        #print("lo", logits.shape)
        # print('~~~~~~~~~~~~~~~~~~~')
        logits = logits + x0
        lr = self.lr(logits)
        # svr = self.svr(logits)
        # lr = self.lr2(lr)
        # lr = self.lr3(lr)
        # lr = self.lr4(lr)
        # print("lr",lr.shape)
        # res = (lr + x_r)/2
        res = torch.squeeze(lr)
        return res
