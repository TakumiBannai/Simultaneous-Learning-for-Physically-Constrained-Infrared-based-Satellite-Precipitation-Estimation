#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn

class ConvBNRelu(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                              kernel_size=3, stride=1, padding=1, dilation=1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class PoolingBNRelu(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.in_channel = in_channel
        self.maxpooling = nn.MaxPool2d(2, 2, 0)
        self.bn = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.maxpooling(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConvReluConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel*2,
                              kernel_size=3, stride=1, padding=1, dilation=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=in_channel*2, out_channels=out_channel,
                              kernel_size=3, stride=1, padding=1, dilation=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x  


class DeconvConvConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.deconv = nn.ConvTranspose2d(in_channels=in_channel,
                                         out_channels=in_channel,
                                         kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel,
                              kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                              kernel_size=3, stride=1, padding=1, dilation=1)

    def forward(self, x):
        x = self.deconv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class PersiannCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # IR channel encoder
        self.conv_IR = nn.Sequential(ConvBNRelu(2, 16),
                                     PoolingBNRelu(16),
                                     ConvBNRelu(16, 32),
                                     PoolingBNRelu(32)
                                     )
                      
        # WV channel encoder
        self.conv_WV = nn.Sequential(ConvBNRelu(2, 16),
                                     PoolingBNRelu(16),
                                     ConvBNRelu(16, 32),
                                     PoolingBNRelu(32)
                                     )
        

        # Decoder network
        self.decoder = nn.Sequential(DeconvConvConv(64, 64),
                                     DeconvConvConv(64, 128),
                                     nn.Conv2d(128, 128, 3, 1, 1),
                                     nn.ReLU(),
                                     nn.Conv2d(128, 1, 3, 1, 1),
                                     nn.ReLU()
                                    )

    def forward(self, x_ir, x_wv):
        x_ir = self.conv_IR(x_ir)
        x_wv = self.conv_WV(x_wv)
        x = torch.cat((x_ir, x_wv), dim=1)
        x = self.decoder(x)
        return x



class PersiannCW(nn.Module):
    def __init__(self):
        super().__init__()
        # IR channel encoder
        self.conv_IR = nn.Sequential(ConvBNRelu(2, 16),
                                     PoolingBNRelu(16),
                                     ConvBNRelu(16, 32),
                                     PoolingBNRelu(32)
                                     )
                      
        # WV channel encoder
        self.conv_WV = nn.Sequential(ConvBNRelu(2, 16),
                                     PoolingBNRelu(16),
                                     ConvBNRelu(16, 32),
                                     PoolingBNRelu(32)
                                     )

       # CW or CI channel encoder
        self.conv_Cloud = nn.Sequential(ConvBNRelu(2, 16),
                                     PoolingBNRelu(16),
                                     ConvBNRelu(16, 32),
                                     PoolingBNRelu(32)
                                     )     

        # Decoder network
        self.decoder = nn.Sequential(DeconvConvConv(96, 96),
                                     DeconvConvConv(96, 128),
                                     nn.Conv2d(128, 128, 3, 1, 1),
                                     nn.ReLU(),
                                     nn.Conv2d(128, 1, 3, 1, 1),
                                     nn.ReLU()
                                    )

    def forward(self, x_ir, x_wv, x_cloud):
        x_ir = self.conv_IR(x_ir)
        x_wv = self.conv_WV(x_wv)
        x_cloud = self.conv_Cloud(x_cloud)
        x = torch.cat((x_ir, x_wv, x_cloud), dim=1)
        x = self.decoder(x)
        return x


class PersiannCWCI(nn.Module):
    def __init__(self):
        super().__init__()
        # IR channel encoder
        self.conv_IR = nn.Sequential(ConvBNRelu(2, 16),
                                     PoolingBNRelu(16),
                                     ConvBNRelu(16, 32),
                                     PoolingBNRelu(32)
                                     )
                      
        # WV channel encoder
        self.conv_WV = nn.Sequential(ConvBNRelu(2, 16),
                                     PoolingBNRelu(16),
                                     ConvBNRelu(16, 32),
                                     PoolingBNRelu(32)
                                     )

       # CW channel encoder
        self.conv_CW = nn.Sequential(ConvBNRelu(2, 16),
                                     PoolingBNRelu(16),
                                     ConvBNRelu(16, 32),
                                     PoolingBNRelu(32)
                                     ) 
        
        # CI channel encoder
        self.conv_CI = nn.Sequential(ConvBNRelu(2, 16),
                                     PoolingBNRelu(16),
                                     ConvBNRelu(16, 32),
                                     PoolingBNRelu(32)
                                     )

        # Decoder network
        self.decoder = nn.Sequential(DeconvConvConv(128, 128),
                                     DeconvConvConv(128, 128),
                                     nn.Conv2d(128, 128, 3, 1, 1),
                                     nn.ReLU(),
                                     nn.Conv2d(128, 1, 3, 1, 1),
                                     nn.ReLU()
                                    )

    def forward(self, x_ir, x_wv, x_cw, x_ci):
        x_ir = self.conv_IR(x_ir)
        x_wv = self.conv_WV(x_wv)
        x_cw = self.conv_CW(x_cw)
        x_ci = self.conv_CI(x_ci)
        x = torch.cat((x_ir, x_wv, x_cw, x_ci), dim=1)
        x = self.decoder(x)
        return x


class EncoderSkipConnect(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv = ConvBNRelu(in_channel, out_channel)
        self.pooling = PoolingBNRelu(out_channel)
    def forward(self, x):
        x = self.conv(x)
        x_out = self.pooling(x)
        return x_out, x


class DecoderSkipConnect(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.deconv = nn.ConvTranspose2d(in_channels=in_channel,
                                         out_channels=in_channel,
                                         kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(in_channels=in_channel*2, out_channels=in_channel,
                              kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                              kernel_size=3, stride=1, padding=1, dilation=1)
        
    def forward(self, x, x_skip):
        x = self.deconv(x)
        x = torch.cat((x, x_skip), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        # IR+WV channel encoder
        self.encoder1 = EncoderSkipConnect(4, 16)
        self.encoder2 = EncoderSkipConnect(16, 32)
        self.encoder3 = EncoderSkipConnect(32, 64)
        # Decoder network
        self.decoder1 = DecoderSkipConnect(64, 32)
        self.decoder2 = DecoderSkipConnect(32, 16)
        self.decoder3 = DecoderSkipConnect(16, 1)

    def forward(self, x_ir, x_wv):
        x = torch.cat((x_ir, x_wv), dim=1 )
        x, x_skip_1 = self.encoder1(x)
        x, x_skip_2 = self.encoder2(x)
        x, x_skip_3 = self.encoder3(x)
        x = self.decoder1(x, x_skip_3)
        x = self.decoder2(x, x_skip_2)
        x = self.decoder3(x, x_skip_1)
        return x


class UnetCW(nn.Module):
    def __init__(self):
        super().__init__()
        # IR+WV channel encoder
        self.encoder1 = EncoderSkipConnect(6, 16)
        self.encoder2 = EncoderSkipConnect(16, 32)
        self.encoder3 = EncoderSkipConnect(32, 64)
        # Decoder network
        self.decoder1 = DecoderSkipConnect(64, 32)
        self.decoder2 = DecoderSkipConnect(32, 16)
        self.decoder3 = DecoderSkipConnect(16, 1)

    def forward(self, x_ir, x_wv, x_cloud):
        x = torch.cat((x_ir, x_wv, x_cloud), dim=1 )
        x, x_skip_1 = self.encoder1(x)
        x, x_skip_2 = self.encoder2(x)
        x, x_skip_3 = self.encoder3(x)
        x = self.decoder1(x, x_skip_3)
        x = self.decoder2(x, x_skip_2)
        x = self.decoder3(x, x_skip_1)
        return x


class UnetCWCI(nn.Module):
    def __init__(self):
        super().__init__()
        # IR+WV channel encoder
        self.encoder1 = EncoderSkipConnect(8, 16)
        self.encoder2 = EncoderSkipConnect(16, 32)
        self.encoder3 = EncoderSkipConnect(32, 64)
        # Decoder network
        self.decoder1 = DecoderSkipConnect(64, 32)
        self.decoder2 = DecoderSkipConnect(32, 16)
        self.decoder3 = DecoderSkipConnect(16, 1)

    def forward(self, x_ir, x_wv, x_cw, x_ci):
        x = torch.cat((x_ir, x_wv, x_cw, x_ci), dim=1 )
        x, x_skip_1 = self.encoder1(x)
        x, x_skip_2 = self.encoder2(x)
        x, x_skip_3 = self.encoder3(x)
        x = self.decoder1(x, x_skip_3)
        x = self.decoder2(x, x_skip_2)
        x = self.decoder3(x, x_skip_1)
        return x



class PersiannMTL_RM(nn.Module):
    def __init__(self):
        super().__init__()
        # IR channel encoder
        self.conv_IR = nn.Sequential(ConvBNRelu(2, 16),
                                     PoolingBNRelu(16),
                                     ConvBNRelu(16, 32),
                                     PoolingBNRelu(32)
                                     )
                      
        # WV channel encoder
        self.conv_WV = nn.Sequential(ConvBNRelu(2, 16),
                                     PoolingBNRelu(16),
                                     ConvBNRelu(16, 32),
                                     PoolingBNRelu(32)
                                     )
        
        # Cloud water channel encoder
        self.conv_CW = nn.Sequential(ConvBNRelu(2, 16),
                                     PoolingBNRelu(16),
                                     ConvBNRelu(16, 32),
                                     PoolingBNRelu(32)
                                    )
        # Cloud ice channel encoder
        self.conv_CI = nn.Sequential(ConvBNRelu(2, 16),
                                     PoolingBNRelu(16),
                                     ConvBNRelu(16, 32),
                                     PoolingBNRelu(32)
                                    )
        

        # Decoder network
        self.decoder = nn.Sequential(DeconvConvConv(64, 64),
                                     DeconvConvConv(64, 128),
                                     nn.Conv2d(128, 128, 3, 1, 1),
                                     nn.ReLU()
                                    )
        
        self.decoder_CW = nn.Sequential(DeconvConvConv(96, 96),
                                     DeconvConvConv(96, 128),
                                     nn.Conv2d(128, 128, 3, 1, 1),
                                     nn.ReLU()
                                    )
        
        self.decoder_CWCI = nn.Sequential(DeconvConvConv(128, 128),
                                     DeconvConvConv(128, 128),
                                     nn.Conv2d(128, 128, 3, 1, 1),
                                     nn.ReLU()
                                    )
        # Output layer
        self.out_rainrate = nn.Sequential(ConvReluConv(128, 1),
                                          nn.ReLU()
                                         )

        self.out_rainmask = nn.Sequential(ConvReluConv(128, 1),
                                          nn.Sigmoid()
                                         )
        self.out_cloudwater = nn.Sequential(ConvReluConv(128, 1),
                                            nn.ReLU()
                                           )
        
        self.out_cloudice = nn.Sequential(ConvReluConv(128, 1),
                                          nn.ReLU()
                                         )
                                    

    def forward(self, x_ir, x_wv):
        x_ir = self.conv_IR(x_ir)
        x_wv = self.conv_WV(x_wv)
        x = torch.cat((x_ir, x_wv), dim=1)
        x = self.decoder(x)
        x_rainrate = self.out_rainrate(x)
        x_rainmask = self.out_rainmask(x)
        return x_rainrate, x_rainmask


# Cloud Water or Cloud Ice input
class PersiannMTL_CW(PersiannMTL_RM):
    def __init__(self):
        super().__init__()
    def forward(self, x_ir, x_wv):
        x_ir = self.conv_IR(x_ir)
        x_wv = self.conv_WV(x_wv)
        x = torch.cat((x_ir, x_wv), dim=1)
        x = self.decoder(x)
        x_rainrate = self.out_rainrate(x)
        x_rainmask = self.out_rainmask(x)
        x_cloudwater = self.out_cloudwater(x)
        return x_rainrate, x_rainmask, x_cloudwater


class PersiannMTLMultiInput_CW(PersiannMTL_RM):
    def __init__(self):
        super().__init__()
    def forward(self, x_ir, x_wv, x_cloud):
        x_ir = self.conv_IR(x_ir)
        x_wv = self.conv_WV(x_wv)
        x_cloud = self.conv_CW(x_cloud)
        x = torch.cat((x_ir, x_wv, x_cloud), dim=1)
        x = self.decoder_CW(x)
        x_rainrate = self.out_rainrate(x)
        x_rainmask = self.out_rainmask(x)
        x_cloudwater = self.out_cloudwater(x)
        return x_rainrate, x_rainmask, x_cloudwater       


# Both input
class PersiannMTL_CWCI(PersiannMTL_RM):
    def __init__(self):
        super().__init__()

    def forward(self, x_ir, x_wv):
        x_ir = self.conv_IR(x_ir)
        x_wv = self.conv_WV(x_wv)        
        x = torch.cat((x_ir, x_wv), dim=1)
        x = self.decoder(x)
        x_rainrate = self.out_rainrate(x)
        x_rainmask = self.out_rainmask(x)
        x_cloudwater = self.out_cloudwater(x)
        x_cloudice = self.out_cloudice(x)
        return x_rainrate, x_rainmask, x_cloudwater, x_cloudice

class PersiannMTLMultiInput_CWCI(PersiannMTL_RM):
    def __init__(self):
        super().__init__()

    def forward(self, x_ir, x_wv, x_cw, x_ci):
        x_ir = self.conv_IR(x_ir)
        x_wv = self.conv_WV(x_wv)   
        x_cw = self.conv_CW(x_cw)
        x_ci = self.conv_CI(x_ci)     
        x = torch.cat((x_ir, x_wv, x_cw, x_ci), dim=1)
        x = self.decoder_CWCI(x)
        x_rainrate = self.out_rainrate(x)
        x_rainmask = self.out_rainmask(x)
        x_cloudwater = self.out_cloudwater(x)
        x_cloudice = self.out_cloudice(x)
        return x_rainrate, x_rainmask, x_cloudwater, x_cloudice


class UnetMTL_RM(nn.Module):
    def __init__(self):
        super().__init__()
        # IR+WV channel encoder
        self.encoder1 = EncoderSkipConnect(4, 16)
        self.encoder2 = EncoderSkipConnect(16, 32)
        self.encoder3 = EncoderSkipConnect(32, 64)
        # Decoder network
        self.decoder1 = DecoderSkipConnect(64, 32)
        self.decoder2 = DecoderSkipConnect(32, 16)
        self.decoder3 = DecoderSkipConnect(16, 8)
        # Output layer
        self.out_rainrate = nn.Sequential(ConvReluConv(8, 1),
                                      nn.ReLU()
                                     )

        self.out_rainmask = nn.Sequential(ConvReluConv(8, 1),
                                          nn.Sigmoid()
                                         )
        self.out_cloudwater = nn.Sequential(ConvReluConv(8, 1),
                                            nn.ReLU()
                                           )
        
        self.out_cloudice = nn.Sequential(ConvReluConv(8, 1),
                                          nn.ReLU()
                                         )

    def forward(self, x_ir, x_wv):
        x = torch.cat((x_ir, x_wv), dim=1 )
        x, x_skip_1 = self.encoder1(x)
        x, x_skip_2 = self.encoder2(x)
        x, x_skip_3 = self.encoder3(x)
        x = self.decoder1(x, x_skip_3)
        x = self.decoder2(x, x_skip_2)
        x = self.decoder3(x, x_skip_1)
        x_rainrate = self.out_rainrate(x)
        x_rainmask = self.out_rainmask(x)
        return x_rainrate, x_rainmask


class UnetMTL_CW(UnetMTL_RM):
    def __init__(self):
        super().__init__()
    def forward(self, x_ir, x_wv):
        x = torch.cat((x_ir, x_wv), dim=1)
        x, x_skip_1 = self.encoder1(x)
        x, x_skip_2 = self.encoder2(x)
        x, x_skip_3 = self.encoder3(x)
        x = self.decoder1(x, x_skip_3)
        x = self.decoder2(x, x_skip_2)
        x = self.decoder3(x, x_skip_1)
        x_rainrate = self.out_rainrate(x)
        x_rainmask = self.out_rainmask(x)
        x_cloudwater = self.out_cloudwater(x)
        return x_rainrate, x_rainmask, x_cloudwater


class UnetMTL_CWCI(UnetMTL_RM):
    def __init__(self):
        super().__init__()
    def forward(self, x_ir, x_wv):
        x = torch.cat((x_ir, x_wv), dim=1)
        x, x_skip_1 = self.encoder1(x)
        x, x_skip_2 = self.encoder2(x)
        x, x_skip_3 = self.encoder3(x)
        x = self.decoder1(x, x_skip_3)
        x = self.decoder2(x, x_skip_2)
        x = self.decoder3(x, x_skip_1)
        x_rainrate = self.out_rainrate(x)
        x_rainmask = self.out_rainmask(x)
        x_cloudwater = self.out_cloudwater(x)
        x_cloudice = self.out_cloudice(x)
        return x_rainrate, x_rainmask, x_cloudwater, x_cloudice
