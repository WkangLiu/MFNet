import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from net.Res2Net import res2net50_v1b_26w_4s

import numpy as np
import math
import copy

from pdb import set_trace as stx
import numbers
from einops import rearrange

class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class EGM(nn.Module):
    def __init__(self):
        super(EGM, self).__init__()
        self.reduce1 = Conv1x1(512, 64)
        self.reduce4 = Conv1x1(2048, 256)
        self.block = nn.Sequential(
            ConvBNR(256 + 64, 256, 3),
            nn.Conv2d(256, 1, 1))

    def forward(self, x4, x1):
        size = x1.size()[2:]
        # print(x1.shape)
        x1 = self.reduce1(x1)
        x4 = self.reduce4(x4)
        x4 = F.interpolate(x4, size, mode='bilinear', align_corners=False)
        out = torch.cat((x4, x1), dim=1)
        out = self.block(out)

        return out
class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = ConvBNR(in_planes, in_planes, 3)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, att):
        if x.size() != att.size():
            att = F.interpolate(att, x.size()[2:], mode='bilinear', align_corners=False)
        x = x * att + x
        x = self.conv(x)
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
    

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias,mode):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv_0 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    
        self.qkv1conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv2conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim,bias=bias)
        self.qkv3conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim,bias=bias)
    
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    
    def forward(self, x,mask=None):
        b,c,h,w = x.shape
        # 先经过1*1卷积，然后再经过3*3卷积分别生成q，k，v
        q=self.qkv1conv(self.qkv_0(x))
        k=self.qkv2conv(self.qkv_1(x))
        v=self.qkv3conv(self.qkv_2(x))
        # 当mask不为none时，那么就将q，k乘以mask
        if mask is not None:
            q=q*mask
            k=k*mask

        # 调整元素顺序，，并且可以修改为多头注意力机制
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # q，k正则化
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # self.temperature:设置num_head*1*1个的全一向量
        # x y z w -> x y w z
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class MSA_head(nn.Module):
    def __init__(self,  dim, mode='dilation', num_heads=8, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias'):
        super(MSA_head, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias,mode)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x,mask=None):
        x = x + self.attn(self.norm1(x),mask)
        x = x + self.ffn(self.norm2(x))
        return x

class MSCA(nn.Module):
    def __init__(self, channels=128, r=4):
        super(MSCA, self).__init__()
        out_channels = int(channels // r)
        # local_att
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        # global_att
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        self.sig = nn.Sigmoid()

    def forward(self, x):

        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sig(xlg)

        return wei

class asyConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False):
        super(asyConv, self).__init__()
        self.deploy = deploy
        if deploy:
            self.fused_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size,kernel_size), stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
            self.initialize()
        else:
            self.square_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(kernel_size, kernel_size), stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=False,
                                         padding_mode=padding_mode)
            self.square_bn = nn.BatchNorm2d(num_features=out_channels)

            center_offset_from_origin_border = padding - kernel_size // 2
            ver_pad_or_crop = (center_offset_from_origin_border + 1, center_offset_from_origin_border)
            hor_pad_or_crop = (center_offset_from_origin_border, center_offset_from_origin_border + 1)
            if center_offset_from_origin_border >= 0:
                self.ver_conv_crop_layer = nn.Identity()
                ver_conv_padding = ver_pad_or_crop
                self.hor_conv_crop_layer = nn.Identity()
                hor_conv_padding = hor_pad_or_crop
            else:
                self.ver_conv_crop_layer = CropLayer(crop_set=ver_pad_or_crop)
                ver_conv_padding = (0, 0)
                self.hor_conv_crop_layer = CropLayer(crop_set=hor_pad_or_crop)
                hor_conv_padding = (0, 0)
            self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 1),
                                      stride=stride,
                                      padding=ver_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)

            self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3),
                                      stride=stride,
                                      padding=hor_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)
            self.ver_bn = nn.BatchNorm2d(num_features=out_channels)
            self.hor_bn = nn.BatchNorm2d(num_features=out_channels)


    def forward(self, input):
        if self.deploy:
            return self.fused_conv(input)
        else:
            square_outputs = self.square_conv(input)
            square_outputs = self.square_bn(square_outputs)
            vertical_outputs = self.ver_conv_crop_layer(input)
            vertical_outputs = self.ver_conv(vertical_outputs)
            vertical_outputs = self.ver_bn(vertical_outputs)
            horizontal_outputs = self.hor_conv_crop_layer(input)
            horizontal_outputs = self.hor_conv(horizontal_outputs)
            horizontal_outputs = self.hor_bn(horizontal_outputs)
            return square_outputs + vertical_outputs + horizontal_outputs

   
class SRPC(nn.Module):
    def __init__(self, hchannel, channel):
        super(SRPC, self).__init__()
        self.conv1_1 = Conv1x1(hchannel ,channel)
        self.conv3_1 = ConvBNR(channel // 4, channel // 4, 3)
        self.conv3_2 = nn.Sequential(
            asyConv(in_channels=channel // 4, out_channels=channel // 4, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False),
            nn.BatchNorm2d(channel // 4),
            nn.ReLU(inplace=True)
        )
        self.dconv5_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=2)
        self.dconv7_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=3)
        self.dconv9_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=4)
        self.conv1_2 = Conv1x1(channel, channel)
        self.conv3_3 = ConvBNR(channel, channel, 3)

    def forward(self, lf):
        
        
        x = self.conv1_1(lf)
        xc = torch.chunk(x, 4, dim=1)
        x0 = self.conv3_2(xc[0] + xc[1])
        x1 = self.dconv5_1(xc[1] + x0 + xc[2])
        x2 = self.dconv7_1(xc[2] + x1 + xc[3])
        x3 = self.dconv9_1(xc[3] + x2)
        xx = self.conv1_2(torch.cat((x0, x1, x2, x3), dim=1))
        x = self.conv3_3(x + xx)

        return x
# for conv5
class MFIM_5(nn.Module):
    def __init__(self, hchannel, lchannel):
        super(MFIM_5, self).__init__()

        # current conv
        self.cur_ca = ChannelAttention(hchannel)

        self.srpc = SRPC(hchannel ,lchannel)
        self.conv1_1 = Conv1x1(hchannel, lchannel)   


        # previous conv
        self.downsample2 = nn.MaxPool2d(2, stride=2)
        self.pre_sa = SpatialAttention()



    def forward(self, x_pre, x_cur, att):
        

        # current conv
        cur_ca = x_cur.mul(self.cur_ca(x_cur, att))
        cur_ca_re = self.conv1_1(cur_ca)

        cur_srpc = self.srpc(cur_ca)

        # previois conv
        x_pre = self.downsample2(x_pre)
        pre_sa = cur_srpc.mul(self.pre_sa(x_pre))
        

        x_LocAndGlo = cur_srpc + pre_sa + cur_ca_re
        
        return x_LocAndGlo

# for conv1
class MFIM_1(nn.Module):
    def __init__(self, hchannel, cchannel, schannel, lchannel):
        super(MFIM_1, self).__init__()
        # current conv

        self.cur_ca = ChannelAttention(hchannel)
        self.conv1_1 = Conv1x1(hchannel, lchannel)   
        self.srpc = SRPC(hchannel, lchannel)


        # latter conv
        self.conv3_1 = Conv1x1(schannel, lchannel) 
        self.msa = MSA_head(lchannel) 


    def forward(self, x_cur, x_lat, att):
        
        # current conv
 
        cur_ca = x_cur.mul(self.cur_ca(x_cur, att))
        cur_ca_re = self.conv1_1(cur_ca)

        cur_srpc = self.srpc(cur_ca)        

        # latter conv
        x_lat = self.conv3_1(x_lat)
        lat_sa = cur_srpc.mul(self.msa(x_lat))

        x_LocAndGlo = cur_srpc + lat_sa + cur_ca_re

        return x_LocAndGlo


    # for conv2/3/4
class MFIM(nn.Module):
    def __init__(self, hchannel, cchannel, schannel, lchannel):
        super(MFIM, self).__init__()

        # current conv
        self.cur_ca = ChannelAttention(hchannel)

        self.srpc = SRPC(hchannel, lchannel)

        self.conv1_1 = Conv1x1(hchannel, lchannel)   

        # previous conv
        self.downsample2 = nn.MaxPool2d(2, stride=2)
        self.pre_sa = SpatialAttention()
        self.msca = MSCA()

        # latter conv
        self.conv3_1 = Conv1x1(cchannel, lchannel) 
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.lat_sa = SpatialAttention()
        self.msa = MSA_head(lchannel) 

    def forward(self, x_pre, x_cur, x_lat, att):

        # current conv
        cur_ca = x_cur.mul(self.cur_ca(x_cur, att))
        cur_ca_re = self.conv1_1(cur_ca)
  
        cur_srpc = self.srpc(cur_ca)  
        
        # previois conv
        if x_pre.size()[2:]!=x_cur.size()[2:]:
            x_pre = self.downsample2(x_pre)
        pre_sa = cur_srpc.mul(self.pre_sa(x_pre))

        # latter conv
        x_lat = self.conv3_1(x_lat)
        x_lat = self.upsample2(x_lat)
        lat_sa = cur_srpc.mul(self.msa(x_lat))

        x_LocAndGlo = cur_srpc + pre_sa + lat_sa + cur_ca_re

        return x_LocAndGlo


class CARM(nn.Module):
    def __init__(self, channel=64):
        super(CARM, self).__init__()


        self.asyConv = asyConv(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False)
        self.atrConv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3, stride=1), nn.BatchNorm2d(channel), nn.PReLU()
        )
        
        self.cat1 = ConvBNR(channel * 2, channel, kernel_size=3)
        self.cat2 = ConvBNR(channel * 2, channel, kernel_size=3)
        self.conv = ConvBNR(channel, channel, kernel_size=3)

        self.gate   = nn.Conv2d(channel, 1, kernel_size=1, bias=True)
        self.deconv = TransBasicConv2d(channel, channel, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        self.S = nn.Conv2d(channel, 1, 3, stride=1, padding=1)
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y):


        xy = self.cat1(torch.cat((x, y), dim=1)) + y
        cat_fea = self.conv(xy)
        att = self.gate(cat_fea)
        att_soft = self.softmax(att)
        fea = cat_fea *att_soft

        fea_atr = self.atrConv(fea)
        fea_asy = self.asyConv(fea)

        fea_fuse = self.cat2(torch.cat((fea_asy, fea_atr), dim=1))

        fea_fuse = self.dropout(fea_fuse)
        fea_fuse = self.deconv(fea_fuse)


        s = self.S(fea_fuse)
        
        return fea_fuse, s


class MFNet(nn.Module):
    def __init__(self, channel=32):
        super(MFNet, self).__init__()
        #Backbone model
        # ---- ResNet50 Backbone ----
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)

        
        self.egm = EGM()       

        self.mfim5 = MFIM_5(2048, 512)
        self.mfim4 = MFIM(1024, 2048, 512, 512)
        self.mfim3 = MFIM(512, 1024, 512, 256)
        self.mfim1 = MFIM(256, 512, 256, 64)

        self.lateral_conv0 = Conv1x1(512, 64)
        self.lateral_conv1 = Conv1x1(512, 64)
        self.lateral_conv2 = Conv1x1(256, 64)
        self.lateral_conv3 = Conv1x1(64, 64)

        self.carm5 = CARM()
        self.carm4 = CARM()
        self.carm3 = CARM()
        self.carm2 = CARM()

        self.upsample16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample0 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        
        self.predict5 = nn.Conv2d(in_channels=128,out_channels=1,kernel_size=3,stride=1,padding=1)


    def forward(self, x_rgb):
        x0, x1, x2, x3, x4 = self.resnet(x_rgb)
        
        edge = self.egm(x4, x2)
        edge_att = torch.sigmoid(edge)

        # print(x4.shape)

        x5_mfim = self.mfim5(x3, x4, edge_att)
        x4_mfim = self.mfim4(x2, x3, x4, edge_att)

        x3_mfim = self.mfim3(x1, x2, x3, edge_att)

        x2_mfim = self.mfim1(x0, x1, x2, edge_att)

        x5_mfim = self.lateral_conv0(x5_mfim)
        # print(x5_mfim.shape)
        
        x4_mfim = self.lateral_conv1(x4_mfim)
        x3_mfim = self.lateral_conv2(x3_mfim)
        x2_mfim = self.lateral_conv3(x2_mfim)
        # print(s5_fea.shape)

        s5_fea ,s5= self.carm5(x5_mfim, x5_mfim)
        s4_fea ,s4= self.carm4(x4_mfim, s5_fea)
        s3_fea ,s3= self.carm3(x3_mfim, s4_fea)
        s2_fea ,s2= self.carm2(x2_mfim, s3_fea)


        s2 = self.upsample2(s2)
        s3 = self.upsample4(s3)
        s4 = self.upsample8(s4)
        s5 = self.upsample16(s5)

        edge = F.interpolate(edge_att, scale_factor=8, mode='bilinear', align_corners=False)

        return s2, s3, s4, s5, edge
