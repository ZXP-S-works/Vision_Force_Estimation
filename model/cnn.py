from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
from model.tf import *

class ResBlock(torch.nn.Module):
    def __init__(self, input_channels, hidden_dim, kernel_size):
        super(ResBlock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(hidden_dim)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),

        )
        self.relu = nn.ReLU(hidden_dim)
        self.upscale = None
        if input_channels != hidden_dim:
            self.upscale = nn.Sequential(
                nn.Conv2d(input_channels, hidden_dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            )

    def forward(self, xx):
        residual = xx
        out = self.layer1(xx)
        out = self.layer2(out)
        if self.upscale:
            out += self.upscale(residual)
        else:
            out += residual
        out = self.relu(out)
        return out


class ResUNet(torch.nn.Module):
    def __init__(self, n_input_channel=1, n_output_channel=3, n_hidden=16, kernel_size=3, resolution=128, dropout=0):
        super().__init__()
        self.h = n_hidden
        self.n_input_channel = n_input_channel
        self.n_output_channel = n_output_channel
        self.kernel_size = kernel_size
        self.resolution = resolution
        self.dropout = dropout

    def build(self):
        self.conv_down_1 = torch.nn.Sequential(OrderedDict([
            ('enc-e2conv-0', nn.Conv2d(self.n_input_channel, self.h, kernel_size=self.kernel_size, padding=1)),
            ('enc-e2relu-0', nn.ReLU()),
            ('enc-e2res-1', ResBlock(self.h, self.h, kernel_size=self.kernel_size)),
        ]))

        self.conv_down_2 = torch.nn.Sequential(OrderedDict([
            ('enc-pool-2', nn.MaxPool2d(2)),
            ('enc-e2res-2', ResBlock(self.h, 2 * self.h, kernel_size=self.kernel_size)),
        ]))
        self.conv_down_4 = torch.nn.Sequential(OrderedDict([
            ('enc-pool-3', nn.MaxPool2d(2)),
            ('enc-e2res-3', ResBlock(2 * self.h, 4 * self.h, kernel_size=self.kernel_size)),
        ]))
        self.conv_down_8 = torch.nn.Sequential(OrderedDict([
            ('enc-pool-4', nn.MaxPool2d(2)),
            ('enc-e2res-4', ResBlock(4 * self.h, 8 * self.h, kernel_size=self.kernel_size)),
        ]))
        self.conv_down_16 = torch.nn.Sequential(OrderedDict([
            ('enc-pool-5', nn.MaxPool2d(2)),
            ('enc-e2res-5', ResBlock(8 * self.h, 8 * self.h, kernel_size=self.kernel_size)),
        ]))

        self.conv_up_8 = torch.nn.Sequential(OrderedDict([
            ('dec-e2res-1', ResBlock(16 * self.h, 4 * self.h, kernel_size=self.kernel_size)),
        ]))
        self.conv_up_4 = torch.nn.Sequential(OrderedDict([
            ('dec-e2res-2', ResBlock(8 * self.h, 2 * self.h, kernel_size=self.kernel_size)),
        ]))
        self.conv_up_2 = torch.nn.Sequential(OrderedDict([
            ('dec-e2res-3', ResBlock(4 * self.h, 1 * self.h, kernel_size=self.kernel_size)),
        ]))
        self.conv_up_1 = torch.nn.Sequential(OrderedDict([
            ('dec-e2res-4', ResBlock(2 * self.h, 1 * self.h, kernel_size=self.kernel_size)),
            ('dec-e2conv-4', nn.Conv2d(1 * self.h, self.n_output_channel, kernel_size=self.kernel_size, padding=1)),
        ]))

        self.upsample_16_8 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample_8_4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample_4_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample_2_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('Free parameters: ', self.total_params)

    def forwardEncoder(self, obs):
        feature_map_1 = self.conv_down_1(obs)
        feature_map_2 = self.conv_down_2(feature_map_1)
        feature_map_4 = self.conv_down_4(feature_map_2)
        feature_map_8 = self.conv_down_8(feature_map_4)
        feature_map_16 = self.conv_down_16(feature_map_8)
        return feature_map_1, feature_map_2, feature_map_4, feature_map_8, feature_map_16

    def forwardDecoder(self, feature_map_1, feature_map_2, feature_map_4, feature_map_8, feature_map_16):
        concat_8 = torch.cat((feature_map_8, self.upsample_16_8(feature_map_16)), dim=1)
        feature_map_up_8 = self.conv_up_8(concat_8)

        concat_4 = torch.cat((feature_map_4, self.upsample_8_4(feature_map_up_8)), dim=1)
        feature_map_up_4 = self.conv_up_4(concat_4)

        concat_2 = torch.cat((feature_map_2, self.upsample_4_2(feature_map_up_4)), dim=1)
        feature_map_up_2 = self.conv_up_2(concat_2)

        concat_1 = torch.cat((feature_map_1, self.upsample_2_1(feature_map_up_2)), dim=1)
        feature_map_up_1 = self.conv_up_1(concat_1)

        return feature_map_up_1

    def forward(self, obs):
        feature_maps = self.forwardEncoder(obs)
        return self.forwardDecoder(*feature_maps)


def global_max_pool(x):
    return x.flatten(start_dim=2).max(-1)[0]


class ResNet(ResUNet):
    def __init__(self, n_input_channel=1, n_output_channel=16, n_hidden=16, kernel_size=3, resolution=128, dropout=0):
        super().__init__(n_input_channel=n_input_channel, n_output_channel=n_output_channel,
                         n_hidden=n_hidden, kernel_size=kernel_size, resolution=resolution, dropout=dropout)

    def build(self):
        pre_half_size = [1, 2, 4]
        scale = np.log2(self.resolution) - 7
        scale = int(np.round(max(scale, 0)))
        pre_half_size = pre_half_size[scale]
        self.pre_conv, self.conv_down, self.flat_dim = resnet(self.n_input_channel, self.h, pre_half_size, self.kernel_size)
        self.fc = torch.nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.flat_dim, 8 * self.h)),
            ('relu', nn.ReLU()),
            ('fc2', nn.Linear(8 * self.h, 8 * self.h)),
            ('relu', nn.ReLU()),
            ('fc3', nn.Linear(8 * self.h, self.n_output_channel)),
        ]))
        self.total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('Free parameters: ', self.total_params)

    def forward(self, obs):
        h = self.pre_conv(obs)
        h = self.conv_down(h)
        flat = global_max_pool(h)
        return self.fc(flat), flat


class MLP(ResUNet):
    def __init__(self, n_input_channel=1, n_output_channel=16, n_hidden=16, kernel_size=3, resolution=128, dropout=0):
        super().__init__(n_input_channel=n_input_channel, n_output_channel=n_output_channel,
                         n_hidden=n_hidden, kernel_size=kernel_size, resolution=resolution, dropout=dropout)

    def build(self):
        self.fc = torch.nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.n_input_channel, 8 * self.h)),
            ('relu', nn.ReLU()),
            ('fc2', nn.Linear(8 * self.h, 8 * self.h)),
            ('relu', nn.ReLU()),
            ('fc3', nn.Linear(8 * self.h, self.n_output_channel)),
        ]))
        self.total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('Free parameters: ', self.total_params)

    def forward(self, obs):
        return self.fc(obs), None


class ResNetCubic(ResNet):
    def __init__(self, n_input_channel=1, n_output_channel=16, n_hidden=16, kernel_size=3, resolution=128, dropout=0):
        super().__init__(n_input_channel=n_input_channel, n_output_channel=n_output_channel,
                         n_hidden=n_hidden, kernel_size=kernel_size, resolution=resolution, dropout=dropout)

    def forward(self, obs):
        out, z = super().forward(obs)
        cubic_idx = torch.linalg.norm(out, dim=1) > 1
        out[cubic_idx] = out[cubic_idx].pow(3)
        return out, z


def resnet(n_input_channel, h, pre_half_size, kernel_size):
    pre_conv = torch.nn.Sequential(OrderedDict([
        ('enc-e2conv-0', nn.Conv2d(n_input_channel, h,
                                   kernel_size=7,
                                   padding=3,
                                   stride=pre_half_size)),
        ('enc-e2relu-0', nn.ReLU())]))
    flat_dim = 8 * h
    conv_down = torch.nn.Sequential(OrderedDict([
        ('enc-e2res-1', ResBlock(h, h, kernel_size=kernel_size)),
        ('enc-pool-2', nn.MaxPool2d(2)),
        ('enc-e2res-2', ResBlock(h, 2 * h, kernel_size=kernel_size)),
        ('enc-pool-3', nn.MaxPool2d(2)),
        ('enc-e2res-3', ResBlock(2 * h, 2 * h, kernel_size=kernel_size)),
        ('enc-pool-4', nn.MaxPool2d(2)),
        ('enc-e2res-4', ResBlock(2 * h, 4 * h, kernel_size=kernel_size)),
        ('enc-pool-5', nn.MaxPool2d(2)),
        ('enc-e2res-5', ResBlock(4 * h, 4 * h, kernel_size=kernel_size)),
        ('enc-pool-6', nn.MaxPool2d(2)),
        ('enc-e2res-6', ResBlock(4 * h, flat_dim, kernel_size=kernel_size)),
    ]))
    return pre_conv, conv_down, flat_dim

class MlpResNet(ResUNet):
    def __init__(self, n_input_channel=1, n_output_channel=16, n_hidden=16, kernel_size=3, resolution=128, dropout=0,
                 n_history=3):
        super().__init__(n_input_channel=n_input_channel, n_output_channel=n_output_channel,
                         n_hidden=n_hidden, kernel_size=kernel_size, resolution=resolution, dropout=dropout)
        self.bottleneck_size = None
        self.n_history = n_history

    def build(self):
        pre_half_size = [1, 2, 4]
        scale = np.log2(self.resolution) - 7
        scale = int(np.round(max(scale, 0)))
        pre_half_size = pre_half_size[scale]
        self.pre_conv, self.conv_down, self.flat_dim = resnet(self.n_input_channel, self.h, pre_half_size, self.kernel_size)
        self.bottleneck_size = self.flat_dim * self.n_history
        self.fc = torch.nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.bottleneck_size, 8 * self.h)),
            ('relu', nn.ReLU()),
            ('fc2', nn.Linear(8 * self.h, 8 * self.h)),
            ('relu', nn.ReLU()),
            ('fc3', nn.Linear(8 * self.h, self.n_output_channel)),
        ]))
        self.total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('Free parameters: ', self.total_params)

    def forward(self, obs, return_entire_history = None):
        h = self.pre_conv(obs)
        h = self.conv_down(h)
        flat = global_max_pool(h)
        flat = flat.reshape(-1, self.bottleneck_size)
        return self.fc(flat), flat


class TransformerResNet(ResUNet):
    def __init__(self, n_input_channel=1, n_output_channel=16, n_hidden=16, n_servo_info=0, kernel_size=3,
                 resolution=128, dropout=0,
                 n_history=3, depth=3, heads=8, mlp_dim=256, num_classes=2, dim=256, dim_head=32, 
                 grad_on_one_frame=True):
        super().__init__(n_input_channel=n_input_channel, n_output_channel=n_output_channel,
                         n_hidden=n_hidden, kernel_size=kernel_size, resolution=resolution, dropout=dropout)
        self.bottleneck_size = None
        self.grad_on_one_frame = grad_on_one_frame
        self.n_history = n_history
        self.n_servo_info = n_servo_info
        pre_half_size = [1, 2, 4]
        scale = np.log2(self.resolution) - 7
        scale = int(np.round(max(scale, 0)))
        pre_half_size = pre_half_size[scale]
        self.pre_conv, self.conv_down, self.flat_dim = resnet(self.n_input_channel, self.h,
                                                              pre_half_size, self.kernel_size)

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(self.flat_dim),
            nn.Linear(self.flat_dim, dim - self.n_servo_info),
            nn.LayerNorm(dim - self.n_servo_info),
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('Free parameters: ', self.total_params)

    def forward(self, obs, servo_info=None, return_entire_history = False):
        if self.grad_on_one_frame:
            with torch.no_grad():
                flat = self.forward_cnn(obs)
            # randomly allowing 1 image over n_history to have gradient;
            # this could speed up training and avoid non-iid issue
            bs = obs.shape[0] // self.n_history
            idx = np.random.randint(0, self.n_history)
            g_obs = obs[idx::self.n_history]
            g_flat = self.forward_cnn(g_obs)
            flat[idx::self.n_history] = g_flat
        else:
            flat = self.forward_cnn(obs)
        return self.forward_tf(flat, servo_info, return_entire_history)

    def forward_cnn(self, obs):
        h = self.pre_conv(obs)
        h = self.conv_down(h)
        flat = global_max_pool(h)
        return flat

    def forward_tf(self, flat, servo_info, return_entire_history = False):
        x = flat.reshape(-1, self.n_history, self.flat_dim)  # b x h x flat_dim
        x = self.to_patch_embedding(x)  # b x h x dim
        pe = posemb_sincos_1d(x)  # b x h x dim
        x = x + pe
        if servo_info is not None:
            servo_info = servo_info.view((x.shape[0], x.shape[1], servo_info.shape[1]))
        x = torch.cat([x, servo_info], dim=-1) if servo_info is not None else x
        x = self.transformer(x)  # b x h x dim
        if not return_entire_history:
            x = x[:, -1]   # b x dim

        return self.linear_head(x), None
