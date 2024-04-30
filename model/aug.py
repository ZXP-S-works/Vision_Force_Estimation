import torch
import torchvision.transforms.functional as F
import torch.nn as nn


# class RandomShiftsAug(nn.Module):
#     # modified from https://github.com/facebookresearch/drqv2/blob/main/drqv2.py
#     def __init__(self, pad):
#         super().__init__()
#         self.pad = pad
#
#     def forward(self, x):
#         n, c, h, w = x.size()
#         l = max(h, w)
#         padding = tuple([self.pad] * 4)
#         if h > w:
#             padding = (self.pad, self.pad + h - w, self.pad, self.pad)
#         elif h < w:
#             padding = (self.pad, self.pad, self.pad, self.pad + w - h)
#         x = F.pad(x, padding, 'replicate')
#         eps = 1.0 / (l + 2 * self.pad)
#         arange = torch.linspace(-1.0 + eps,
#                                 1.0 - eps,
#                                 l + 2 * self.pad,
#                                 device=x.device,
#                                 dtype=x.dtype)[:l]
#         arange = arange.unsqueeze(0).repeat(l, 1).unsqueeze(2)
#         base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
#         base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
#
#         shift = torch.randint(0,
#                               2 * self.pad + 1,
#                               size=(n, 1, 1, 2),
#                               device=x.device,
#                               dtype=x.dtype)
#         shift *= 2.0 / (l + 2 * self.pad)
#
#         grid = base_grid + shift
#         x_aug = F.grid_sample(x,
#                               grid,
#                               padding_mode='zeros',
#                               align_corners=False)
#         return x_aug[:, :, :h, :w]


class RotTransAug:
    def __init__(self, drot=15, dtrans=10, vflip=False):
        self.drot = drot
        self.dtrans = dtrans
        self.vflip = vflip

    def get_rot_trans_param(self, bs):
        rot = torch.FloatTensor(bs).uniform_(-self.drot, self.drot)
        trans = torch.FloatTensor(bs, 2).uniform_(-self.dtrans, self.dtrans)
        rot_mat = torch.FloatTensor(bs, 2, 2)
        rot_rad = torch.deg2rad(rot)
        rot_mat[:, 0, 0] = torch.cos(rot_rad)
        rot_mat[:, 0, 1] = -torch.sin(rot_rad)
        rot_mat[:, 1, 0] = torch.sin(rot_rad)
        rot_mat[:, 1, 1] = torch.cos(rot_rad)
        return rot, trans, rot_mat

    def aug(self, img, vec):
        b, c, h, w = img.shape
        rot, trans, rot_mat = self.get_rot_trans_param(b)
        aug_img = torch.stack([
            F.affine(image, angle=-rotation_angle.item(), translate=translation_offset.tolist(), scale=1, shear=0)
            for image, rotation_angle, translation_offset in
            zip(img, rot, trans)
        ])
        aug_vec = torch.matmul(rot_mat, vec.unsqueeze(2)).squeeze()
        if self.vflip and torch.randn(1) > 0:
            aug_img = F.vflip(aug_img)
            aug_vec[:, 0] = -aug_vec[:, 0]
        # adding noise doesn't help
        # aug_img += torch.randn_like(aug_img) * 0.01
        return aug_img, aug_vec
