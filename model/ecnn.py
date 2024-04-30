import torch
import numpy as np
import scipy
import os
import urllib.request
from urllib.error import HTTPError
from escnn.group import *
from escnn import group
from escnn import gspaces
from escnn import nn
from escnn.nn import GeometricTensor
import torch.nn.functional as F


class AdaptiveNormMaxPool(nn.NormMaxPool):
    def __init__(self, in_type):
        super().__init__(in_type, kernel_size=0, stride=1)

    def forward(self, input: GeometricTensor) -> GeometricTensor:
        r"""

        Run the norm-based max-pooling on the input tensor

        Args:
            input (GeometricTensor): the input feature map

        Returns:
            the resulting feature map

        """

        assert input.type == self.in_type

        b, c, hi, wi = input.tensor.shape

        # compute the output shape (see 'torch.nn.MaxPool2D')
        b, c, ho, wo = self.evaluate_output_shape(input.tensor.shape)
        ho, wo = 1, 1

        # compute the squares of the values of each channel
        # n = torch.mul(input.data, input.data)
        n = input.tensor ** 2

        # pre-allocate the output tensor
        output = torch.empty(b, c, ho, wo, device=input.tensor.device)

        # reshape the input to merge the spatial dimensions
        input = input.tensor.reshape(b, c, -1)

        # iterate through all field sizes
        for s, contiguous in self._contiguous.items():
            indices = getattr(self, f"indices_{s}")

            if contiguous:
                # if the fields were contiguous, we can use slicing

                # compute the norms
                norms = n[:, indices[0]:indices[1], :, :] \
                    .view(b, -1, s, hi, wi) \
                    .sum(dim=2) \
                    .sqrt()

                # run max-pooling on the norms-tensor
                _, indx = F.max_pool2d(norms,
                                       (hi, wi),
                                       self.stride,
                                       self.padding,
                                       self.dilation,
                                       self.ceil_mode,
                                       return_indices=True)

                # in order to use the pooling indices computed for the norms to retrieve the fields, they need to be
                # expanded in the inner field dimension
                indx = indx.view(b, -1, 1, ho * wo).expand(-1, -1, s, -1)

                # retrieve the fields from the input tensor using the pooling indeces
                output[:, indices[0]:indices[1], :, :] = input[:, indices[0]:indices[1], :] \
                    .view(b, -1, s, hi * wi) \
                    .gather(3, indx) \
                    .view(b, -1, ho, wo)

            else:
                # otherwise we have to use indexing

                # compute the norms
                norms = n[:, indices, :, :] \
                    .view(b, -1, s, hi, wi) \
                    .sum(dim=2) \
                    .sqrt()

                # run max-pooling on the norms-tensor
                _, indx = F.max_pool2d(norms,
                                       (hi, wi),
                                       self.stride,
                                       self.padding,
                                       self.dilation,
                                       self.ceil_mode,
                                       return_indices=True)


                # in order to use the pooling indices computed for the norms to retrieve the fields, they need to be
                # expanded in the inner field dimension
                indx = indx.view(b, -1, 1, ho * wo).expand(-1, -1, s, -1)

                # retrieve the fields from the input tensor using the pooling indeces
                output[:, indices, :, :] = input[:, indices, :] \
                    .view(b, -1, s, hi * wi) \
                    .gather(3, indx) \
                    .view(b, -1, ho, wo)

        # wrap the result in a GeometricTensor
        return GeometricTensor(output, self.out_type, coords=None)


class SO2SteerableCNN(torch.nn.Module):
    """modified from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/Geometric_deep_learning"""
    "/tutorial2_steerable_cnns_unanswered.html?highlight=steerable%20cnns"
    def __init__(self, n_input_channel=1, n_output_channel=3, n_hidden=16, kernel_size=3, resolution=128):
        super().__init__()
        self.h = n_hidden
        self.n_input_channel = n_input_channel
        assert n_output_channel == 2
        # self.pooling = nn.NormMaxPool
        # self.pooling = nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        # self.n_output_channel = n_output_channel
        # self.kernel_size = kernel_size
        # self.resolution = resolution
        super(SO2SteerableCNN, self).__init__()

        # the model is equivariant under all planar rotations
        self.r2_act = gspaces.rot2dOnR2(N=-1)

        # the group SO(2)
        G: SO2 = self.r2_act.fibergroup

        # the input image is a scalar field, corresponding to the trivial representation
        in_type = nn.FieldType(self.r2_act, self.n_input_channel * [self.r2_act.trivial_repr])

        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type

        # # We need to mask the input image since the corners are moved outside the grid under rotations
        # self.mask = nn.MaskModule(in_type, 29, margin=1)

        # convolution 1
        # first we build the non-linear layer, which also constructs the right feature type
        # we choose 8 feature fields, each transforming under the regular representation of SO(2) up to frequency 3
        # When taking the ELU non-linearity, we sample the feature fields on N=16 points
        activation1 = nn.FourierELU(self.r2_act, self.h, irreps=G.bl_irreps(3), N=16, inplace=True)
        out_type = activation1.in_type
        self.block1 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=7, padding=1, bias=False),
            nn.IIDBatchNorm2d(out_type),
            activation1,
        )
        # to reduce the downsampling artifacts, we use a Gaussian smoothing filter
        self.pool1 = nn.SequentialModule(
            nn.NormMaxPool(out_type, 2)
        )

        # convolution 2
        # the old output type is the input type to the next layer
        in_type = self.block1.out_type
        # the output type of the second convolution layer are 16 regular feature fields
        activation2 = nn.FourierELU(self.r2_act, 2 * self.h, irreps=G.bl_irreps(3), N=16, inplace=True)
        out_type = activation2.in_type
        self.block2 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.IIDBatchNorm2d(out_type),
            activation2
        )
        # to reduce the downsampling artifacts, we use a Gaussian smoothing filter
        self.pool2 = nn.SequentialModule(
            nn.NormMaxPool(out_type, 2)
        )

        # convolution 3
        # the old output type is the input type to the next layer
        in_type = self.block2.out_type
        # the output type of the third convolution layer are 32 regular feature fields
        activation3 = nn.FourierELU(self.r2_act, 4 * self.h, irreps=G.bl_irreps(3), N=16, inplace=True)
        out_type = activation3.in_type
        self.block3 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.IIDBatchNorm2d(out_type),
            activation3
        )
        self.pool3 = nn.SequentialModule(
            nn.NormMaxPool(out_type, 2)
        )

        # convolution 4
        # the old output type is the input type to the next layer
        in_type = self.block3.out_type
        # the output type of the fourth convolution layer are 64 regular feature fields
        activation4 = nn.FourierELU(self.r2_act, 4 * self.h, irreps=G.bl_irreps(3), N=16, inplace=True)
        out_type = activation4.in_type
        self.block4 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.IIDBatchNorm2d(out_type),
            activation4
        )
        self.pool4 = nn.SequentialModule(
            nn.NormMaxPool(out_type, 2)
        )

        # convolution 5
        # the old output type is the input type to the next layer
        in_type = self.block4.out_type
        # the output type of the fifth convolution layer are 96 regular feature fields
        activation5 = nn.FourierELU(self.r2_act, 8 * self.h, irreps=G.bl_irreps(3), N=16, inplace=True)
        out_type = activation5.in_type
        self.block5 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.IIDBatchNorm2d(out_type),
            activation5
        )

        # convolution 6
        # the old output type is the input type to the next layer
        in_type = self.block5.out_type
        # the output type of the sixth convolution layer are 64 regular feature fields
        activation6 = nn.FourierELU(self.r2_act, 8 * self.h, irreps=G.bl_irreps(3), N=16, inplace=True)
        out_type = activation6.in_type
        self.block6 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False),
            nn.IIDBatchNorm2d(out_type),
            activation6
        )
        self.pool5 = AdaptiveNormMaxPool(out_type)
        # self.pool3 = nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2, padding=0)
        # last 1x1 convolution layer, which maps the regular fields to c=64 invariant scalar fields
        # this is essential to provide *invariant* features in the final classification layer
        output_standard_type = nn.FieldType(self.r2_act, [self.r2_act.irrep(1)])
        self.standard_map = nn.R2Conv(out_type, output_standard_type, kernel_size=1, bias=False)

        self.total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('Free parameters: ', self.total_params)

        # # Fully Connected classifier
        # self.fully_net = torch.nn.Sequential(
        #     torch.nn.BatchNorm1d(c),
        #     torch.nn.ELU(inplace=True),
        #     torch.nn.Linear(c, n_classes),
        # )

    def forward(self, input: torch.Tensor):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = self.input_type(input)

        # mask out the corners of the input image
        # x = self.mask(x)

        # apply each equivariant block

        # Each layer has an input and an output type
        # A layer takes a GeometricTensor in input.
        # This tensor needs to be associated with the same representation of the layer's input type
        #
        # Each layer outputs a new GeometricTensor, associated with the layer's output type.
        # As a result, consecutive layers need to have matching input/output types
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)
        x = self.pool3(x)
        x = self.block4(x)
        x = self.pool4(x)

        x = self.block5(x)
        x = self.block6(x)

        # pool over the spatial dimensions
        x = self.pool5(x)

        # extract invariant features
        x = self.standard_map(x)

        # unwrap the output GeometricTensor
        # (take the Pytorch tensor and discard the associated representation)
        x = x.tensor.squeeze()

        return x