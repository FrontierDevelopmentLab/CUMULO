"""
Code adapted from "Invertible Residual Networks"
http://proceedings.mlr.press/v97/behrmann19a.html
ICML, 2019
"""

import pdb

import numpy as np
import torch
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import constraints

from src.iresnet_utils import injective_pad, ActNorm2D, Split
from src.iresnet_utils import squeeze as Squeeze
from src.iresnet_utils import MaxMinGroup
from src.matrix_utils import exact_matrix_logarithm_trace, power_series_matrix_logarithm_trace
from src.spectral_norm import spectral_norm_conv, spectral_norm_fc

class LogisticTransform(torch.distributions.Transform):
    r"""
    Transform via the mapping :math:`y = \frac{1}{1 + \exp(-x)}` and :math:`x = \text{logit}(y)`.
    """
    codomain = constraints.real
    domain = constraints.unit_interval
    bijective = True
    sign = +1

    def __eq__(self, other):
        return isinstance(other, LogisticTransform)

    def _call(self, x):
        return x.log() - (-x).log1p()

    def _inverse(self, y):
        return torch.sigmoid(y)

    def log_abs_det_jacobian(self, x, y):
        return F.softplus(y) + F.softplus(-y)


def logistic_distribution(loc, log_scale):
    scale = torch.exp(log_scale) + 1e-5
    base_distribution = distributions.Uniform(torch.zeros_like(loc), torch.ones_like(loc))
    transforms = [LogisticTransform(), distributions.AffineTransform(loc=loc, scale=scale)]
    logistic = distributions.TransformedDistribution(base_distribution, transforms)
    return logistic


def downsample_shape(shape):
    return (shape[0] * 4, shape[1] // 2, shape[2] // 2)


class conv_iresnet_block(nn.Module):
    def __init__(self, in_shape, int_ch, numTraceSamples=0, numSeriesTerms=0,
                 stride=1, coeff=.97, input_nonlin=True,
                 actnorm=True, n_power_iter=5, nonlin="elu"):
        """
        buid invertible bottleneck block
        :param in_shape: shape of the input (channels, height, width)
        :param int_ch: dimension of intermediate layers
        :param stride: 1 if no downsample 2 if downsample
        :param coeff: desired lipschitz constant
        :param input_nonlin: if true applies a nonlinearity on the input
        :param actnorm: if true uses actnorm like GLOW
        :param n_power_iter: number of iterations for spectral normalization
        :param nonlin: the nonlinearity to use
        """
        super(conv_iresnet_block, self).__init__()
        assert stride in (1, 2)
        self.stride = stride
        self.squeeze = Squeeze(stride)
        self.coeff = coeff
        self.numTraceSamples = numTraceSamples
        self.numSeriesTerms = numSeriesTerms
        self.n_power_iter = n_power_iter
        nonlin = {
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "softplus": nn.Softplus,
            "sorting": lambda: MaxMinGroup(group_size=2, axis=1)
        }[nonlin]

        # set shapes for spectral norm conv
        in_ch, h, w = in_shape
            
        layers = []
        if input_nonlin:
            layers.append(nonlin())

        in_ch = in_ch * stride**2
        kernel_size1 = 3 # kernel size for first conv
        layers.append(self._wrapper_spectral_norm(nn.Conv2d(in_ch, int_ch, kernel_size=kernel_size1, stride=1, padding=1),
                                                  (in_ch, h, w), kernel_size1))
        layers.append(nonlin())
        kernel_size2 = 1 # kernel size for second conv
        layers.append(self._wrapper_spectral_norm(nn.Conv2d(int_ch, int_ch, kernel_size=kernel_size2, padding=0),
                                                  (int_ch, h, w), kernel_size2))
        layers.append(nonlin())
        kernel_size3 = 3 # kernel size for third conv
        layers.append(self._wrapper_spectral_norm(nn.Conv2d(int_ch, in_ch, kernel_size=kernel_size3, padding=1),
                                                  (int_ch, h, w), kernel_size3))
        self.bottleneck_block = nn.Sequential(*layers)
        if actnorm:
            self.actnorm = ActNorm2D(in_ch)
        else:
            self.actnorm = None

    def forward(self, x, ignore_logdet=False):
        """ bijective or injective block forward """

        if self.actnorm is not None:
            x, an_logdet = self.actnorm(x)
        else:
            an_logdet = 0.0

        Fx = self.bottleneck_block(x)
        # Compute approximate trace for use in training
        if (self.numTraceSamples == 0 and self.numSeriesTerms == 0) or ignore_logdet:
            trace = torch.tensor(0.)
        else:
            trace = power_series_matrix_logarithm_trace(Fx, x, self.numSeriesTerms, self.numTraceSamples)

        # add residual to output
        y = Fx + x

        return y, trace + an_logdet

    def inverse(self, y, maxIter=100):
        # inversion of ResNet-block (fixed-point iteration)
        x = y
        for iter_index in range(maxIter):
            summand = self.bottleneck_block(x)
            x = y - summand

        if self.actnorm is not None:
            x = self.actnorm.inverse(x)

        return x
    
    def _wrapper_spectral_norm(self, layer, shapes, kernel_size):
        if kernel_size == 1:
            # use spectral norm fc, because bound are tight for 1x1 convolutions
            return spectral_norm_fc(layer, self.coeff, 
                                    n_power_iterations=self.n_power_iter)
        else:
            # use spectral norm based on conv, because bound not tight
            return spectral_norm_conv(layer, self.coeff, shapes,
                                      n_power_iterations=self.n_power_iter)


class scale_block(nn.Module):
    def __init__(self, steps, in_shape, int_dim, squeeze=False, n_terms=0, n_samples=0,
                 coeff=.9, input_nonlin=True, actnorm=True, split=True,
                 n_power_iter=5, nonlin="relu"):
        super(scale_block, self).__init__()
        self.in_shape = in_shape
        # if squeeze:
        #     self.squeeze = Squeeze(2)
        #     conv_shape = downsample_shape(in_shape)
        # else:
        self.squeeze = None
        conv_shape = in_shape

        if split:
            self.split = Split()
            n = int(conv_shape[0] // 2)
            out_shape1 = (n, conv_shape[1], conv_shape[2])
            out_shape2 = (conv_shape[0] - n, conv_shape[1], conv_shape[2])
            self.out_shapes = [out_shape1, out_shape2]
        else:
            self.split = None
            self.out_shapes = [conv_shape]

        self.stack = self._make_stack(steps, n_terms, n_samples, conv_shape, int_dim,
                                      input_nonlin, coeff, actnorm, n_power_iter, nonlin)

    @staticmethod
    def _make_stack(steps, n_terms, n_samples, in_shape, int_dim,
                    input_nonlin, coeff, actnorm, n_power_iter, nonlin):
        """ Create stack of iresnet blocks """
        block_list = nn.ModuleList()
        for i in range(steps):
            block_list.append(conv_iresnet_block(in_shape, int_dim, n_samples, n_terms,
                                                 stride=1, input_nonlin=True if input_nonlin else i > 0,
                                                 coeff=coeff, actnorm=actnorm,
                                                 n_power_iter=n_power_iter, nonlin=nonlin))

        return block_list

    def forward(self, x, ignore_logdet=False):
        if self.squeeze is not None:
            x = self.squeeze(x)

        traces = []
        z = x
        for block in self.stack:
            z, trace = block(z, ignore_logdet=ignore_logdet)
            traces.append(trace)

        trace = torch.zeros_like(traces[0])
        for k in range(len(traces)):
            trace += traces[k]

        if self.split is None:
            return [z], trace
        else:
            z1, z2 = self.split(z)
            return [z1, z2], trace

    def inverse(self, z, z2=None, maxIter=100):
        if self.split is None:
            x = z
        else:
            assert z2 is not None
            x = self.split.inverse(z, z2)

        for block in reversed(self.stack):
            x = block.inverse(x, maxIter=maxIter)

        if self.squeeze is None:
            return x
        else:
            return self.squeeze.inverse(x)


class multiscale_conv_iResNet(nn.Module):
    def __init__(self, in_shape, nBlocks, nStrides, nChannels, init_squeeze=False, inj_pad=0,
                 coeff=.9, nClasses=None, numTraceSamples=1, numSeriesTerms=1, n_power_iter=5,
                 actnorm=True, learn_prior=True, nonlin="relu", lin_classifier=True):
        super(multiscale_conv_iResNet, self).__init__()
        assert len(nBlocks) == len(nStrides) == len(nChannels)
        if init_squeeze:
            self.init_squeeze = Squeeze(2)
        else:
            self.init_squeeze = None

        if inj_pad > 0:
            self.inj_pad = injective_pad(inj_pad)
        else:
            self.inj_pad = None

        if init_squeeze:
            in_shape = downsample_shape(in_shape)
        in_shape = (in_shape[0] + inj_pad, in_shape[1], in_shape[2])  # adjust channels

        self.nBlocks = nBlocks
        self.nClasses = nClasses
        # parameters for trace estimation
        self.numTraceSamples = numTraceSamples
        self.numSeriesTerms = numSeriesTerms
        self.n_power_iter = n_power_iter

        self.stack, self.in_shapes = self._make_stack(in_shape, nBlocks,
                                                      nStrides, nChannels, numSeriesTerms, numTraceSamples,
                                                      coeff, actnorm, n_power_iter, nonlin)
        # make prior distribution
        self._make_prior(learn_prior)
        # make classifier
        self._make_classifier(self.final_shape(), lin_classifier)

    def final_shape(self):
        return self.stack[-1].out_shapes[-1]

    def z_shapes(self):
        shapes = []
        for block in self.stack:
            if len(block.out_shapes) == 2:
                shapes.append(block.out_shapes[0])
        shapes.append(self.final_shape())
        return shapes

    def get_in_shapes(self):
        return self.in_shapes

    def _make_stack(self, in_shape, nSteps, nStrides, nChannels, n_terms,
                    n_samples, coeff, actnorm, n_power_iter, nonlin):

        blocks = nn.ModuleList()
        n_blocks = len(nSteps)
        in_shapes = [in_shape]
        for i, (steps, stride, channels) in enumerate(zip(nSteps, nStrides, nChannels)):
            block = scale_block(steps, in_shape, channels,
                                stride == 2, n_terms, n_samples,
                                coeff, i > 0, actnorm,
                                i < n_blocks - 1,
                                n_power_iter, 
                                nonlin)  # split on all but last layer
            in_shape = block.out_shapes[-1]
            in_shapes.append(in_shape)
            blocks.append(block)
        return blocks, in_shapes

    def _make_prior(self, learn_prior):

        dim = np.prod(self.in_shapes[0])
        self.prior_mu = nn.Parameter(torch.zeros((dim,)).float(), requires_grad=learn_prior)
        self.prior_logstd = nn.Parameter(torch.zeros((dim,)).float(), requires_grad=learn_prior)

    def _make_classifier(self, final_shape, linear):
        if self.nClasses is None:
            
            self.logits = None

        else:
            self.bn1 = nn.BatchNorm2d(final_shape[0], momentum=0.9)

            if linear:
                self.logits = nn.Linear(final_shape[0], self.nClasses)

            else:
                self.logits = nn.Sequential(
                                nn.Linear(final_shape[0], final_shape[0] * 10),
                                nn.ReLU(),
                                nn.Linear(final_shape[0] * 10, self.nClasses))

    def classifier(self, z):
        out = F.relu(self.bn1(z))
        out = F.avg_pool2d(out, out.size(2))
        out = out.view(out.size(0), out.size(1))
        return self.logits(out)

    def prior(self):
        return distributions.Normal(self.prior_mu, torch.exp(self.prior_logstd))

    def logpz(self, z):
        return self.prior().log_prob(z.view(z.size(0), -1)).sum(dim=1)

    def forward(self, x, ignore_logdet=False):

        """ iresnet forward """
        if self.init_squeeze is not None:
            x = self.init_squeeze.forward(x)

        if self.inj_pad is not None:
            x = self.inj_pad.forward(x)

        zs = []
        traces = []
        cur_act = x
        for block in self.stack:
            this_zs, trace = block(cur_act, ignore_logdet=ignore_logdet)
            if len(this_zs) == 1:
                cur_act = this_zs[0]
            else:
                assert len(this_zs) == 2
                cur_act = this_zs[1]
                zs.append(this_zs[0])
            traces.append(trace)
        zs.append(cur_act)  # add last activation to zs

        # add logdets
        tmp_trace = torch.zeros_like(traces[0])
        for k in range(len(traces)):
            tmp_trace += traces[k]

        bs = zs[0].size(0)

        zs_flat = [z.view(bs, -1) for z in zs]
        z = torch.cat(zs_flat, 1)
        
        logpz = self.logpz(z)

        # classification head
        if self.nClasses is not None:
            logits = self.classifier(zs[-1])
        
        return logits, zs, logpz, tmp_trace

    def inverse(self, zs, max_iter=10):
        """ iresnet inverse """
        with torch.no_grad():
            cur_act = zs[-1]
            zs = zs[:-1]
            for block in reversed(self.stack):
                # if this block has a split
                if len(block.out_shapes) == 2:
                    # pass in top z and cur act
                    cur_act = block.inverse(zs[-1], cur_act, maxIter=max_iter)
                    # shorten zs
                    zs = zs[:-1]
                # if there is no split
                else:
                    cur_act = block.inverse(cur_act, maxIter=max_iter)

            x = cur_act
            if self.inj_pad is not None:
                x = self.inj_pad.inverse(x)

            if self.init_squeeze is not None:
                x = self.init_squeeze.inverse(x)
        return x

    def split_zs(self, z):
        zs = []
        cur_dim = 0
        for z_shape in self.z_shapes():
            z_dim = np.prod(z_shape)
            this_z = z[:, cur_dim: cur_dim + z_dim]
            this_z = this_z.view(z.size(0), *z_shape)
            zs.append(this_z)
            cur_dim += z_dim
        return zs


    def sample(self, batch_size, max_iter=10):
        """sample from prior and invert"""
        with torch.no_grad():
            prior = self.prior()
            z = prior.rsample((batch_size,))
            zs = self.split_zs(z)
            return self.inverse(zs, max_iter=max_iter)

    def set_num_terms(self, n_terms):
        for block in self.stack:
            for layer in block.stack:
                layer.numSeriesTerms = n_terms
