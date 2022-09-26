import sparseconvnet as scn
import torch
import torch.nn as nn
import torch.optim as optim
import sys, os, time


class Naive_UNet(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        m = 32  # 16 or 32
        residual_blocks = True  # True or False
        block_reps = 2  # Conv block repetition factor: 1 or 2

        self.sparseModel = scn.Sequential().add(
            scn.InputLayer(config['dimension'], config['full_scale'], mode=4)).add(
            scn.SubmanifoldConvolution(config['dimension'], 3, m, 3, False)).add(
            scn.UNet(config['dimension'], block_reps, [m, 2*m, 3*m, 4*m, 5*m, 6*m, 7*m], residual_blocks)).add(
            scn.BatchNormReLU(m)).add(
            scn.OutputLayer(config['dimension']))
        self.linear = nn.Linear(m, 20)

    def forward(self, x, increment=False):
        x = self.sparseModel(x, increment)
        x = self.linear(x)
        return x
