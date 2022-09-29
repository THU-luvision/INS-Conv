import torch
import copy
import numpy as np
import math
import time
import sparseconvnet as scn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import sparseconvnet as scn
import model
import sys


config = {'dimension': 3, 'full_scale': 4096}

Model = model.Naive_UNet(config)

Model = Model.cuda()

points_0 = torch.randint(100, 150, size=(10000, 3)).cuda()     # frame 0 point xyz
features_0 = torch.rand((10000, 3)).cuda()                      # frame 0 point features

# the initial update of the network, use inccrement=False
# this will save a checkpoint of network feataures, do it every 100 frames.
output_0 = Model([points_0, features_0], increment=False)    


points_1 = torch.randint(100, 150, size=(1000, 3)).cuda()     # xyz of incremental points of frame 1
features_1 = torch.rand((1000, 3)).cuda()                      # features of incremental points of frame 1


# incremental update of the network, use inccrement=True
# just need to input the incremental points and their features, the input residuals will be computed automaticlly
# the output is the results of incremental points of frame 1
output_1 = Model([points_1, features_1], increment=True)    

'''
    ...
    more frame
'''
