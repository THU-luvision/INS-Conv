from xml.sax.handler import feature_external_ges
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

sys.path.append('')

config = {'dimension': 3, 'full_scale': 4096}

Model = model.Naive_UNet(config)

Model = Model.cuda()

points = torch.randint(1000, 2000, size=(10000, 3)).cuda()     # initial point xyz
features = torch.rand((10000, 3)).cuda()                      # initial point features


output = Model([points, features], increment=False) 


