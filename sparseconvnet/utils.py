# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch, glob, os
from .sparseConvNetTensor import SparseConvNetTensor
from .metadata import Metadata
import sparseconvnet as scn
import pdb

def toLongTensor(dimension, x):
    if hasattr(x, 'type') and x.type() == 'torch.LongTensor':
        return x
    elif isinstance(x, (list, tuple)):
        assert len(x) == dimension
        return torch.LongTensor(x)
    else:
        return torch.LongTensor(dimension).fill_(x)


def optionalTensor(a, b):
    return getattr(a, b) if hasattr(a, b) else torch.Tensor()


def optionalTensorReturn(a):
    return a if a.numel() else None


def threadDatasetIterator(d):
    try:
        import queue
    except BaseException:
        import Queue as queue
    import threading

    def iterator():
        def worker(i):
            for k in range(i, len(d), 8):
                q.put(d[k])
        q = queue.Queue(16)
        for i in range(8):
            t = threading.Thread(target=worker, args=(i,))
            t.start()
        for _ in range(len(d)):
            item = q.get()
            yield item
            q.task_done()
        q.join()
    return iterator



def concatenate_feature_planes(input):
    output = SparseConvNetTensor()
    output.metadata = input[0].metadata
    output.spatial_size = input[0].spatial_size
    output.features = torch.cat([i.features for i in input], 1)
    return output


def extract_featrue(input, start, end):
    output = SparseConvNetTensor()
    output.metadata = input.metadata
    output.spatial_size = input.spatial_size

    output.features = input.features[:,start:end]
    return output


def upsample_feature(lr, hr, stride, bilinear = False):

#    pdb.set_trace()
    loc_lr = lr.get_spatial_locations().cuda().int()
    loc_hr = hr.get_spatial_locations().cuda().int()
    batch_size = torch.max(loc_hr[:,3]).item() + 1

    output = SparseConvNetTensor()
    output.metadata = hr.metadata
    output.spatial_size = hr.spatial_size

    lr_start_index = 0
    hr_start_index = 0
    for k in range(batch_size):
        if not bilinear:
            correspondence = scn.SCN.ResolutionBasedScattering(lr.metadata,loc_lr[loc_lr[:,3] == k,0:3], loc_hr[loc_hr[:,3] == k,0:3], stride)
            correspondence += lr_start_index
            correspondence[correspondence < 0] = 0
            output_feature  = torch.index_select(lr.features,0,correspondence.long())
        else:
            location_lr = loc_lr[loc_lr[:,3] == k,0:3]
            location_hr = loc_hr[loc_hr[:,3] == k,0:3]
            candidate = location_hr.clone().float()
            candidate = (candidate - (stride - 1) / 2) / stride
            ceil_candidate = torch.ceil(candidate)
            floor_candidate = torch.floor(candidate)
            diff = [ceil_candidate - candidate, candidate - floor_candidate]
            anchors = [ceil_candidate, floor_candidate]
#           ceil, up
#           floor, bottom
            for x in [0,1]:
                for y in [0,1]:
                    for z in [0,1]:
                        w = (1-diff[x][:,0])*(1-diff[y][:,1])*(1-diff[z][:,2])
                        query = location_hr.clone()
                        query[:,0] = anchors[x][:,0]
                        query[:,1] = anchors[y][:,1]
                        query[:,2] = anchors[z][:,2]
                        if x==0 and y ==0 and z==0:
                            weight = w
                            lr_candidates = query
                        else:
                            weight = torch.cat([weight,w],0)
                            lr_candidates = torch.cat([lr_candidates ,query],0)

            neighbor_correspondence = scn.SCN.ResolutionBasedScattering(lr.metadata,location_lr, lr_candidates, 1).long()
            weight[neighbor_correspondence < 0] = 0
            neighbor_correspondence.requires_grad = False
            weight.requires_grad = False
            neighbor_correspondence[neighbor_correspondence < 0] = 0
            hr_feature = torch.index_select(lr.features,0,neighbor_correspondence  + lr_start_index)  * weight.view([-1,1]).expand(-1,lr.features.shape[1])
            output_feature = sum([hr_feature[i * location_hr.shape[0]: i * location_hr.shape[0] + location_hr.shape[0]]  for i in range(8)])
            total_weight = sum([weight[i * location_hr.shape[0]: i * location_hr.shape[0] + location_hr.shape[0]]  for i in range(8)])
            output_feature /= total_weight.view([-1,1]).expand(-1,lr.features.shape[1])
        if k == 0:
            output.features = output_feature
        else:
            output.features = torch.cat([output.features, output_feature ], 0)
        lr_start_index = lr_start_index + torch.sum(loc_lr[:,3] == k)
        hr_start_index = hr_start_index + torch.sum(loc_hr[:,3] == k)
    return output


def add_feature_planes(input):
    output = SparseConvNetTensor()
    output.metadata = input[0].metadata
    output.spatial_size = input[0].spatial_size
    output.features = sum([i.features for i in input])
    return output

def append_tensors(tensors):
    spatial_size=tensors[0].spatial_size
    dimension=len(spatial_size)
    x=SparseConvNetTensor(
        features=torch.cat([t.features for t in tensors],0),
        metadata=Metadata(dimension),
        spatial_size=spatial_size)
    for t in tensors:
        x.metadata.appendMetadata(t.metadata,spatial_size)
    return x

class AddCoords(torch.nn.Module):
    def forward(self, input):
        output = SparseConvNetTensor()
        if input.features.numel():
            with torch.no_grad():
                coords = input.get_spatial_locations()
                d = (input.spatial_size.type_as(input.features)-1)/2
                coords=coords[:,:-1].type_as(input.features)/ d[None,:] - 1
            output.features = torch.cat([input.features,coords],1)
        else:
            output.features = input.features
        output.metadata = input.metadata
        output.spatial_size = input.spatial_size
        return output

def compare_sparse(x, y):
    cL,cR,L,R = x.metadata.compareSparseHelper(y.metadata, x.spatial_size)
    if x.features.is_cuda:
        cL=cL.cuda()
        cR=cR.cuda()
        L=L.cuda()
        R=R.cuda()
    e = 0
    if cL.numel():
        e += (x.features[cL]-y.features[cR]).pow(2).sum()
    if L.numel():
        e += x.features[L].pow(2).sum()
    if R.numel():
        e += y.features[R].pow(2).sum()
    return e / (cL.numel() + L.numel() + R.numel())

def spectral_norm_svd(module):
    w=module.weight
    if w.ndimension()==3:
        w=w.view(-1,w.size(2))
    _,s,_=torch.svd(w)
    return s[0]

def pad_with_batch_idx(x,idx): #add a batch index to the list of coordinates
    return torch.cat([x,torch.LongTensor(x.size(0),1).fill_(idx)],1)

def batch_location_tensors(location_tensors):
    a=[]
    for batch_idx, lt in enumerate(location_tensors):
        if lt.numel():
            a.append(pad_with_batch_idx(lt,batch_idx))
    return torch.cat(a,0)

def checkpoint_restore(model,exp_name,name2,use_cuda=True,epoch=0):
    if use_cuda:
        model.cpu()
    if epoch>0:
        f=exp_name+'-%09d-'%epoch+name2+'.pth'
        assert os.path.isfile(f)
        print('Restore from ' + f)
        model.load_state_dict(torch.load(f))
    else:
        f=sorted(glob.glob(exp_name+'-*-'+name2+'.pth'))
        if len(f)>0:
            f=f[-1]
            print('Restore from ' + f)
            model.load_state_dict(torch.load(f))
            epoch=int(f[len(exp_name)+1:-len(name2)-5])
    if use_cuda:
        model.cuda()
    return epoch+1

def is_power2(num):
    return num != 0 and ((num & (num - 1)) == 0)
def checkpoint_save(model,exp_name,name2,epoch, use_cuda=True):
    f=exp_name+'-%04d-'%epoch+name2+'.pth'
    model.cpu()
    torch.save(model.state_dict(),f)
    if use_cuda:
        model.cuda()
    #remove previous checkpoints unless they are a power of 2 to save disk space
    epoch=epoch-1
    f=exp_name+'-%04d-'%epoch+name2+'.pth'
    if os.path.isfile(f):
        if not is_power2(epoch):
            os.remove(f)
