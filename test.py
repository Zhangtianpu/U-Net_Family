import os
import yaml
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
def dotproduct(seg,cls):
    B,N,H,W=seg.size()
    seg=seg.view(B,N,H*W)
    re=torch.einsum("ijk,ij->ijk",seg,cls)
    re=re.view(B,N,H,W)
    return re

if __name__ == '__main__':
    input=torch.rand(size=(64,2,12,12))
    output=torch.rand(size=(64,2,12,12))
    pool=nn.AdaptiveMaxPool2d(1)
    cls=pool(input)
    cls=F.sigmoid(cls)
    cls=torch.flatten(cls,start_dim=1,end_dim=-1)
    cls=torch.argmax(cls,dim=1)
    cls=cls[:,np.newaxis].float()
    re=dotproduct(output,cls)
    print(re.shape)
