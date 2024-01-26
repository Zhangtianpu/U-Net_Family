import torch
from torch import nn
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self,alpha=0.6,gamma=2,eps=1e-7):
        super(FocalLoss,self).__init__()
        self.alpha=torch.tensor([alpha,1-alpha])
        self.gamma=gamma
        self.eps=torch.tensor(eps)

    def forward(self,pred,target):
        #pred:[B,C,H,W], target:[B,C,H,W] onehot
        pred=torch.softmax(pred,dim=1)
        log_pred=torch.log(pred+self.eps)
        loss=torch.einsum("bchw,bchw->bchw",log_pred,target)
        loss=-1*torch.pow(1-pred,self.gamma)*loss
        #[B,H,W,C]
        loss=loss.permute(0,2,3,1)
        loss=torch.einsum('c,bhwc->bhwc',self.alpha,loss)
        loss=torch.sum(loss,dim=-1)
        avg_loss=torch.mean(loss)
        return avg_loss

if __name__ == '__main__':
    f_loss=FocalLoss(alpha=0.6,gamma=2)
    pred=torch.randn(size=(2,2,3,3))
    target=torch.randint(low=0,high=2,size=(2,2,3,3))
    loss=f_loss(pred,target)
    print(loss)