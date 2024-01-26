import torch
from torch import nn
from torch.nn import functional

class DiceLoss(nn.Module):
    def __init__(self,num_classes,smoothing=1):
        super(DiceLoss,self).__init__()
        self.num_classes=num_classes
        self.smoothing=smoothing




    def forward(self,pred,target):
        B,_,H,W=pred.size()
        num_samples=B*H*W
        #softmax
        pred=functional.softmax(pred,dim=1)

        #BCHW
        if self.num_classes>2:
            target=functional.one_hot(target,self.num_classes).permute(0,3,1,2)

        intersection=torch.einsum('bchw,bchw->bchw',pred,target)
        denominator=torch.square(pred)+torch.square(target)
        dice_coefficient=(2*intersection+self.smoothing)/(denominator+self.smoothing)
        dice_loss=-torch.sum(dice_coefficient)/num_samples
        # dice_loss=dice_loss/(torch.square(target)+torch.square(pred))
        # dice_loss=-torch.mean(dice_loss)

        return dice_loss

