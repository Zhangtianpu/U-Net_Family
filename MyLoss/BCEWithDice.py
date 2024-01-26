import torch
from torch import nn
from MyLoss.dice_loss import DiceLoss
class BCEWithDiceLoss_(nn.Module):
    def __init__(self,num_classes,smoothing=1):
        super(BCEWithDiceLoss_,self).__init__()
        self.num_class=num_classes
        self.smoothing=smoothing
        self.diceloss = DiceLoss(num_classes=self.num_class,smoothing=self.smoothing)
        self.bceloss = nn.BCEWithLogitsLoss()

    def forward(self,pred,target):
        dice_loss=self.diceloss(pred,target)
        bce_loss=self.bceloss(pred,target)
        return dice_loss+bce_loss