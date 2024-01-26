import torch
from torch import nn
from MyLoss.dice_loss import DiceLoss
class BCEWithDiceLoss_(nn.Module):
    def __init__(self,num_classes,yita=None,smoothing=1):
        """
        :param num_classes: int. It is equivalent with the number of output of model
        :param yita: list. It represents the weight for each encoder output when model has multiple encoder output.
                           The length of yita should align with the number of output of model
        :param smoothing: int. It is serverd as a minor constant, added to molecule and denominator.
                               As calculating dice loss, it can be used to prevent denominator from being zero.
        """
        super(BCEWithDiceLoss_,self).__init__()
        self.num_class=num_classes
        self.smoothing=smoothing
        self.yita=yita
        self.diceloss = DiceLoss(num_classes=self.num_class,smoothing=self.smoothing)
        self.bceloss = nn.BCEWithLogitsLoss()


    def forward(self,pred,target):
        if not isinstance(pred,list):

            dice_loss=self.diceloss(pred,target)
            bce_loss=self.bceloss(pred,target)
            return dice_loss+bce_loss
        total_loss=0
        yita=[1]*len(pred)
        if self.yita is not None:
            yita=self.yita
        for index,p in enumerate(pred):
            dice_loss=self.diceloss(p,target)
            bce_loss = self.bceloss(p, target)
            loss=dice_loss+bce_loss
            total_loss+=yita[index]*loss
        return total_loss
