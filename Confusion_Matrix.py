import torch

class ConfusionMatrix():
    def __init__(self,num_class):
        super().__init__()
        self.num_class=num_class
        self.matrix=torch.zeros((num_class,num_class),dtype=torch.int64).cuda()
        self.miou=None
        self.miou_withou_background=None
        self.accuracy=None

    def update_matrix(self,true,pred):
        with torch.no_grad():
            k=(true>=0)&(true<self.num_class)
            true_=true[k]
            index=self.num_class*true_+pred[k]
            current_matrix = torch.bincount(index, minlength=self.num_class ** 2).reshape(self.num_class, self.num_class)
            self.matrix+=current_matrix


    def MIOU(self,background_position=1):
        with torch.no_grad():
            diag_matrix=torch.diag(self.matrix)
            union=torch.sum(self.matrix,dim=1)+torch.sum(self.matrix,dim=0)-diag_matrix
            #[num_class]
            IOU_classes=diag_matrix/union
            self.miou=torch.sum(IOU_classes)/self.num_class
            self.miou_withou_background=(torch.sum(IOU_classes)-IOU_classes[background_position])/(len(IOU_classes)-1)
            self.accuracy=torch.sum(diag_matrix)/torch.sum(self.matrix)

            return self.miou,self.miou_withou_background,self.accuracy

