import torch
from torch.utils.data import Dataset,DataLoader
import os
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
from torch.nn import functional
class RemoteSensingData(Dataset):

    def __init__(self,data_folder,resize=[256,256],train=True):
        self.image_folder=os.path.join(data_folder,'image')
        self.label_folder=os.path.join(data_folder,'label')
        self.resize=resize
        self.image_filepath_list = self.load_filepath_list(self.image_folder)
        self.label_filepath_list = self.load_filepath_list(self.label_folder)
        self.total_samples = len(self.image_filepath_list)


        self.label_meaning=['building,other']
        self.colormap=[[166, 166, 166],[160, 193, 133]]
        self.colormap2labels=self.colormap2labeling(self.colormap)

        self.image_samples,self.label_samples=self.getData(train=train,p=0.3)


        #Image Normalization
        """
        Data Argumentation
        """
        #randomly change the brightness, contrast,saturation of image
        img_resize=transforms.Resize(size=self.resize)
        colorJitter=transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.5)
        # vertificalFlip=transforms.RandomVerticalFlip()
        # horizonFlip=transforms.RandomHorizontalFlip()
        # rotation=transforms.RandomRotation(degrees=(0,360))
        self.trans=transforms.Compose([
            img_resize,
            colorJitter,
            # vertificalFlip,
            # horizonFlip,
            # rotation,
            transforms.ToTensor()
            ]
        )

    def __getitem__(self, item):
        image_path=self.image_samples[item]
        label_path=self.label_samples[item]
        image=Image.open(image_path)
        label=Image.open(label_path)
        label=F.resize(label,size=self.resize).convert('RGB')
        image=self.trans(image)
        label=self.label_index(label,self.colormap2labels)

        return image,label

    def __len__(self):
        return len(self.image_samples)

    def load_filepath_list(self,folder):
        filename_list=os.listdir(folder)
        return [os.path.join(folder,filename)for filename in filename_list]

    def colormap2labeling(self,colormap):
        colormap2labels=torch.zeros(256**3,dtype=torch.int64)
        for index,color in enumerate(colormap):
            colormap2labels[(color[0]*256+color[1])*256+color[2]]=index
        return colormap2labels

    def label_index(self,label,colormap2labels):
        label=np.array(label).astype('int32')
        idx=(label[:,:,0]*256+label[:,:,1])*256+label[:,:,2]
        return colormap2labels[idx]

    def getData(self,train=True,p=0.1):
        samples=int(self.total_samples*p)
        if train:
            image_samples=self.image_filepath_list[:samples]
            label_samples=self.label_filepath_list[:samples]
        else:
            image_samples=self.image_filepath_list[samples:]
            label_samples=self.label_filepath_list[samples:]

        return image_samples,label_samples



if __name__ == '__main__':

    train_datasets=RemoteSensingData(data_folder='/home/ztp/workspace/dataset/aerial_imagery_dataset',train=True)
    train_dataloader=DataLoader(dataset=train_datasets,batch_size=32)
    for data,label in train_dataloader:
        print(data.shape)
        print(label.shape)
        print(data)
