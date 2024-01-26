from PIL import Image
import os
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import functional
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader

class VOCSegmentationDataset(Dataset):
    def __init__(self,is_train,crop_size,voc_root):
        """
        :param is_train:
        :param crop_size: (h,w)
        :param voc_root:
        """
        self._VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                            [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                            [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                            [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                            [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                            [0, 64, 128]]

        self._VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                           'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                           'diningtable', 'dog', 'horse', 'motorbike', 'person',
                           'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

        self._colorMap2Label=self._colormap2label(self._VOC_COLORMAP)

        self.voc_root=voc_root
        self.rgb_mean=np.array([0.485,0.456,0.406])
        self.rgb_std=np.array([0.229,0.224,0.225])
        self._h,self._w=crop_size

        self._trans = transforms.Compose([
            transforms.Resize(size=(256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.rgb_mean, std=self.rgb_std)
        ])
        images_path,labels_path=self._read_file_list(self.voc_root,is_train=is_train)
        self._images_path=self._filter(images_path)
        self._labels_path=self._filter(labels_path)
        print("Valid examples have %d\n"%(len(self._images_path)))

    def __getitem__(self, item):
        image_path=self._images_path[item]
        label_path=self._labels_path[item]
        image=Image.open(image_path).convert('RGB')
        label=Image.open(label_path).convert('RGB')

        # image,label=self._voc_rand_crop(image,label,self._h,self._w)

        image=self._trans(image)
        label=functional.resize(label,size=(256,256))
        label=self._voc_label_indices(label,self._colorMap2Label)

        return image,label

    def __len__(self):
        return len(self._images_path)

    def _read_file_list(self,root, is_train=True):
        name = 'train.txt' if is_train else 'trainval.txt'
        text_file_path = os.path.join(root, 'TrainVal/ImageSets/Segmentation', name)
        with open(text_file_path, 'r') as f:
            image_names = f.read().split()
        images_path = [os.path.join(root, 'TrainVal/JPEGImages', image_name + ".jpg") for image_name in image_names]
        labels_path = [os.path.join(root, 'TrainVal/SegmentationClass', image_name + '.png') for image_name in
                       image_names]

        return images_path, labels_path

    def _filter(self,images_path):
        filter_images_path=[]
        for image_path in images_path:
            image=Image.open(image_path)
            if image.size[0]>=self._w and image.size[1]>=self._h:
                filter_images_path.append(image_path)
        return filter_images_path


    def _voc_rand_crop(self,image, label, height, width):
        """
        random crop image (PIL image) and label (PIL image)
        :param image:
        :param label:
        :param height: the height of photo after being cut
        :param width:  the width of photo after being cut
        :return:
        """

        i, j, h, w = transforms.RandomCrop.get_params(img=image,
                                                      output_size=(height, width))
        image = functional.crop(image, i, j, h, w)
        label = functional.crop(label, i, j, h, w)
        return image, label

    def _colormap2label(self,VOC_COLORMAP):
        colorMap2Label=torch.zeros(256**3,dtype=torch.int64)
        for index, colormap in enumerate(VOC_COLORMAP):
            colorMap2Label[(colormap[0]*256+colormap[1])*256+colormap[2]]=index
        return colorMap2Label

    def _voc_label_indices(self,label, colorMap2Label):
        """
        find out corresponding colormap with label (PIL image)
        :param label: PIL image
        :return: serial number of colormap
        """
        label = np.array(label).astype('int32')
        idx = (label[:, :, 0] * 256 + label[:, :, 1]) * 256 + label[:, :, 2]
        # [w,h] corresponding label with each pixel
        re = colorMap2Label[idx]
        return re


if __name__ == '__main__':

    root='/home/ztp/workspace/Dataset-Tool-Segmentation/data/VOC/VOC2012'
    vocDataset=VOCSegmentationDataset(is_train=True,crop_size=(300,480),voc_root=root)
    vocDataloader=DataLoader(dataset=vocDataset,batch_size=1,shuffle=False)
    for data in vocDataloader:
        imgs,label =data
        print(imgs.shape)

