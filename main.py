from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
if __name__ == '__main__':
    image_folder='/home/ztp/workspace/dataset/aerial_imagery_dataset/image'
    image_name='8187.jpg'
    image_path=os.path.join(image_folder,image_name)
    img=Image.open(image_path)
    # (512,512)
    print(img.size)

    label_folder='/home/ztp/workspace/dataset/aerial_imagery_dataset/label'
    label_name='8187.png'
    label_path=os.path.join(label_folder,label_name)
    label_img=Image.open(label_path)
    plt.imshow(img)
    plt.show()
    plt.imshow(label_img)
    plt.show()
    palette=label_img.getpalette()
    palette=np.array(palette).reshape(-1,3)
    print(palette)


