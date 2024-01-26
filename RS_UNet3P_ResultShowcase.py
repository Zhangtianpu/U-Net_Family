import Utils
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import yaml
from ModelNet.U_Net3P import UNet3P
from Dataset.RemoteSensingDatasets import RemoteSensingData
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def plot_curve(x, train, test, title, xlabel, ylabel, is_save=False, path=None):
    plt.figure(figsize=(20, 5))
    plt.plot(x, train)
    plt.plot(x, test)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(['train', 'test'])
    plt.title(title)
    plt.grid(visible=True, linestyle='--')

    if is_save:
        if path is None:
            raise Exception("Path is None")
        plt.savefig(path, dpi=600, format='png', bbox_inches='tight')
    plt.show()


def case_study_display(data_config, model_config, train_config,
                       sample_number, epoch, is_save=False, path=None):
    model = UNet3P(channel_list=model_config.get("channel_list"),
                   out_channel=model_config.get("output_channel"),
                   bias=train_config.get("bias"),
                   CGM=model_config.get('CGM'),
                   deep_supervision=model_config.get('deep_supervision')).to(device)

    Utils.load_model(model, experiment_dir=train_config.get('experiment_folder'),
                     model_name=train_config.get('model_name'), epoch=epoch, device=device)

    testDataset = RemoteSensingData(data_folder=data_config.get('data_folder'), resize=data_config.get('resize'),
                                    train=False)

    palette = np.array(testDataset.colormap).astype(dtype=np.uint8)

    _, figs = plt.subplots(nrows=sample_number, ncols=3, figsize=(12, 10))
    for index in range(sample_number):
        input_image_path = testDataset.image_samples[index]
        input_img = Image.open(input_image_path)
        processed_image = testDataset.trans(input_img)
        processed_image = processed_image.unsqueeze(dim=0)

        label_image = testDataset.label_samples[index]
        label_img = Image.open(label_image)

        output = model(processed_image.to(device))
        output = torch.argmax(output[-1], dim=1).cpu().squeeze()
        pred_img = palette[output]
        if index == 0:
            figs[index, 0].set_title(label='Origin Img')
            figs[index, 1].set_title(label='Labeled Img')
            figs[index, 2].set_title(label='Predicted Img')
        figs[index, 0].imshow(input_img)
        figs[index, 1].imshow(label_img)
        figs[index, 2].imshow(pred_img)
    if is_save:
        if path is None:
            raise Exception("Path is None")
        plt.savefig(path, dpi=600, format='png', bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    config_file = './config/RS_Unet3P_config.yml'
    showcase_stored_folder = './ExperimentImages/RS_UNet3P'
    if os.path.exists(showcase_stored_folder) is False:
        os.makedirs(showcase_stored_folder)
    with open(config_file) as f:
        yaml_config = yaml.safe_load(f)

        data_config = yaml_config['data_configuration']
        model_config = yaml_config['model_configuration']
        train_config = yaml_config['train_configuration']

    logs = Utils.load_logs(folder_path=train_config.get("experiment_result_folder"),
                           filename=train_config.get('experiment_result_file'))

    train_loss = np.array(logs['train_loss'])
    test_loss = [loss for loss in logs['test_loss']]
    test_loss = np.array(test_loss)
    X = range(len(train_loss))

    train_acc = np.array(logs['train_acc'])
    test_acc = [acc for acc in logs['test_acc']]
    test_acc = np.array(test_acc)

    train_miou = np.array(logs['train_miou'])
    test_miou = [miou for miou in logs['test_miou']]
    test_miou = np.array(test_miou)

    plot_curve(X, train_loss, test_loss, "Learning Curve", "Epoch", "Loss", is_save=True,
               path=os.path.join(showcase_stored_folder, 'LearningCurve.png'))
    plot_curve(X, train_acc, test_acc, "Accuracy Curve", "Epoch", "Accuracy", is_save=True,
               path=os.path.join(showcase_stored_folder, 'AccuracyCurve.png'))
    plot_curve(X, train_miou, test_miou, "Mean Intersection Over Union", "Epoch", "MIOU", is_save=True,
               path=os.path.join(showcase_stored_folder, 'MIOU.png'))

    case_study_display(data_config=data_config, train_config=train_config,
                       model_config=model_config, sample_number=6, epoch=95,
                       is_save=True, path=os.path.join(showcase_stored_folder, 'CaseStudy.png'))
