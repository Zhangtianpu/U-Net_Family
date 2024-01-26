import torch
from ModelNet.U_NetPP import UNetPP
from Dataset.RemoteSensingDatasets import RemoteSensingData
from MyLoss.BCEWithDice4Upp import BCEWithDiceLoss_
import argparse
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from BasicTrainer.UnetPPBasicTrainer import Trainer
import Utils


def worker(rank, word_size, config_file):
    dist.init_process_group('nccl', rank=rank, world_size=word_size)
    torch.cuda.set_device(rank)

    with open(config_file) as f:
        yaml_config = yaml.safe_load(f)

        data_config = yaml_config['data_configuration']
        model_config = yaml_config['model_configuration']
        train_config = yaml_config['train_configuration']

    # get the label of GPU in current process
    train_config['local_rank'] = dist.get_rank()
    train_config['word_size'] = dist.get_world_size()
    train_config['device'] = "cuda:{}".format(train_config['local_rank'])

    # load dataset
    trainDataset = RemoteSensingData(data_folder=data_config.get('data_folder'), resize=data_config.get('resize'),
                                     train=True)
    testDataset = RemoteSensingData(data_folder=data_config.get('data_folder'), resize=data_config.get('resize'),
                                    train=False)

    # trainVocDataloader = DataLoader(dataset=trainVocDataset, batch_size=64, shuffle=True)
    # testVocDataLoader = DataLoader(dataset=testVocDataset, batch_size=64, shuffle=False)

    # make sure each process call independent subset from raw datasets
    # if we have two running processes and the batch_size=32, our model will train 32*2 samples at once.
    dist_trainSampler = torch.utils.data.distributed.DistributedSampler(trainDataset,
                                                                        shuffle=True)
    dist_testSampler = torch.utils.data.distributed.DistributedSampler(testDataset, shuffle=False)

    trainDataloader = DataLoader(dataset=trainDataset,
                                 batch_size=train_config.get('batch_size'),
                                 sampler=dist_trainSampler,
                                 num_workers=word_size)

    testDataloader = DataLoader(dataset=testDataset,
                                batch_size=train_config.get('batch_size'),
                                sampler=dist_testSampler,
                                num_workers=word_size)

    """
    create U-Net model
    """
    unetpp = UNetPP(channel_list=model_config.get("channel_list"),
                    num_classes=model_config.get("output_channel"),
                    deep_supervision=model_config.get('deep_supervision'),
                    bias=train_config.get("bias"),
                    residual_connection=model_config.get('residual_connection'),
                    forward_bn=model_config.get('forward_bn')).cuda()

    # to alter BatchNorm in UNet with SyncBatchNorm
    unetpp = nn.SyncBatchNorm.convert_sync_batchnorm(unetpp)
    torch.nn.parallel.DistributedDataParallel(module=unetpp,
                                              device_ids=[train_config.get('local_rank')],
                                              output_device=[train_config.get('local_rank')])
    """
    set up optimizer and loss function
    """
    loss = BCEWithDiceLoss_(num_classes=train_config.get("num_class"))
    optimizer = optim.Adam(params=unetpp.parameters(), lr=train_config.get('base_lr'), eps=train_config.get('epsilon'))
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=train_config['steps'],
                                                  gamma=train_config['lr_decay_ratio'])

    """
    train model
    """

    trainer = Trainer(model=unetpp,
                      loss=loss,
                      optimizer=optimizer,
                      scheduler=lr_scheduler,
                      train_loader=trainDataloader,
                      test_loader=testDataloader,
                      train_sampler=dist_trainSampler,
                      test_sampler=dist_testSampler,
                      train_config=train_config)
    trainer.train()
    Utils.save_logs(data=trainer.logs, folder_path=train_config.get('experiment_result_folder'),
                    filename=train_config.get('experiment_result_file'))

    dist.destroy_process_group()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch U-Net Training')

    # If it's 1, start off training progress with "torch.distributed.launch", If it's 0,begin with "mp.spawn"
    parser.add_argument('--multiprocessing_distributed', type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=-1)
    # It represents the total count of your GPU which used to train model
    parser.add_argument("--ngpus_per_node", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)

    # specify the configuration file path
    parser.add_argument('--config_file', type=str, default='./config/RS_UnetPP_config.yml')

    args = parser.parse_args()

    # set up random seed for current cpu
    torch.manual_seed(args.seed)
    # set up random seed for current gpu
    torch.cuda.manual_seed(args.seed)
    # set up random seed for all gpus
    torch.cuda.manual_seed_all(args.seed)
    # set up random seed for numpy
    np.random.seed(args.seed)

    if args.multiprocessing_distributed:
        local_rank = args.local_rank
        worker(local_rank, args.ngpus_per_node, args.config_file)
    else:
        mp.spawn(worker, nprocs=2, args=(args.ngpus_per_node, args.config_file))
