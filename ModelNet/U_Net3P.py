import torch
from torch import nn
from collections import OrderedDict


class UNet3P(nn.Module):
    def __init__(self, channel_list, out_channel, deep_supervision=True, CGM=True, bias=False):
        super(UNet3P, self).__init__()
        self.channel_list = channel_list
        self.out_channel = out_channel
        self.CGM = CGM
        self.bias = bias
        self.decoder_channel = 64*len(self.channel_list[1:])
        self.deep_supervision = deep_supervision
        self.scale = len(channel_list) - 1

        self.encoder = Encoder(channel_list=self.channel_list, bias=self.bias)
        self.decoder = Decoder(channel_list=self.channel_list, out_channel=self.decoder_channel, bias=self.bias)
        self.final_conv = None
        self.deep_supervision_list = None

        if self.deep_supervision:
            self.deep_supervision_list = nn.ModuleList()
            for i in range(self.scale):
                if i == 0:
                    decoder_channel = self.channel_list[-1]
                else:
                    decoder_channel = self.decoder_channel

                self.deep_supervision_list.append(nn.Sequential(
                    nn.Conv2d(in_channels=decoder_channel, out_channels=self.out_channel,
                              kernel_size=3, stride=1, padding=1),
                    nn.Upsample(scale_factor=2 ** (self.scale - 1 - i), mode='bilinear',align_corners=True)
                ))
        else:
            self.final_conv = nn.Conv2d(in_channels=self.decoder_channel,
                                        out_channels=self.out_channel,
                                        kernel_size=3, stride=1, padding=1, bias=self.bias)

        if self.CGM:
            self.CGM_module = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Conv2d(in_channels=self.channel_list[-1], out_channels=2, kernel_size=1, stride=1, padding=0),
                nn.AdaptiveMaxPool2d(output_size=1),
                nn.Sigmoid(),
                nn.Flatten(start_dim=1, end_dim=-1),
            )

    def forward(self, input):
        encoder_list = self.encoder(input)
        decoder_list = self.decoder(encoder_list)
        re_list = []

        if self.CGM:
            # [B,2]
            classification = self.CGM_module(decoder_list[0])
            classification = torch.argmax(classification, dim=1)
            if self.deep_supervision:
                for index, decoder_re in enumerate(decoder_list):
                    supervision_decoder = self.deep_supervision_list[index](decoder_re)
                    supervision_decoder = supervision_decoder.permute(1, 2, 3, 0)
                    re = torch.einsum("chwb,b->bchw", supervision_decoder, classification)
                    re_list.append(re)
            else:
                # [B,C,H,W]
                re = self.final_conv(decoder_list[-1])
                # [C,H,W,B]
                re = re.permute(1, 2, 3, 0)
                # [B,C,H,W]
                re = torch.einsum("chwb,b->bchw", re, classification)
                re_list.append(re)

        else:
            re = self.final_conv(decoder_list[-1])
            re_list.append(re)
        return re_list


class Decoder(nn.Module):
    # channel_list:[3,64,128,256,512,1024]
    def __init__(self, channel_list, out_channel=320, bias=False):
        super(Decoder, self).__init__()
        self.channel_list = channel_list
        self.out_channel = out_channel
        self.bias = bias
        self.num_layers_without_last = len(self.channel_list) - 2
        self.decoder_path = nn.ModuleList()
        for index in range(0, self.num_layers_without_last):
            sub_decoder_path = nn.ModuleDict(
                {'encoder': nn.ModuleList(), 'decoder': nn.ModuleList(), 'fusion': nn.ModuleList()})
            encoder_layers = [i for i in range(self.num_layers_without_last - index)]
            # [1],[1,2]
            decoder_layers = [i for i in range(index + 1)]
            sub_decoder_path['fusion'].append(
                ConvBlock(in_channel=self.out_channel, out_channel=self.out_channel, bias=self.bias)
            )
            down_sampling_weight = self.num_layers_without_last - index - 1
            # [0,1,2]
            for encoder_layer_index in encoder_layers:
                # 3,2,1
                downsampling_weight = down_sampling_weight - encoder_layer_index
                downsampling_block = nn.Sequential(
                    nn.MaxPool2d(kernel_size=2 ** downsampling_weight,
                                 stride=2 ** downsampling_weight,
                                 padding=0),
                    nn.Conv2d(in_channels=channel_list[encoder_layer_index + 1],
                              out_channels=64, stride=1, padding=1, kernel_size=3,
                              bias=self.bias)
                )
                sub_decoder_path['encoder'].append(downsampling_block)
            upsample_weight = index + 1
            for decoder_layer_index in decoder_layers:
                upsampling_weight = upsample_weight - decoder_layer_index
                if decoder_layer_index == 0:
                    upsample_inchannel = self.channel_list[-1]
                else:
                    upsample_inchannel = self.out_channel

                upsampling_block = nn.Sequential(
                    nn.Upsample(scale_factor=2 ** upsampling_weight, mode='bilinear',align_corners=True),
                    nn.Conv2d(in_channels=upsample_inchannel,
                              out_channels=64, kernel_size=3, stride=1,
                              padding=1, bias=self.bias)
                )
                sub_decoder_path['decoder'].append(upsampling_block)

            self.decoder_path.append(sub_decoder_path)

    def forward(self, encoder_results):
        decoder_input = [encoder_results[-1]]

        for index, sub_decoder_path in enumerate(self.decoder_path):
            encoder_module_list = sub_decoder_path['encoder']
            decoder_module_list = sub_decoder_path['decoder']
            fusion_module_list = sub_decoder_path['fusion']

            fusion_list = []

            # convolution for contracting path
            for encoder_layer_index, encoder_block in enumerate(encoder_module_list):
                encoder_re = encoder_block(encoder_results[encoder_layer_index])
                fusion_list.append(encoder_re)

            # convoution for expansive path
            for decoder_layer_index, decoder_block in enumerate(decoder_module_list):
                decoder_re = decoder_block(decoder_input[decoder_layer_index])
                fusion_list.append(decoder_re)

            # feature aggregation, fusion_feature [B,64*5,H,W]
            fusion_feature = torch.cat(fusion_list, dim=1)
            fusion_feature = fusion_module_list[-1](fusion_feature)

            decoder_input.append(fusion_feature)

        return decoder_input


class Encoder(nn.Module):
    # channel_list:[3,64,128,256,512,1024]
    def __init__(self, channel_list, bias=False):
        super(Encoder, self).__init__()
        self.channel_list = channel_list
        self.bias = bias
        self.encoder_path = nn.ModuleList()
        for index, in_channel in enumerate(self.channel_list[:-1]):
            out_channel = channel_list[index + 1]
            sub_path = nn.ModuleList([ConvBlock(in_channel, out_channel, self.bias),
                                      nn.MaxPool2d(kernel_size=2, stride=2, padding=0)])
            self.encoder_path.append(sub_path)

    def forward(self, input):
        encoder_results = []
        for index, sub_path in enumerate(self.encoder_path):
            # convolution
            input = sub_path[0](input)
            encoder_results.append(input)
            # down-sampling
            if index < len(self.encoder_path) - 1:
                input = sub_path[1](input)

        return encoder_results


class ConvBlock(nn.Module):

    def __init__(self, in_channel, out_channel, bias=False):
        super(ConvBlock, self).__init__()

        conv_1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                           kernel_size=3, padding=1, stride=1, bias=bias)
        bn = nn.BatchNorm2d(num_features=out_channel)
        relu = nn.ReLU()
        self.block = nn.Sequential(OrderedDict([('conv_1', conv_1),
                                                ('bn', bn),
                                                ('relu', relu)]))

    def forward(self, input):
        return self.block(input)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    channel_list = [3, 64, 128, 256, 512]
    input = torch.randn(size=[2, 3, 256, 256], dtype=torch.float32).cuda()
    unetppp = UNet3P(channel_list=channel_list, out_channel=2,
                     deep_supervision=True, CGM=True, bias=False).cuda()
    output = unetppp(input)
    print(output)
