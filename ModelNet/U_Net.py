import torch
from collections import OrderedDict
from torch import nn
from torch.nn import functional as F




class UNet(nn.Module):
    # constructing_path[(1,64),(64,128),(128,256),(256,512),(512,1024)]

    def __init__(self, channel_list, num_classes,
                 bias, residual_connection, forward_bn):
        """
        :param channel_list: list represents input channel of each level, for instance [3,64,128,256,512,1024]
        :param num_classes: int, final output is equivalent to the number of class
        :param bias: Bool, True represents to add bias in each convolution network
        :param residual_connection: Bool, True represents to add residual connection for each convolution block.
        :param forward_bn: Bool, True represents to implement batch normalization first when data is fed into each convolution network at convolution block.
        """
        super().__init__()
        self.channel_list = channel_list
        self.num_classes = num_classes
        self.bias = bias
        self.residual_connection = residual_connection
        self.forward_bn = forward_bn
        self.encoder = Encoder(channel_list=self.channel_list, bias=self.bias,
                               residual_connection=self.residual_connection, forward_bn=self.forward_bn)
        self.decoder = Decoder(channel_list=self.channel_list, bias=self.bias,
                               residual_connection=self.residual_connection, forward_bn=self.forward_bn)
        self.final_layer = nn.Sequential(OrderedDict([
            ('bn', nn.BatchNorm2d(self.channel_list[1])),
            ('relu', nn.ReLU()),
            ('conv', nn.Conv2d(in_channels=self.channel_list[1], out_channels=self.num_classes,
                               kernel_size=1, stride=1, padding=0))
        ]))
    def forward(self, input):
        encoder_list = self.encoder(input)
        decoder_output= self.decoder(encoder_list)
        output=self.final_layer(decoder_output)

        return output


class Encoder(nn.Module):
    def __init__(self,channel_list,bias,residual_connection,forward_bn):
        #channel_list: list, for example [3,64,128,256,512]
        super().__init__()
        self.channel_list=channel_list
        self.contracting_path=nn.ModuleList()
        self.bias=bias
        self.residual_connection=residual_connection
        self.forward_bn=forward_bn

        for index,in_channel in enumerate(self.channel_list[:-1]):
            output_channel=channel_list[index+1]
            contracting_block = nn.ModuleList(
                [ConvBlock(in_channel=in_channel, out_channel=output_channel, bias=self.bias,
                           residual_connection=self.residual_connection, forward_bn=self.forward_bn),
                 nn.MaxPool2d(kernel_size=2, stride=2)])


            self.contracting_path.append(contracting_block)


    def forward(self,input):
        output_list=[]
        for step, blocks in enumerate(self.contracting_path):
            # convolution
            re=blocks[0](input)
            output_list.append(re)
            # down-sampling
            if step<len(self.contracting_path)-1:
                input=blocks[1](re)

        return output_list

class Decoder(nn.Module):
    #in_channel 512 out_channel 1024
    #expensive_path[(1024,512),(512,256),(256,128),(128,64),(64,2)]
    def __init__(self, channel_list, bias, residual_connection, forward_bn):
        # channel_list: list,[3，64，128，256，512,1024]
        super().__init__()

        self.channel_list = channel_list
        self.bias = bias
        self.residual_connection = residual_connection
        self.forward_bn = forward_bn
        self.expensive_path = nn.ModuleList()
        #expensive_channel [1024,512,256,128,64]
        self.expensive_channel = list(reversed(self.channel_list[1:]))
        for index,in_channel in enumerate(self.expensive_channel[:-1]):
            output_channel=self.expensive_channel[index+1]
            expensive_block = nn.ModuleList([
                nn.ConvTranspose2d(in_channels=in_channel, out_channels=output_channel,
                                   kernel_size=2, stride=2, padding=0, bias=self.bias),
                ConvBlock(in_channel=in_channel, out_channel=output_channel,
                          bias=self.bias, residual_connection=self.residual_connection,
                          forward_bn=self.forward_bn)])
            self.expensive_path.append(expensive_block)


    def forward(self,encoder_results):
        """
        :param input:
        :param encoder_results:
        :return:
        """
        reverse_encoder_results=list(reversed(encoder_results))
        decoder_input=reverse_encoder_results[0]
        for index,encoder_re in enumerate(reverse_encoder_results[1:]):
            #up-sampling
            decoder_input=self.expensive_path[index][0](decoder_input)
            #skip connection
            decoder_merge=torch.cat([encoder_re,decoder_input],dim=1)
            #convolution
            decoder_input=self.expensive_path[index][1](decoder_merge)

        return decoder_input

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, bias=False, forward_bn=False,
                 residual_connection=False):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.bias = bias
        self.forward_bn = forward_bn
        self.residual_connection = residual_connection


        if self.residual_connection:
            self.conv_rc = nn.Conv2d(in_channels=self.in_channel,
                                     out_channels=self.out_channel,
                                     stride=1, padding=0,
                                     kernel_size=1, bias=self.bias)

        if forward_bn:
            self.conv_block = nn.Sequential(
                OrderedDict([('bn_1', nn.BatchNorm2d(self.in_channel)),
                             ('conv_1', nn.Conv2d(in_channels=self.in_channel,
                                                  out_channels=self.out_channel,
                                                  stride=1, padding=1,
                                                  kernel_size=3, bias=self.bias)),
                             ('relu_1', nn.ReLU()),
                             ('bn_2', nn.BatchNorm2d(self.out_channel)),
                             ('conv_2', nn.Conv2d(in_channels=self.out_channel,
                                                  out_channels=self.out_channel,
                                                  stride=1, padding=1,
                                                  kernel_size=3, bias=self.bias)),
                             ('relu_2', nn.ReLU())]))

        else:
            self.conv_block = nn.Sequential(
                OrderedDict([('conv_1', nn.Conv2d(in_channels=self.in_channel,
                                                  out_channels=self.out_channel,
                                                  stride=1, padding=1,
                                                  kernel_size=3, bias=self.bias)),
                             ('relu_1', nn.ReLU()),
                             ('bn_1', nn.BatchNorm2d(self.out_channel)),
                             ('conv_2', nn.Conv2d(in_channels=self.out_channel,
                                                  out_channels=self.out_channel,
                                                  stride=1, padding=1,
                                                  kernel_size=3, bias=self.bias)),
                             ('relu_2', nn.ReLU()),
                             ('bn_2', nn.BatchNorm2d(self.out_channel))]))

    def forward(self, input):
        output = self.conv_block(input)
        if self.residual_connection:
            rc_input = self.conv_rc(input)
            output = output + rc_input
            output = F.relu(output)
        return output


if __name__ == '__main__':
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    channel_list=[3,64,128,256,512,1024]
    input=torch.randn(size=[2,3,512,512],dtype=torch.float32).cuda()
    unet=UNet( channel_list=channel_list, num_classes=21,
                 bias=False, residual_connection=False, forward_bn=False).cuda()
    output=unet(input)
    print(output.shape)