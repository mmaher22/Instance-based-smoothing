import torch
import torch.nn as nn


"""### DenseNet"""

####network
class conv_blk(nn.Module):
    def __init__(self,in_channel,num_channel):
        super(conv_blk, self).__init__()
        self.blk=nn.Sequential(nn.BatchNorm2d(in_channel,eps=1e-3),
                          nn.ReLU(),
                          nn.Conv2d(in_channels=in_channel,out_channels=num_channel,kernel_size=3,padding=1))
        
    def forward(self, x):
        return self.blk(x)

class DenseBlock(nn.Module):
    def __init__(self,in_channel,num_convs,num_channels):
        super(DenseBlock,self).__init__()
        layers=[]
        for i in range(num_convs):
            layers+=[conv_blk(in_channel,num_channels)]
            in_channel=in_channel+num_channels
        self.net=nn.Sequential(*layers)

    def forward(self,x):
        for blk in self.net:
            y=blk(x)
            x=torch.cat((x,y),dim=1)
        return x


def transition_blk(in_channel,num_channels):
    blk=nn.Sequential(nn.BatchNorm2d(in_channel,eps=1e-3),
                      nn.ReLU(),
                      nn.Conv2d(in_channels=in_channel,out_channels=num_channels,kernel_size=1),
                      nn.AvgPool2d(kernel_size=2,stride=2))
    return blk


class DenseNet(nn.Module):
    def __init__(self,in_channel = 1,num_classes = 10, tmp_scale = True):
        super(DenseNet,self).__init__()
        self.block1=nn.Sequential(nn.Conv2d(in_channels=in_channel,out_channels=64,kernel_size=7,stride=2,padding=3),
                                  nn.BatchNorm2d(64,eps=1e-3),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
        self.tmp_scale = tmp_scale
        if tmp_scale:
          self.temperature = torch.nn.Parameter(torch.ones(1), requires_grad = False)
        num_channels, growth_rate = 64, 32  # num_channels
        num_convs_in_dense_blocks = [4, 4, 4, 4]

        layers=[]
        for i ,num_convs in enumerate(num_convs_in_dense_blocks):
            layers+=[DenseBlock(num_channels,num_convs,growth_rate)]
            num_channels+=num_convs*growth_rate
            if i!=len(num_convs_in_dense_blocks)-1:
                layers+=[transition_blk(num_channels,num_channels//2)]
                num_channels=num_channels//2
        layers+=[nn.BatchNorm2d(num_channels),nn.ReLU(),nn.AvgPool2d(kernel_size=3)]
        self.block2=nn.Sequential(*layers)
        self.dense=nn.Linear(248,10)

    def forward(self,x):
        y=self.block1(x)
        y=self.block2(y)
        y=y.view(-1,248)
        y=self.dense(y)
        if self.tmp_scale:
            y /= self.temperature
        return y

def densetNet(tmp_scale = True, num_classes = 10):
    return DenseNet(tmp_scale = tmp_scale, num_classes = num_classes)