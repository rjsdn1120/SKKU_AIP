import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models

class BasicBlock(nn.Module):
    def __init__(self,in_channel,out_channel,stride=1):
        super(BasicBlock,self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channel,out_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channel),
        )
        
        self.i = nn.Sequential()
        self.relu=nn.ReLU(inplace=True)
        
        if stride!=1 or in_channel != out_channel:
            self.i = nn.Sequential(
                nn.Conv2d(in_channel,out_channel, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channel)
            )
        
    def forward(self,x):
        x = self.block(x) + self.i(x)
        x = self.relu(x)
        
        return x
        

class MyResnet(nn.Module):
    def __init__(self, block, num_blocks, num_class=10):
        super(MyResnet, self).__init__()
        self.k = 32
        k=self.k

        self.conv1 = nn.Conv2d(3,k,kernel_size=7,padding=3,stride=2)
        self.bn1 = nn.BatchNorm2d(k)
        self.relu = nn.ReLU(inplace=True)
        self.maxp = nn.MaxPool2d(kernel_size=2,stride=2)

        self.layer1 = self._make_layer(block,k,num_blocks[0],stride=1)
        self.layer2 = self._make_layer(block,k*2,num_blocks[1],stride=2)
        self.layer3 = self._make_layer(block,k*4,num_blocks[2],stride=2)
        self.layer4 = self._make_layer(block,k*8,num_blocks[3],stride=2)

        
        self.fc = nn.Linear(k*8,num_class)

    def _make_layer(self, block, out_channel, num_block, stride):
        strides = [stride] + [1]*(num_block-1)
        layers=[]
        for stride in strides:
            layers.append(block(self.k,out_channel,stride))
            self.k = out_channel
        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxp(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x=nn.AdaptiveAvgPool2d(output_size=(1, 1))(x)
        x=x.view(x.size(0),-1)
        x=self.fc(x)

        x=F.log_softmax(x)
        return x

class ConvLSTM(nn.Module):
    def __init__(self, param):
        super(ConvLSTM, self).__init__()
        num_classes = param["num_classes"]
        dr_rate= param["dr_rate"]
        lstm_layers=param["lstm_layers"]
        lstm_h = param["lstm_h"]
        mtype = param["mtype"]
  
        if mtype=="lstm":
            baseModel = resnet()
        elif mtype=="rlstm":
            baseModel = models.resnet18()
                
        num_features = baseModel.fc.in_features
        baseModel.fc = Identity()
        self.baseModel = baseModel
        self.dropout= nn.Dropout(dr_rate)
        self.rnn = nn.LSTM(num_features, hidden_size=lstm_h, num_layers=lstm_layers)
        self.fc1 = nn.Linear(lstm_h, num_classes)
    def forward(self, x):
        b, f, c, h, w = x.shape
        ii = 0
        y = self.baseModel(x[:,ii])
        output, (hn, cn) = self.rnn(y.unsqueeze(1))
        for ii in range(1, f):
            y = self.baseModel((x[:,ii]))
            out, (hn, cn) = self.rnn(y.unsqueeze(1), (hn, cn))
        out = self.dropout(out[:,-1])
        out = self.fc1(out) 
        return out 
    
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x    

class BasicBlock3d(nn.Module):
    def __init__(self,in_channel,out_channel,stride=1):
        super(BasicBlock3d,self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channel,out_channel,kernel_size=(3,3,3),stride=(1,2,2),padding=(1,3,3),bias=False),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channel,out_channel,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1),bias=False),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True)
        )
    
    def forward(self,x):
        return self.block(x)

class Conv3d(nn.Module):
    def __init__(self,num_class=20):
        super(Conv3d,self).__init__()
        k=64
        self.k=k
        self.conv1 = nn.Conv3d(3,k,kernel_size=(3,7,7),stride=(1,4,4),padding=(1,3,3),bias=False)
        self.bn1= nn.BatchNorm3d(k)
        self.relu=nn.ReLU(inplace=True)
        
        block = BasicBlock3d
        self.layer1=self._make_layer(block,k)
        self.layer2=self._make_layer(block,k*2)
        self.layer3=self._make_layer(block,k*4)
        self.layer4=self._make_layer(block,k*8)
        
        self.fc = nn.Linear(k*8,num_class)
        
    def _make_layer(self,block,out_channel):
        layers=[]
        for i in range(2):
            layers.append(block(self.k,out_channel))
            self.k=out_channel
        return nn.Sequential(*layers)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x=nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))(x)
        x=x.view(x.size(0),-1)
        x=self.fc(x)
        
        return x
        
        
        
        
    
def resnet():
    return MyResnet(BasicBlock,[3,3,6,6])