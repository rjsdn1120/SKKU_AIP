import argparse
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from glob import glob
from torchvision import models
import numpy as np
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
import os
import copy
from tqdm import tqdm_notebook
from torchvision.transforms.functional import to_pil_image
import matplotlib.pylab as plt
from tqdm import tqdm
import random
import albumentations.pytorch
import albumentations as A
import cv2
import time
import pandas as pd
from model import *
from utils import *
import shutil
import sys

def getT():
    return str(time.time()).split('.')[0]

parser = argparse.ArgumentParser(description='AIP')
parser.add_argument('-mode', default="test")
parser.add_argument('-epochs', default=20)
parser.add_argument('-gpu', default=0)
parser.add_argument('-aug',default=0)
parser.add_argument('-lr',default=1e-3)
parser.add_argument('-mtype',default='lstm')
parser.add_argument('-saveName',default=getT())
parser.add_argument('-wdecay',default=1e-4)

args=parser.parse_args()

random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)
device = torch.cuda.device(int(args.gpu))
os.environ["CUDA_VISIBLE_DEVICES"]='1'
print('GPU Number : ',args.gpu)


path='/media/data2/rjsdn/AIP/data/pprm/'
# path='./data'
data = [sorted(glob(os.path.join(x[0],'*.jpg')))for x in os.walk(path)]
data = list(filter(None, data))
random.shuffle(data)
t = len(data)//10

train=data[t:]
test=data[:t]

            
mtype = args.mtype


train_transform = transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ]
)
if int(args.aug)==1:
    train_transform = transforms.Compose(
        [
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.ToTensor()
        ]
    )
val_transform = transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ]
)
batch_size = 64



D3=False

if mtype == "lstm" or mtype=="rlstm":
    params_model={
        "num_classes": 20,
        "dr_rate": 0.05,
        "lstm_layers": 3,
        "lstm_h": 256,
        "mtype": mtype
    }
    model = ConvLSTM(params_model)        
else:
    model = Conv3d()
    D3=True


train_ds=VideoDataset(train,train_transform,d3=D3)
test_ds=VideoDataset(test,val_transform,d3=D3)

train = DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=8)
test = DataLoader(test_ds,batch_size=batch_size,shuffle=False,num_workers=8)

# device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
model = model.cuda()     
if args.mode=='test': # 0: lstm, 1:rlstm, 2: 3dconv
    if args.mtype=='lstm':
        model.load_state_dict(torch.load('model/model0.pt'))
    elif args.mtype=='rlstm':
        model.load_state_dict(torch.load('model/model1.pt'))
    else:
        model.load_state_dict(torch.load('model/model2.pt'))
    model.eval()
    with torch.no_grad():
        cnt=0
        for x,y in tqdm(test):
            x=x.cuda()
            y=y.cuda()
            output=model(x)
            _,pred = output.max(1)
            cnt+= (pred==y).sum().item()
        
        acc = cnt/len(test.dataset)
        print(f'Test Accuracy={acc*100:.3f}')
        

        
elif args.mode=='train':        
    lr=float(args.lr)
    epochs = int(args.epochs)
    loss_func = nn.CrossEntropyLoss(reduction="sum")
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=float(args.wdecay))
    scheduler = CosineAnnealingLR(opt,T_max=epochs,eta_min=lr/10,last_epoch=-1)
    # scheduler = ReduceLROnPlateau(opt, mode='min',factor=0.5, patience=5,verbose=1)

    SAVE_PATH=f"./results/{args.saveName}"
    os.makedirs(SAVE_PATH,exist_ok=True)

    params_train={
        "epochs": epochs,
        "optimizer": opt,
        "loss_func": loss_func,
        "tdl": train,
        "vdl": test,
        "scheduler": scheduler,
        "weight_path": SAVE_PATH,
        }


    train_val(model,params_train)                   