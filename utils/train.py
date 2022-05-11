import copy
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import os
import torch
import pandas as pd


scaler = GradScaler()

os.environ["CUDA_VISIBLE_DEVICES"]="1"

def train_val(model, params):
    epochs=params["epochs"]
    loss_func=params["loss_func"]
    opt=params["optimizer"]
    train_dl=params["tdl"]
    val_dl=params["vdl"]
    scheduler=params["scheduler"]
    SAVE_PATH=params["weight_path"]
    
    
    history={
        "train_acc":[],
        "val_acc":[],
        "train_loss":[],
        "val_loss":[]
    }
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss=float('inf')
    
    for epoch in range(epochs):
        current_lr=get_lr(opt)
        print(f'Epoch [{epoch+1}/{epochs}], lr={current_lr}')
        model.train()
        train_loss, train_acc=loss_epoch(model,loss_func,train_dl,opt)
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        model.eval()
        with torch.no_grad():
            val_loss, val_acc=loss_epoch(model,loss_func,val_dl)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), SAVE_PATH+'/best.pth')
            print("Updated best model!!!")
        
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        
        
        scheduler.step()
        
        print(f'train loss={train_loss:.6f}, val loss={val_loss:.6f} | train acc={train_acc*100:.3f}, val_acc={val_acc*100:.3f}')
        print("="*10)
    
    pd.DataFrame(history).to_csv(SAVE_PATH+'/res.csv', index=False)
    
def get_lr(opt):
    return opt.param_groups[0]['lr']

def loss_epoch(model,loss_func,dataset_dl,opt=None):
    total_loss=0.0
    total_acc=0.0
    len_data = len(dataset_dl.dataset)
    for x, y in tqdm(dataset_dl):
        x=x.cuda()
        y=y.cuda()
        with autocast():
            output=model(x)
            loss,acc=loss_batch(loss_func, output, y, opt)
        total_loss+=loss
        total_acc+=acc

    loss=total_loss/float(len_data)
    acc=total_acc/float(len_data)
    
    return loss, acc   
def loss_batch(loss_func, output, target, opt=None):
    loss = loss_func(output, target)
    with torch.no_grad():
        acc = cal_acc(output,target)
    if opt is not None:
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
#         loss.backward()
#         opt.step()
    return loss.item(), acc
def cal_acc(output, target):
    _,pred = output.max(1)
    ans = (pred==target).sum().item()
    return ans


