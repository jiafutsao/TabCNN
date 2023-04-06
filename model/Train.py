#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import time

from util import *
from Metrics import *


# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[3]:


def crossentropy_by_string(pred, target):
    loss = 0
    crossentropy = nn.CrossEntropyLoss()
    for i in range(pred.shape[1]):
        loss += crossentropy(pred[:, i, :], target[:, i, :])
    return loss

def avg_acc(y_pred, y_true):
    return torch.mean(torch.eq(torch.argmax(y_pred, axis=-1), torch.argmax(y_true, axis=-1)).float())
    


# In[4]:


def train(model, train_dataloader, lr, optimizer):
    
    model.train()
    
    optimizer = optimizer

    train_loss = 0
    train_acc = 0
    train_pp, train_pr, train_pf, train_tp, train_tr, train_tf, train_TDR = 0, 0, 0, 0, 0, 0, 0
    start_time = time.perf_counter()

    for batch_idx, (data, labels) in enumerate(train_dataloader):
        progress_bar(batch_idx, len(train_dataloader), start_time)

        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model.forward(data)
        loss = crossentropy_by_string(output, labels)
        loss.backward()
        optimizer.step()

    # 速度考量，故只算一個epoch最後一個batch的準確率跟loss來當作這個epoch的指標，方便畫圖
    # 所以batch_size開大一點數值會比較準
    train_loss = loss.item() # 每一單筆資料(平均)的六條弦CrossEntropy總和
    train_acc = avg_acc(output, labels).item()
    train_pp = pitch_precision(output, labels).item()
    train_pr = pitch_recall(output, labels).item()
    train_pf = (train_pp * train_pr * 2) / (train_pp + train_pr)
    train_tp = tab_precision(output, labels).item()
    train_tr = tab_recall(output, labels).item()
    train_tf = (train_tp * train_tr * 2) / (train_tp + train_tr)
    train_TDR = train_tp / train_pp

    print("\nTrain Loss : ", train_loss, "\nTrain Acc : ", train_acc, "\npp/pr/pf : {:.4f}/{:.4f}/{:.4f}".format(train_pp, train_pr, train_pf), ", tp/tr/tf : {:.4f}/{:.4f}/{:.4f}".format(train_tp, train_tr, train_tf), ", TDR : {:.4f}".format(train_TDR))
    
    return train_loss, train_acc, train_pp, train_pr, train_pf, train_tp, train_tr, train_tf, train_TDR
    
def validation(model, val_dataloader):

    model.eval()

    val_loss = 0
    val_acc = 0
    val_pp, val_pr, val_pf, val_tp, val_tr, val_tf, val_TDR = 0, 0, 0, 0, 0, 0, 0
    start_time = time.perf_counter()

    for batch_idx, (data, labels) in enumerate(val_dataloader):
        progress_bar(batch_idx, len(val_dataloader), start_time)

        data = data.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            output = model.forward(data)
        loss = crossentropy_by_string(output, labels)

        val_loss += loss.item()
        val_acc += avg_acc(output, labels).item()
        val_pp += pitch_precision(output, labels).item()
        val_pr += pitch_recall(output, labels).item()
        #val_pf += val_pp * val_pr * 2 / (val_pp + val_pr)
        val_tp += tab_precision(output, labels).item()
        val_tr += tab_recall(output, labels).item()
        #val_tf += val_tp * val_tr * 2 / (val_tp + val_tr)
        #val_TDR += val_tp / val_pp

    val_loss /= len(val_dataloader) # 每一單筆資料(平均)的六條弦CrossEntropy總和
    val_acc /= len(val_dataloader)
    val_pp /= len(val_dataloader)
    val_pr /= len(val_dataloader)
    val_pf += val_pp * val_pr * 2 / (val_pp + val_pr)
    val_tp /= len(val_dataloader)
    val_tr /= len(val_dataloader)
    val_tf += val_tp * val_tr * 2 / (val_tp + val_tr)
    val_TDR += val_tp / val_pp

    print("\nVal Loss : ", val_loss, "\nVal acc : ", val_acc, "\npp/pr/pf : {:.4f}/{:.4f}/{:.4f}".format(val_pp, val_pr, val_pf), ", tp/tr/tf : {:.4f}/{:.4f}/{:.4f}".format(val_tp, val_tr, val_tf), ", TDR : {:.4f}".format(val_TDR))
    
    return val_loss, val_acc, val_pp, val_pr, val_pf, val_tp, val_tr, val_tf, val_TDR

def fit(model, train_dataloader, val_dataloader, lr, epochs, save_pth_file):
    
    results = {}
    
    results["train_loss"] = []
    results["train_acc"] = []
    results["train_pp"] = []
    results["train_pr"] = []
    results["train_pf"] = []
    results["train_tp"] = []
    results["train_tr"] = []
    results["train_tf"] = []
    results["train_TDR"] = []
    
    results["space"] = []
    
    results["val_loss"] = []
    results["val_acc"] = []
    results["val_pp"] = []
    results["val_pr"] = []
    results["val_pf"] = []
    results["val_tp"] = []
    results["val_tr"] = []
    results["val_tf"] = []
    results["val_TDR"] = []
    
    results["Epoch_num"] = []
    
    optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
    
    highest_acc = 0
    highest_pp = 0
    highest_pr = 0
    highest_pf = 0
    highest_tp = 0
    highest_tr = 0
    highest_tf = 0
    highest_TDR = 0
    highest_epoch_idx = 0
    
    for epoch in range(0, epochs):
        print("\n----------------------------Epoch : {}/{}------------------------------\n".format(epoch+1, epochs))
        
        results["Epoch_num"].append(epoch)
        
        ######## training
        train_loss, train_acc, train_pp, train_pr, train_pf, train_tp, train_tr, train_tf, train_TDR = train(model, train_dataloader, lr, optimizer)
        
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["train_pp"].append(train_pp)
        results["train_pr"].append(train_pr)
        results["train_pf"].append(train_pf)
        results["train_tp"].append(train_tp)
        results["train_tr"].append(train_tr)
        results["train_tf"].append(train_tf)
        results["train_TDR"].append(train_TDR)
        
        ######## validation
        val_loss, val_acc, val_pp, val_pr, val_pf, val_tp, val_tr, val_tf, val_TDR = validation(model, val_dataloader)
        
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
        results["val_pp"].append(val_pp)
        results["val_pr"].append(val_pr)
        results["val_pf"].append(val_pf)
        results["val_tp"].append(val_tp)
        results["val_tr"].append(val_tr)
        results["val_tf"].append(val_tf)
        results["val_TDR"].append(val_TDR)
        
        ## no meaning
        results["space"].append("///")
        
        ## record highest accoding validation tab recall
        #if val_acc > highest_acc:
        if val_tr > highest_tr:
            highest_acc = val_acc
            highest_pp = val_pp
            highest_pr = val_pr
            highest_pf = val_pf
            highest_tp = val_tp
            highest_tr = val_tr
            highest_tf = val_tf
            highest_TDR = val_TDR
            highest_epoch_idx = epoch
            torch.save(model.state_dict(), save_pth_file)
            
    return results, [highest_acc, highest_pp, highest_pr, highest_pf, highest_tp, highest_tr, highest_tf, highest_TDR, highest_epoch_idx]

