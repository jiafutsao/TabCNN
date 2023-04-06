#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time
import sys
import pandas as pd
import numpy as np


# In[2]:


def progress_bar(num, total, start_time):
    rate = float(num)/total
    ratenum = int(100*rate)
    dur = time.perf_counter() - start_time
    
    r = '\r Progress:[{}{}] {}%, {:.2f}s' .format("*"*ratenum, " "*(100-ratenum), ratenum, dur)
    sys.stdout.write(r)
    sys.stdout.flush()

    
def partition_data(csv_path, save_folder, data_split):
    
    list_IDs = list(pd.read_csv(csv_path, header=None)[0])
    
    partition = {}
    partition["training"] = []
    partition["validation"] = []
    for ID in list_IDs:
        guitarist = int(ID.split("_")[0])
        if guitarist == data_split:
            partition["validation"].append(ID)
        else:
            partition["training"].append(ID)

    split_folder = save_folder + str(data_split) + "/"
    if not os.path.exists(split_folder):
        os.makedirs(split_folder)
        
    return partition

def highest_distribute(highest, highest_list):
    highest["Accuracy"].append(highest_list[0])
    highest["pp"].append(highest_list[1])
    highest["pr"].append(highest_list[2])
    highest["pf"].append(highest_list[3])
    highest["tp"].append(highest_list[4])
    highest["tr"].append(highest_list[5])
    highest["tf"].append(highest_list[6])
    highest["TDR"].append(highest_list[7])
    highest["Epoch index"].append(highest_list[8])
    return highest

def save_split_results_csv(results, split_folder):
    df = pd.DataFrame.from_dict(results)
    df.to_csv(split_folder + "results_" + split_folder.split("/")[-2] + ".csv")
    
def save_results_csv(highest, epochs, batch_size, save_folder):
    output = {}
    output["Accuracy"] = []
    output["pp"] = []
    output["pr"] = []
    output["pf"] = []
    output["tp"] = []
    output["tr"] = []
    output["tf"] = []
    output["TDR"] = []
    output["data"] = []
    output["Epoch index"] = []
    output["Epochs"] = [str(epochs), "", "", "", "", "", "", ""]
    output["Batch size"] = [str(batch_size), "", "", "", "", "", "", ""]
    for key in highest.keys():
        if key!="Epoch index":
            vals = highest[key]
            mean = np.mean(vals)
            std = np.std(vals)
            output[key] = vals + [mean, std]
        if key=="Epoch index":
            vals = highest[key]
            output[key] = vals + ["", ""]
    output["data"] = ["g0", "g1", "g2", "g3", "g4", "g5"]
    output["data"].append("mean")
    output["data"].append("std dev")
    df = pd.DataFrame.from_dict(output)
    df.to_csv(save_folder + "results.csv")

