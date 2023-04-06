#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
from torch.utils import data


# In[2]:


class GuitarSetDataset(data.Dataset):
    
    def __init__(self, list_IDs, root="../data/spec_repr/", spec_repr="c", con_win_size=9):
        
        self.list_IDs = list_IDs
        self.data_path = root + spec_repr + "/"
        self.con_win_size = con_win_size
        
        self.halfwin = con_win_size // 2
                
    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        # determine filename
        filename = "_".join(self.list_IDs[index].split("_")[:-1]) + ".npz"
        frame_idx = int(self.list_IDs[index].split("_")[-1])
        
        # load a context window centered around the frame index
        loaded = np.load(self.data_path + filename)
        full_x = np.pad(loaded["repr"], [(self.halfwin, self.halfwin), (0, 0)], mode="constant")
        sample_x = full_x[frame_idx: frame_idx + self.con_win_size]
        X = torch.tensor(np.expand_dims(np.swapaxes(sample_x, 0, 1), 0).astype("float32"))
        y = torch.tensor(loaded["labels"][frame_idx].astype("float32"))
        
        return X, y


# In[ ]:





# In[ ]:





# In[ ]:





# In[3]:


#import pandas as pd


# In[4]:


#list_IDs = list(pd.read_csv("../data/spec_repr/id_c.csv", header=None)[0])


# In[5]:


#data_split = 0
#
#partition = {}
#partition["training"] = []
#partition["validation"] = []
#
#for ID in list_IDs:
#    guitarist = int(ID.split("_")[0])
#    if guitarist == data_split:
#        partition["validation"].append(ID)
#    else:
#        partition["training"].append(ID)


# In[6]:


#partition["training"]


# In[7]:


#len(partition["training"])


# In[8]:


#train_dataset = GuitarSetDataset(partition["training"])


# In[9]:


#train_dataset[39345][0].size()

