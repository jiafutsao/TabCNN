#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
import datetime
from torch.utils.data import DataLoader
from torchinfo import summary

from GuitarSetDataset import GuitarSetDataset
from TabCNN import TabCNN
from Train import *
from util import *


# In[ ]:


print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)


# In[ ]:


os.environ['CUDA_VISIBLE_DEVICES'] = "0"


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device = ", device)


# In[ ]:





# In[ ]:


#batch_size = 256
#lr = 1
#epochs = 1


# In[ ]:


batch_size = 512
lr = 1
epochs = 20


# In[ ]:





# In[ ]:


input_shape = (1, 192, 9)
num_strings = 6
num_classes = 21


# In[ ]:


root="../data/spec_repr/"
csv_path = "../data/spec_repr/id_c.csv"
save_path = "./saved/"
spec_repr = "c"


# In[ ]:





# In[ ]:





# In[ ]:


save_folder = save_path + spec_repr + "/" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "/"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)


# In[ ]:


###### save the log of model
tabcnn = TabCNN(input_shape, num_strings, num_classes).to(device)
with open(save_folder + "model.log", "w") as f:
    f.write(str(summary(tabcnn, (batch_size, input_shape[0], input_shape[1], input_shape[2]))))


# In[ ]:


folds = [0, 1, 2, 3, 4, 5]


# In[ ]:


highest = {}
highest["Accuracy"] = []
highest["pp"] = []
highest["pr"] = []
highest["pf"] = []
highest["tp"] = []
highest["tr"] = []
highest["tf"] = []
highest["TDR"] = []
highest["Epoch index"] = []


# In[ ]:


for fold in folds:
    print("\n//////////////////////////////////////////////////////////////////////////////////////////////")
    print("\n/////////////////////////////////////////// Fold-" + str(fold) + " ///////////////////////////////////////////")
    print("\n//////////////////////////////////////////////////////////////////////////////////////////////")
    split_folder = save_folder + str(fold) + "/"
    save_pth_file = split_folder + "model_state_dict.pth"
    
    # load data
    partition = partition_data(csv_path, save_folder, fold)
    dataset_train = GuitarSetDataset(partition["training"], root=root, spec_repr=spec_repr, con_win_size=9)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)
    dataset_val = GuitarSetDataset(partition["validation"], root=root, spec_repr=spec_repr, con_win_size=9)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=8)

    # create TabCNN
    tabcnn = TabCNN(input_shape, num_strings, num_classes).to(device)
    tabcnn.weight_init(mean=0, std=0.02)

    # train
    results, highest_list = fit(tabcnn, train_loader, val_loader, lr, epochs, save_pth_file)
    highest = highest_distribute(highest, highest_list)
    
    # save results for each split
    save_split_results_csv(results, split_folder)
    
    # print the highest
    print("\n########################### HIGHEST ACCURACY of Validation in Fold-{} ###########################".format(str(fold)))
    print("\nAccuracy : ", highest_list[0], "\npp/pr/pf : {:.4f}/{:.4f}/{:.4f}".format(highest_list[1], highest_list[2], highest_list[3]), ", tp/tr/tf : {:.4f}/{:.4f}/{:.4f}".format(highest_list[4], highest_list[5], highest_list[6]), ", TDR : {:.4f}".format(highest_list[7]))

save_results_csv(highest, epochs, batch_size, save_folder)

