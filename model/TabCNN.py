#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn


# In[2]:


class TabCNN(nn.Module):
    
    def __init__(self, input_shape, num_strings, num_classes):
        super().__init__()
        
        self.audio_shape = list(input_shape)
        self.num_strings = num_strings
        self.num_classes = num_classes
        
        channels = [1, 32, 64, 64]
        for i in range(1, len(channels)):
            setattr(self, "conv"+str(i), nn.Sequential(
                nn.Conv2d(channels[i-1], channels[i], kernel_size=(3,3), stride=(1,1), padding=(0, 0)),
                nn.ReLU(inplace=True)
            ))
            self.audio_shape[0] = channels[i]
            self.audio_shape[1] = (self.audio_shape[1] - 3 + 1) // 1
            self.audio_shape[2] = (self.audio_shape[2] - 3 + 1) // 1
            
        self.maxpool2d = nn.MaxPool2d(kernel_size=(2,2))
        self.audio_shape[1] = self.audio_shape[1] // 2
        self.audio_shape[2] = self.audio_shape[2] // 2
        
        self.dropout1 = nn.Dropout(p=0.25)
        self.flatten = nn.Flatten()
        self.audio_shape = self.audio_shape[0] * self.audio_shape[1] * self.audio_shape[2]
        
        self.fc1 = nn.Linear(self.audio_shape, 128)
        self.audio_shape = 128
        
        self.activation = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(self.audio_shape, self.num_strings*self.num_classes)
        self.audio_shape = [self.num_strings, self.num_classes]
        
    def forward(self, X): # input dim is (batch_size, 1, 192, 9)
        X = self.conv1(X) # become (batch_size, 32, 190, 7)
        X = self.conv2(X) # become (batch_size, 64, 188, 5)
        X = self.conv3(X) # become (batch_size, 64, 186, 3)
        X = self.maxpool2d(X) # become (batch_size, 64, 93, 1)
        X = self.dropout1(X)
        X = self.flatten(X) # become (batch_size, 64*93*1)
        X = self.fc1(X) # become (batch_size, 128)
        X = self.activation(X)
        X = self.dropout2(X)
        X = self.fc2(X) # become (batch_size, 126)
        X = X.view(X.size(0), self.num_strings, self.num_classes)
        
        return X
        
    def weight_init(self, mean, std):
        for m in self._modules:
            if isinstance(self._modules[m], nn.Conv2d) or isinstance(self._modules[m], nn.Linear):
                self._modules[m].weight.data.normal_(mean, std)
                self._modules[m].bias.data.zero_()

