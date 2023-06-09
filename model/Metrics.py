#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch


# In[2]:


def tab2pitch(tab):
    pitch_vector = np.zeros(44)
    string_pitches = [40, 45, 50, 55, 59, 64]
    for string_num in range(len(tab)):
        fret_vector = tab[string_num]
        fret_class = torch.argmax(fret_vector, axis=-1)
        # 0 means that the string is closed
        if fret_class > 0:
            pitch_num = (fret_class - 1) + string_pitches[string_num] - 40
            pitch_vector[pitch_num] = 1
    return pitch_vector

def tab2bin(tab):
    tab_arr = np.zeros((6,20))
    for string_num in range(len(tab)):
        fret_vector = tab[string_num]
        fret_class = torch.argmax(fret_vector, axis=-1)
        # 0 means that the string is closed
        if fret_class > 0:
            fret_num = fret_class - 1
            tab_arr[string_num][fret_num] = 1
    return tab_arr


# In[3]:


def pitch_precision(pred, gt):
    pitch_pred = torch.tensor(np.array(list(map(tab2pitch, pred))))
    pitch_gt = torch.tensor(np.array(list(map(tab2pitch, gt))))
    numerator = torch.sum(torch.mul(pitch_pred, pitch_gt).flatten())
    denominator = torch.sum(pitch_pred.flatten())
    return (1.0 * numerator) / denominator

def pitch_recall(pred, gt):
    pitch_pred = torch.tensor(np.array(list(map(tab2pitch, pred))))
    pitch_gt = torch.tensor(np.array(list(map(tab2pitch, gt))))
    numerator = torch.sum(torch.mul(pitch_pred, pitch_gt).flatten())
    denominator = torch.sum(pitch_gt.flatten())
    return (1.0 * numerator) / denominator

def pitch_f_measure(pred, gt):
    p = pitch_precision(pred, gt)
    r = pitch_recall(pred, gt)
    f = (2 * p * r) / (p + r)
    return f

def tab_precision(pred, gt):
    # get rid of "closed" class, as we only want to count positives
    tab_pred = torch.tensor(np.array(list(map(tab2bin, pred))))
    tab_gt = torch.tensor(np.array(list(map(tab2bin, gt))))
    numerator = torch.sum(torch.multiply(tab_pred, tab_gt).flatten())
    denominator = torch.sum(tab_pred.flatten())
    return (1.0 * numerator) / denominator

def tab_recall(pred, gt):
    # get rid of "closed" class, as we only want to count positives
    tab_pred = torch.tensor(np.array(list(map(tab2bin, pred))))
    tab_gt = torch.tensor(np.array(list(map(tab2bin, gt))))
    numerator = torch.sum(torch.multiply(tab_pred, tab_gt).flatten())
    denominator = torch.sum(tab_gt.flatten())
    return (1.0 * numerator) / denominator


def tab_f_measure(pred, gt):
    p = tab_precision(pred, gt)
    r = tab_recall(pred, gt)
    f = (2 * p * r) / (p + r)
    return f


# In[4]:


def tab_disamb(pred, gt):
    pp = pitch_precision(pred, gt)
    tp = tab_precision(pred, gt)
    return tp / pp

