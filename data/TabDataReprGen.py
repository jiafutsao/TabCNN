#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import jams
from scipy.io import wavfile
import sys
import librosa
import csv


# In[2]:


class TabDataReprGen:
    
    def __init__(self, mode="c"):
        # file path to GuitarSet dataset
        path = "GuitarSet/"
        self.path_audio = path + "audio/audio_mono-mic/"
        self.path_anno = path + "annotation/"
        
        # Labeling parameters
        self.string_midi_pitches = [40, 45, 50 ,55, 59, 64]
        self.highest_fret = 19
        self.num_classes = self.highest_fret + 2 # for open/closed
        
        # prepresentation and its labels storage
        self.output = {}
        
        # preprocessing modes
        # 
        # c = cqt
        # m = melspec
        # cm = cqt + melspec
        # s = stft
        #
        self.preproc_mode = mode
        self.downsample = True
        self.normalize = True
        self.sr_downs = 22050
        
        # CQT parameters
        self.cqt_n_bins = 192
        self.cqt_bins_per_octave = 24
        
        # STFT patameters
        self.n_fft = 2048
        self.hop_length = 512
        
        # save file path
        self.save_path = "spec_repr/" + self.preproc_mode + "/"
        
    def load_rep_and_labels_from_raw_file(self, filename):
        file_audio = self.path_audio + filename + "_mic.wav"
        file_anno = self.path_anno + filename + ".jams"
        jam = jams.load(file_anno)
        self.sr_original, data = wavfile.read(file_audio)
        self.sr_curr = self.sr_original
        
        # preprocess audio, store in output dict
        self.output["repr"] = np.swapaxes(self.preprocess_audio(data),0 , 1)# frames * 192
        
        # construct labels
        frame_indices = range(len(self.output["repr"]))
        times = librosa.frames_to_time(frame_indices, sr=self.sr_curr, hop_length=self.hop_length)
        
        # loop over all strings and sample annotations
        labels = []
        for string_num in range(6):
            anno = jam.annotations["note_midi"][string_num]
            string_label_samples = anno.to_samples(times)
            # replace midi pitch values with fret numbers
            for i in frame_indices:
                if string_label_samples[i] == []:
                    string_label_samples[i] = -1
                else:
                    string_label_samples[i] = int(round(string_label_samples[i][0]) - self.string_midi_pitches[string_num])
            labels.append([string_label_samples])
        
        labels = np.array(labels)
        # remove the extra dimension
        labels = np.squeeze(labels)
        labels = np.swapaxes(labels, 0, 1) # frames * 6 (fret position)
        
        # clean labels
        labels = self.clean_labels(labels) # frames * 6 * 21 (one-hot)
        
        # store and return
        self.output["labels"] = labels
        return len(labels)
    
    def correct_numbering(self, n):
        n += 1
        if n < 0 or n > self.highest_fret:
            n = 0
        return n
    
    def categorical(self, label):
        temp = np.zeros((len(label), self.num_classes), dtype=int)
        temp[np.arange(len(label)), label] = 1
        return temp
    
    def clean_label(self, label):
        label = [self.correct_numbering(n) for n in label]
        return self.categorical(label)
    
    def clean_labels(self, labels):
        return np.array([self.clean_label(label) for label in labels])
    
    def preprocess_audio(self, data):
        data = data.astype(float)
        if self.normalize:
            data = librosa.util.normalize(data)
        if self.downsample:
            data = librosa.resample(data, orig_sr=self.sr_original, target_sr=self.sr_downs)
            self.sr_curr = self.sr_downs
        if self.preproc_mode == "c":
            data = np.abs(librosa.cqt(data,
                                     hop_length=self.hop_length,
                                     sr=self.sr_curr,
                                     n_bins=self.cqt_n_bins,
                                     bins_per_octave=self.cqt_bins_per_octave))
        elif self.preproc_mode == "m":
            data = librosa.feature.melspectrogram(y=data, sr=self.sr_curr, n_fft=self.n_fft, hop_length=self.hop_length)
        elif self.preproc_mode == "cm":
            cqt = np.abs(librosa.cqt(data,
                                     hop_length=self.hop_length,
                                     sr=self.sr_curr,
                                     n_bins=self.cqt_n_bins,
                                     bins_per_octave=self.cqt_bins_per_octave))
            mel = librosa.feature.melspectrogram(y=data, sr=self.sr_curr, n_fft=self.n_fft, hop_length=self.hop_length)
            data = np.concatenate((cqt, mel), axis=0)
        elif self.preproc_mode == "s":
            data = np.abs(librosa.stft(data, n_fft=self.n_fft, hop_length=self.hop_length))
        else:
            print("Invalid Representation Mode")
            
        return data

    def save_data(self, filename):
        np.savez(filename, **self.output)
        
    def get_nth_filename(self, n):
        # returns the filename with no extension
        filenames = np.sort(np.array(os.listdir(self.path_anno)))
        filenames = list(filter(lambda x: x[-5:] == ".jams", filenames))
        return filenames[n][:-5] 
        
    def load_and_save_repr_nth_file(self, n):
        # filename has no extenstion
        filename = self.get_nth_filename(n)
        num_frames = self.load_rep_and_labels_from_raw_file(filename)
        print("done: " + filename + ", " + str(num_frames) + " frames")
        # write to csv
        frame_name = []
        for i in range(num_frames):
            frame_name.append([filename + "_" + str(i)])
        file = open('./spec_repr/id_' + self.preproc_mode + '.csv', 'a', newline ='')
        with file:   
            write = csv.writer(file)
            write.writerows(frame_name)
            
        save_path = self.save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.save_data(save_path + filename + ".npz")


# In[3]:


def main(args):
    n = args[0]
    m = args[1]
    gen = TabDataReprGen(mode=m)
    gen.load_and_save_repr_nth_file(n)
    
#if __name__ == "__main__":
#    main(args)


# In[ ]:





# In[ ]:





# In[4]:


#output = [["aaaaa16"], ["cccccc88"], ["ddddd77"]]
#file = open('test.csv', 'a', newline ='')
#with file:   
#    write = csv.writer(file)
#    write.writerows(output)


# In[ ]:





# In[ ]:





# In[ ]:





# In[5]:


#def preprocess_audio(data):
#    data = data.astype(float)
#    data = librosa.util.normalize(data)
#    data = librosa.resample(data, orig_sr=44100, target_sr=22050)
#    data = np.abs(librosa.cqt(data,
#        hop_length=512, 
#        sr=22050, 
#        n_bins=192, 
#        bins_per_octave=24))
#
#    return data


# In[6]:


#output = {}
#sr_original, data = wavfile.read("GuitarSet/audio/audio_mono-mic/00_BN1-129-Eb_solo_mic.wav")
#output["repr"] = np.swapaxes(preprocess_audio(data),0,1)
#print(len(output["repr"]))
#
#frame_indices = range(len(output["repr"]))
#times = librosa.frames_to_time(frame_indices, sr = 22050, hop_length=512)


# In[7]:


#jam = jams.load("GuitarSet/annotation/00_BN1-129-Eb_solo.jams")
#anno = jam.annotations["note_midi"][2]
#string_label_samples = anno.to_samples(times)


# In[8]:


#anno


# In[9]:


#times


# In[10]:


#string_label_samples

