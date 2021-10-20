#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 16:40:51 2018

@author: saulius
"""

import csv
import numpy as np
import time
import pyroomacoustics as pra
from scipy.signal import fftconvolve
from scipy.io import wavfile
from scipy import signal
import os
import datetime
import targetroomgauss
import librosa
import librosa.display
import datetime
from scipy import signal
import tensorflow as tf
import pickle
import os
import matplotlib.pyplot as plt
import scipy.io.wavfile as wf
import glob
import csv
import itertools
from collections import namedtuple

import numpy as np
import time
import pyroomacoustics as pra
from scipy.signal import fftconvolve
from scipy.io import wavfile
from scipy.interpolate import griddata
import os
import datetime
import librosa
import librosa.display
import datetime
from scipy import signal
import tensorflow as tf
import pickle
import os
from adjustText import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import axes3d
from matplotlib.widgets import Slider
from itertools import combinations

from matplotlib.animation import FuncAnimation
import matplotlib.colors as colors


from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import *
from keras.models import Sequential

# from keras.utils import plot_model
from keras import optimizers
import keras
import os

import soundfile as sf
from livelossplot.keras import PlotLossesCallback

import pyroomacoustics as pra

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib
from matplotlib.animation import FFMpegWriter

from datetime import datetime
import tables

def read_tab_file_old(filename,lines_start=None,lines_end=None):
    ''' Reads lines from a tab separated file
    '''
    with open(filename) as tsv:
#        with open ('data.txt', 'r') as f:
        if lines_start is None:
            print("none")
        else:
            if lines_end is None:
                head = [next(tsv) for x in range(lines_start)]
            else:
                head = [next(tsv) for x in range(lines_start,lines_end)]
            data = []
            for line in csv.reader(head, dialect="excel-tab"): #You can also use delimiter="\n" rather than giving a dialect.
                print(line)
#                data.append(line)
    return data


def read_tab_file_header(filename):
    ''' Reads first line from a tab separated file
    '''
    data = read_tab_file("position_source_loudspeaker1.txt",1)
    return data



def read_tab_file(filename,lines_start=None,lines_n=None):
    ''' Reads lines from a tab separated file
    '''
    with open(filename) as tsv:
#        with open ('data.txt', 'r') as f:
        if lines_start is None:
            print("from the begining")
            head = [next(tsv) for x in range(lines_n)]
        else:
            print("from ",lines_start,"-th line")
            for z in range(lines_start):
                next(tsv)
            head = [next(tsv) for x in range(lines_n)]
        data = []
        for line in csv.reader(head, dialect="excel-tab"): #You can also use delimiter="\t" rather than giving a dialect.
#            print(line)
            data.append(line)
    return data


#class PositionSample:
#     def __init__(self):
#        self.


def get_source_position_samples_to_arrays(data):
    year = []
    month = [] 
    day  = []
    hour  = []
    minute  = []
    second  = []
    x  = []
    y  = []
    z  = []
    ref_vec_x  = []
    ref_vec_y  = []
    ref_vec_z  = []
    rotation_11  = []
    rotation_12  = []
    rotation_13  = []
    rotation_21  = []
    rotation_22  = []
    rotation_23  = []
    rotation_31  = []
    rotation_32  = []
    rotation_33 = []
    
    data = [[float(float(j)) for j in i] for i in data]
    
    for k in range(len(data)):
        year.append(data[k][0])
        month.append(data[k][1]) 
        day.append(data[k][2])
        hour.append(data[k][3])
        minute.append(data[k][4])
        second.append(data[k][5])
        x.append(data[k][6])
        y.append(data[k][7])
        z.append(data[k][8])
        ref_vec_x.append(data[k][9])
        ref_vec_y.append(data[k][10])
        ref_vec_z.append(data[k][11])
        rotation_11.append(data[k][12])
        rotation_12.append(data[k][13])
        rotation_13.append(data[k][14])
        rotation_21.append(data[k][15])
        rotation_22.append(data[k][16])
        rotation_23.append(data[k][17])
        rotation_31.append(data[k][18])
        rotation_32.append(data[k][19])
        rotation_33.append(data[k][20])
        
    return year, month, day, hour, minute, second, x, y, z, ref_vec_x, ref_vec_y, ref_vec_z, rotation_11, rotation_12, rotation_13, rotation_21, rotation_22, rotation_23, rotation_31, rotation_32, rotation_33


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def get_audio_channel(array_audio_mch_data,ch):
    return array_audio_mch_data[:,ch]

def get_audio_slice(data,start_sample,samples_in_slice):
    audio_slice = data[start_sample:samples_in_slice,:] #  of all channels
    return audio_slice
#%%
def getITD(data,a,b,m,n):
    ''' returns ITD in samples between signals a and b of a multichannel audio sclices of length n starting from sample m 
    '''
    start_sample = m
    samples_in_slice = n
 
    # get a slice of audio
    audio_slice = data[start_sample:samples_in_slice,:] #  of all channels
    audio_slice_0 = audio_slice[:,a]
    audio_slice_1 = audio_slice[:,b]
    
    corr = signal.correlate(audio_slice_0,audio_slice_1, mode='same') / samples_in_slice
    corr_tau = np.arange(-corr.shape[0]/2,corr.shape[0]/2)    
    ITD = np.argmax(corr)+corr_tau[0]
    return ITD
#%%
# ILD #########################################################################
#%%
    
def getILD(data,a,b,m,n):
    '''
    Get ILD between channels a and b of an audio data slice from sample m to sample n
    '''
    start_sample = m
    samples_in_slice = n

    # get a slice of audio
    audio_slice = data[start_sample:samples_in_slice,:] #  of all channels
    audio_slice_0 = audio_slice[:,a]
    audio_slice_1 = audio_slice[:,b]
    
    energy_0 = np.sum(np.square(audio_slice_0))
    energy_1 = np.sum(np.square(audio_slice_1))
    
    ILD = 20*np.log10(energy_0/energy_1)
    return ILD

def print_ild_itd_for_all_pairs(array_audio_mch_data):
    N_CH = array_audio_mch_data.shape[1]
    for i,j in list(itertools.combinations(range(0, N_CH), r=2)):
        print("i: ",i,", j: ",j)
        
        print("ITD: ", getITD(array_audio_mch_data,i,j,0,1000))
        print("ILD: ", getILD(array_audio_mch_data,i,j,0,1000))
#%%
def get_ild_itd_for_all_pairs(array_audio_mch_data,start,length):
    '''
    Get ILD for all pairs in provided multichannel audio sample array.
    '''
    if isinstance(array_audio_mch_data, (list,)):
        data = np.asarray(array_audio_mch_data)
    elif isinstance(array_audio_mch_data, (np.ndarray, np.generic) ):
        data = array_audio_mch_data
    if data.shape[0] < data.shape[1]:
        data = np.transpose(data)
        
    N_CH = data.shape[1]
    print('data shape is ',data.shape)
    itd = []
    ild = []
    ii = []
    jj = []
    for i,j in list(itertools.combinations(range(0, N_CH), r=2)):
        print("i: ",i,", j: ",j)
        itd.append(getITD(data,i,j,start,length))
        ild.append(getILD(data,i,j,start,length))
        ii.append(i)
        jj.append(j)
        print("ITD: ", itd)
        print("ILD: ", ild)
    return ild, itd, ii, jj
#%%
## more pairs
#for i,j in list(itertools.combinations(range(0, N_CH), r=2)):
#    print("i: ",i,", j: ",j)
#    
#    print("ITD: ", getITD(array_audio_mch_data,i,j,0,1000))
#    print("ILD: ", getILD(array_audio_mch_data,i,j,0,1000))


def seconds_to_samples(seconds,fs):
    return  int(seconds*fs)

class PositionSample:
    '''used for training NN with pyroomacoustics'''
    
    def __init__(self, datapoint = None):
        self.datapoint = datapoint
        self.year = int(self.datapoint[0])
        self.month = int(self.datapoint[1]) 
        self.day = int(self.datapoint[2])
        self.hour = int(self.datapoint[3])
        self.minute = int(self.datapoint[4])
        self.second = int(self.datapoint[5])
        self.x = self.datapoint[6]
        self.y = self.datapoint[7]
        self.z = self.datapoint[8]
        self.ref_vec_x = self.datapoint[9]
        self.ref_vec_y = self.datapoint[10]
        self.ref_vec_z = self.datapoint[11]
        self.rotation_11 = self.datapoint[12]
        self.rotation_12 = self.datapoint[13]
        self.rotation_13 = self.datapoint[14]
        self.rotation_21 = self.datapoint[15]
        self.rotation_22 = self.datapoint[16]
        self.rotation_23 = self.datapoint[17]
        self.rotation_31 = self.datapoint[18]
        self.rotation_32 = self.datapoint[19]
        self.rotation_33 = self.datapoint[20]



class LOCATASoundSource:
    '''used for training NN with pyroomacoustics'''
    def __init__(self, data = None):
        self.year = []
        self.month = [] 
        self.day  = []
        self.hour  = []
        self.minute  = []
        self.second  = []
        self.x  = []
        self.y  = []
        self.z  = []
        self.ref_vec_x  = []
        self.ref_vec_y  = []
        self.ref_vec_z  = []
        self.rotation_11  = []
        self.rotation_12  = []
        self.rotation_13  = []
        self.rotation_21  = []
        self.rotation_22  = []
        self.rotation_23  = []
        self.rotation_31  = []
        self.rotation_32  = []
        self.rotation_33 = []
    
        self.data = [[float(float(j)) for j in i] for i in data]
    
        for k in range(len(data)):
            self.year.append(self.data[k][0])
            self.month.append(self.data[k][1]) 
            self.day.append(self.data[k][2])
            self.hour.append(self.data[k][3])
            self.minute.append(self.data[k][4])
            self.second.append(self.data[k][5])
            self.x.append(self.data[k][6])
            self.y.append(self.data[k][7])
            self.z.append(self.data[k][8])
            self.ref_vec_x.append(self.data[k][9])
            self.ref_vec_y.append(self.data[k][10])
            self.ref_vec_z.append(self.data[k][11])
            self.rotation_11.append(self.data[k][12])
            self.rotation_12.append(self.data[k][13])
            self.rotation_13.append(self.data[k][14])
            self.rotation_21.append(self.data[k][15])
            self.rotation_22.append(self.data[k][16])
            self.rotation_23.append(self.data[k][17])
            self.rotation_31.append(self.data[k][18])
            self.rotation_32.append(self.data[k][19])
            self.rotation_33.append(self.data[k][20])
        
        self.year = list(map(int, self.year))
        self.month = list(map(int, self.month))
        self.day = list(map(int, self.day))
        self.hour = list(map(int, self.hour))
        self.minute = list(map(int, self.minute))
        self.second = list(map(int, self.second))



def get_source_position_samples_to_objects(data):
    data = [[float(float(j)) for j in i] for i in data]
    
    sp_sample_list = []
    for k in data:
        sp_sample = PositionSample(k)    
        sp_sample_list.append(sp_sample)
        
    return sp_sample_list


def get_source_position_data_to_objects_old(source_idx,sources):
    flen = file_len(sources[source_idx])    
    data = read_tab_file(sources[source_idx],1,flen-1)
    sp_sample_list = get_source_position_samples_to_objects(data)
    return sp_sample_list

def get_source_position_data_to_objects(source_idx,sources):
    flen = file_len(sources[source_idx])    
    data = read_tab_file(sources[source_idx],1,flen-1)
    source_position_data = LOCATASoundSource(data)
    return source_position_data

def make_3d_scatter_fig():
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    return fig, ax

def plot_source_position_3d(source,start,length,red_ratio, fig=None, ax=None):
    
    X = np.asarray(source.x[start:length])
    Y = np.asarray(source.y[start:length])
    Z = np.asarray(source.z[start:length])
    X = X[::red_ratio]
    Y = Y[::red_ratio]
    Z = Z[::red_ratio]
    
    if fig is None:
        fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot(111,projection='3d')
    
    t = np.arange(X.shape[0])
    ax.scatter(X,Y,Z, c=t)
    plt.show()
    return
    
    
class LOCATAMicrophone:
    '''only works with eigenmike now'''
    def __init__(self, data = None):
        self.year = []
        self.month = []
        self.day = []
        self.hour = []
        self.minute = []
        self.second = []
        self.x = []
        self.y = []
        self.z = []
        self.ref_vec_x = []
        self.ref_vec_y = []
        self.ref_vec_z = []
        self.rotation_11 = []
        self.rotation_12 = []
        self.rotation_13 = []
        self.rotation_21 = []
        self.rotation_22 = []
        self.rotation_23 = []
        self.rotation_31 = []
        self.rotation_32 = []
        self.rotation_33 = []
        self.mic1_x = []
        self.mic1_y = []
        self.mic1_z = []
        self.mic2_x = []
        self.mic2_y = []
        self.mic2_z = []
        self.mic3_x = []
        self.mic3_y = []
        self.mic3_z = []
        self.mic4_x = []
        self.mic4_y = []
        self.mic4_z = []
        self.mic5_x = []
        self.mic5_y = []
        self.mic5_z = []
        self.mic6_x = []
        self.mic6_y = []
        self.mic6_z = []
        self.mic7_x = []
        self.mic7_y = []
        self.mic7_z = []
        self.mic8_x = []
        self.mic8_y = []
        self.mic8_z = []
        self.mic9_x = []
        self.mic9_y = []
        self.mic9_z = []
        self.mic10_x = []
        self.mic10_y = []
        self.mic10_z = []
        self.mic11_x = []
        self.mic11_y = []
        self.mic11_z = []
        self.mic12_x = []
        self.mic12_y = []
        self.mic12_z = []
        self.mic13_x = []
        self.mic13_y = []
        self.mic13_z = []
        self.mic14_x = []
        self.mic14_y = []
        self.mic14_z = []
        self.mic15_x = []
        self.mic15_y = []
        self.mic15_z = []
        self.mic16_x = []
        self.mic16_y = []
        self.mic16_z = []
        self.mic17_x = []
        self.mic17_y = []
        self.mic17_z = []
        self.mic18_x = []
        self.mic18_y = []
        self.mic18_z = []
        self.mic19_x = []
        self.mic19_y = []
        self.mic19_z = []
        self.mic20_x = []
        self.mic20_y = []
        self.mic20_z = []
        self.mic21_x = []
        self.mic21_y = []
        self.mic21_z = []
        self.mic22_x = []
        self.mic22_y = []
        self.mic22_z = []
        self.mic23_x = []
        self.mic23_y = []
        self.mic23_z = []
        self.mic24_x = []
        self.mic24_y = []
        self.mic24_z = []
        self.mic25_x = []
        self.mic25_y = []
        self.mic25_z = []
        self.mic26_x = []
        self.mic26_y = []
        self.mic26_z = []
        self.mic27_x = []
        self.mic27_y = []
        self.mic27_z = []
        self.mic28_x = []
        self.mic28_y = []
        self.mic28_z = []
        self.mic29_x = []
        self.mic29_y = []
        self.mic29_z = []
        self.mic30_x = []
        self.mic30_y = []
        self.mic30_z = []
        self.mic31_x = []
        self.mic31_y = []
        self.mic31_z = []
        self.mic32_x = []
        self.mic32_y = []
        self.mic32_z = []
        
    
        self.data = [[float(float(j)) for j in i] for i in data]
        
        self.mics = []
        for i in range(32):
            self.mics.append([])  
            
        for k in range(len(data)):
            self.year.append(self.data[k][0])
            self.month.append(self.data[k][1]) 
            self.day.append(self.data[k][2])
            self.hour.append(self.data[k][3])
            self.minute.append(self.data[k][4])
            self.second.append(self.data[k][5])
            self.x.append(self.data[k][6])
            self.y.append(self.data[k][7])
            self.z.append(self.data[k][8])
            self.ref_vec_x.append(self.data[k][9])
            self.ref_vec_y.append(self.data[k][10])
            self.ref_vec_z.append(self.data[k][11])
            self.rotation_11.append(self.data[k][12])
            self.rotation_12.append(self.data[k][13])
            self.rotation_13.append(self.data[k][14])
            self.rotation_21.append(self.data[k][15])
            self.rotation_22.append(self.data[k][16])
            self.rotation_23.append(self.data[k][17])
            self.rotation_31.append(self.data[k][18])
            self.rotation_32.append(self.data[k][19])
            self.rotation_33.append(self.data[k][20])
            
            for i in range(32):
                self.mics[i].append([self.data[k][20+i*3+1],
                                     self.data[k][20+i*3+2],
                                     self.data[k][20+i*3+3]])
                
            self.mic1_x.append(self.data[k][21])
            self.mic1_y.append(self.data[k][22])
            self.mic1_z.append(self.data[k][23])
            self.mic2_x.append(self.data[k][24])
            self.mic2_y.append(self.data[k][25])
            self.mic2_z.append(self.data[k][26])
            self.mic3_x.append(self.data[k][27])
            self.mic3_y.append(self.data[k][28])
            self.mic3_z.append(self.data[k][29])
            self.mic4_x.append(self.data[k][30])
            self.mic4_y.append(self.data[k][31])
            self.mic4_z.append(self.data[k][32])
            self.mic5_x.append(self.data[k][33])
            self.mic5_y.append(self.data[k][34])
            self.mic5_z.append(self.data[k][35])
            self.mic6_x.append(self.data[k][36])
            self.mic6_y.append(self.data[k][37])
            self.mic6_z.append(self.data[k][38])
            self.mic7_x.append(self.data[k][39])
            self.mic7_y.append(self.data[k][40])
            self.mic7_z.append(self.data[k][41])
            self.mic8_x.append(self.data[k][42])
            self.mic8_y.append(self.data[k][43])
            self.mic8_z.append(self.data[k][44])
            self.mic9_x.append(self.data[k][45])
            self.mic9_y.append(self.data[k][46])
            self.mic9_z.append(self.data[k][47])
            self.mic10_x.append(self.data[k][48])
            self.mic10_y.append(self.data[k][49])
            self.mic10_z.append(self.data[k][50])
            self.mic11_x.append(self.data[k][51])
            self.mic11_y.append(self.data[k][52])
            self.mic11_z.append(self.data[k][53])
            self.mic12_x.append(self.data[k][54])
            self.mic12_y.append(self.data[k][55])
            self.mic12_z.append(self.data[k][56])
            self.mic13_x.append(self.data[k][57])
            self.mic13_y.append(self.data[k][58])
            self.mic13_z.append(self.data[k][59])
            self.mic14_x.append(self.data[k][60])
            self.mic14_y.append(self.data[k][61])
            self.mic14_z.append(self.data[k][62])
            self.mic15_x.append(self.data[k][63])
            self.mic15_y.append(self.data[k][64])
            self.mic15_z.append(self.data[k][65])
            self.mic16_x.append(self.data[k][66])
            self.mic16_y.append(self.data[k][67])
            self.mic16_z.append(self.data[k][68])
            self.mic17_x.append(self.data[k][69])
            self.mic17_y.append(self.data[k][70])
            self.mic17_z.append(self.data[k][71])
            self.mic18_x.append(self.data[k][72])
            self.mic18_y.append(self.data[k][73])
            self.mic18_z.append(self.data[k][74])
            self.mic19_x.append(self.data[k][75])
            self.mic19_y.append(self.data[k][76])
            self.mic19_z.append(self.data[k][77])
            self.mic20_x.append(self.data[k][78])
            self.mic20_y.append(self.data[k][79])
            self.mic20_z.append(self.data[k][80])
            self.mic21_x.append(self.data[k][81])
            self.mic21_y.append(self.data[k][82])
            self.mic21_z.append(self.data[k][83])
            self.mic22_x.append(self.data[k][84])
            self.mic22_y.append(self.data[k][85])
            self.mic22_z.append(self.data[k][86])
            self.mic23_x.append(self.data[k][87])
            self.mic23_y.append(self.data[k][88])
            self.mic23_z.append(self.data[k][89])
            self.mic24_x.append(self.data[k][90])
            self.mic24_y.append(self.data[k][91])
            self.mic24_z.append(self.data[k][92])
            self.mic25_x.append(self.data[k][93])
            self.mic25_y.append(self.data[k][94])
            self.mic25_z.append(self.data[k][95])
            self.mic26_x.append(self.data[k][96])
            self.mic26_y.append(self.data[k][97])
            self.mic26_z.append(self.data[k][98])
            self.mic27_x.append(self.data[k][99])
            self.mic27_y.append(self.data[k][100])
            self.mic27_z.append(self.data[k][101])
            self.mic28_x.append(self.data[k][102])
            self.mic28_y.append(self.data[k][103])
            self.mic28_z.append(self.data[k][104])
            self.mic29_x.append(self.data[k][105])
            self.mic29_y.append(self.data[k][106])
            self.mic29_z.append(self.data[k][107])
            self.mic30_x.append(self.data[k][108])
            self.mic30_y.append(self.data[k][109])
            self.mic30_z.append(self.data[k][110])
            self.mic31_x.append(self.data[k][111])
            self.mic31_y.append(self.data[k][112])
            self.mic31_z.append(self.data[k][113])
            self.mic32_x.append(self.data[k][114])
            self.mic32_y.append(self.data[k][115])
            self.mic32_z.append(self.data[k][116])
        
        self.year = list(map(int, self.year))
        self.month = list(map(int, self.month))
        self.day = list(map(int, self.day))
        self.hour = list(map(int, self.hour))
        self.minute = list(map(int, self.minute))
        self.second = list(map(int, self.second))
    
        
def get_mic_positions(micposfile):
    flen = file_len(micposfile)    
    data = read_tab_file(micposfile,1,flen-1)
    mic_position_data = LOCATAMicrophone(data)
    return mic_position_data

def get_micposes_to_array(micposfile):
    flen = file_len(micposfile)    
    data = read_tab_file(micposfile,1,flen-1)
    mic_position_data = LOCATAMicrophone(data)
    mic_position_data_array = np.array(mic_position_data.mics)
    return mic_position_data_array

def get_micposarraycenter(micposarr):
    '''
    Returns an array of microphone array center, calculated as mean 
    coordinates of all the microphones in the array, for each timestep
    '''
    return micposarr[:,:].mean(axis=0)

def plot_scatter_3d(X,Y,Z,pointsize=100,color=None,aspect='equal'):
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    if color is None:
        ax.scatter(X,Y,Z,s=pointsize)
    else:
        ax.scatter(X,Y,Z,s=pointsize,c=color)
    ax.axis('equal')
    ax.autoscale(enable=True, axis='both', tight=True)
    ax.set_aspect(aspect, 'box')
    plt.show()
    return ax
    
def plot_scatter_3d_append(X,Y,Z,ax,pointsize=50):
    ax.scatter(X,Y,Z,s=pointsize)
    plt.show()
#%%
def get_quivers(TP,TC):
    quivers = []
    for src_idx in range(len(TP)): 
        
        MX = TC[:,0]
        MY = TC[:,1]
        MZ = TC[:,2]
        
        SX = TP[src_idx][:,0]
        SY = TP[src_idx][:,1]
        SZ = TP[src_idx][:,2]
        U = SX-MX
        V = SY-MY
        W = SZ-MZ
        
        quivers.append(np.vstack([MX, MY, MZ, U, V, W]).T)
    return quivers
#%%
def get_quiver_single_source(source,micposarrcenter):
    X = source[plot_slice_start:plot_slice_end,0]
    Y = source[plot_slice_start:plot_slice_end,1]
    Z = source[plot_slice_start:plot_slice_end,2]
    U = X-micposarrcenter[0]
    V = Y-micposarrcenter[1]
    W = Z-micposarrcenter[2]
    quivers = np.array([X,Y,Z,U,V,W]).T
    return quivers  
#%%
## FROM LocalizationTDOA
def locate(sensor_positions,timed):
    s = sensor_positions.shape
    len = s[0]
    time_delays = np.zeros((len, 1))
    time_delays = timed
    Amat = np.zeros((len, 1))
    Bmat = np.zeros((len, 1))
    Cmat = np.zeros((len, 1))
    Dmat = np.zeros((len, 1))
    
    for i in range(2, len):
        x1 = sensor_positions[0, 0]
        y1 = sensor_positions[0, 1]
        z1 = sensor_positions[0, 2]
        x2 = sensor_positions[1, 0]
        y2 = sensor_positions[1, 1]
        z2 = sensor_positions[1, 2]
        xi = sensor_positions[i, 0]
        yi = sensor_positions[i, 1]
        zi = sensor_positions[i, 2]
        
        Amat[i] = (1 / (340.29 * time_delays[i])) * (-2 * x1 + 2 * xi) - (1 / (340.29 * time_delays[1])) * (
            -2 * x1 + 2 * x2)
        Bmat[i] = (1 / (340.29 * time_delays[i])) * (-2 * y1 + 2 * yi) - (1 / (340.29 * time_delays[1])) * (
            -2 * y1 + 2 * y2)
        Cmat[i] = (1 / (340.29 * time_delays[i])) * (-2 * z1 + 2 * zi) - (1 / (340.29 * time_delays[1])) * (
            -2 * z1 + 2 * z2)
        Sum1 = (x1 ** 2) + (y1 ** 2) + (z1 ** 2) - (xi ** 2) - (yi ** 2) - (zi ** 2)
        Sum2 = (x1 ** 2) + (y1 ** 2) + (z1 ** 2) - (x2 ** 2) - (y2 ** 2) - (z2 ** 2)
        Dmat[i] = 340.29 * (time_delays[i] - time_delays[1]) + (1 / (340.29 * time_delays[i])) * Sum1 - (1 / (
            340.29 * time_delays[1])) * Sum2
    
    M = np.zeros((len + 1, 3))
    D = np.zeros((len + 1, 1))
    for i in range(len):
        M[i, 0] = Amat[i]
        M[i, 1] = Bmat[i]
        M[i, 2] = Cmat[i]
        D[i] = Dmat[i]
    
    M = np.array(M[2:len, :])
    D = np.array(D[2:len])
    
    D = np.multiply(-1, D)
    
    Minv = np.linalg.pinv(M)
    
    T = np.dot(Minv, D)
    x = T[0]
    y = T[1]
    z = T[2]
    
    return x, y, z
#%%
# as in https://math.stackexchange.com/questions/1722021/trilateration-using-tdoa
def tdoa_mat_A_row(lateration_mics,i,data,m,n,v):
    return np.array([ (lateration_mics[0][0]-lateration_mics[i][0]), (lateration_mics[0][1]-lateration_mics[i][1]), (lateration_mics[0][2]-lateration_mics[i][2]), getITD(data,0,i,m,n)*v])

def tdoa_mat_b_row(lateration_mics,i,data,m,n,v):
    xyz = np.asarray(lateration_mics)
    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]
    return np.array([ 1/2 * (x[0]**2 - x[i]**2 + y[0]**2 - y[i]**2 + z[0]**2 - z[i]**2 + getITD(data,0,i,m,n)**2)*v ])



def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=16):
    '''
    courtesy of https://github.com/xiongyihui/tdoa/blob/master/gcc_phat.py
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)method.
    '''
    
    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0] + refsig.shape[0]

    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)

    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))

    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))

    # find max cross correlation index
    shift = np.argmax(np.abs(cc)) - max_shift

    tau = shift / float(interp * fs)
    
    return tau, cc



def normalize(a, low=0, high=1):
    a_low = np.min(a)
    a_high = np.max(a)
    a_range = a_high - a_low
    
    a = a-a_low
    
    b_low = low
    b_high = high
    b_range = b_high - b_low
    
    b = a/a_range*b_range + b_low
    
    return b



def get_source_vad(source_idx,sources):
    flen = file_len(sources[source_idx])    
    data = read_tab_file(sources[source_idx],1,flen-1)
    source_vad_data = np.array(data)
    source_vad_data = [int(i) for i in source_vad_data]
    return np.array(source_vad_data)


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
def asSpherical(sxyz,mxyz):
    #takes list xyz (single coord)
    '''
    0, 0 corresponds to positive x axis
    '''
    sxyz = np.array(sxyz)
    mxyz = np.array(mxyz)
    x       = sxyz[:,0]-mxyz[:,0]
    y       = sxyz[:,1]-mxyz[:,1]
    z       = sxyz[:,2]-mxyz[:,2]
    r       =  np.sqrt(x*x + y*y + z*z)
    theta   =  np.arcsin(z/r)
    phi     =  np.arctan2(y,x)
    return np.array([r,phi,theta]).T

def get_DoAs(sxyz,mxyz):
    sxyz = np.array(sxyz)
    src_doas = []
    for i in range(sxyz.shape[0]):
        polarcoords = asSpherical(sxyz[i,:,:],mxyz)
        src_doas.append(np.array([np.rad2deg(polarcoords[:,1]),np.rad2deg(polarcoords[:,2])]).T)
    return src_doas

def generate_mxyz_for_each_sample(point_polar_coords,mic_arr_center):
    return np.tile(mic_arr_center, (point_polar_coords.shape[0], 1))   




### FUCNTION DEFINITIONS ###
def createTetrahedron(center, side):
    ''' center is 3D: x, y, z '''
    half_h = np.sin(60)/2*side
    ##    x                   y                   z           
    A = [ center[0] - side/2, center[1] - half_h, center[2] - half_h ]
    B = [ center[0]         , center[1] + half_h, center[2] - half_h]
    C = [ center[0] + side/2, center[1] - half_h, center[2] - half_h]
    D = [ center[0]         , center[1]         , center[2] + half_h]
    
    micpos = np.array([[A[0], B[0], C[0], D[0]],
                      [ A[1], B[1], C[1], D[1]],
                      [ A[2], B[2], C[2], D[2]] ])
    return micpos

def createSquare(center, side):
    ''' center is 3D: x, y, z '''
    
    backLeft   = [ center[0] - side/2, center[1] - side/2, center[2] ]
    backRight  = [ center[0] + side/2, center[1] - side/2, center[2] ]
    frontLeft  = [ center[0] - side/2, center[1] + side/2, center[2] ]
    frontRight = [ center[0] + side/2, center[1] + side/2, center[2] ]
    
    micpos = np.array([[backLeft[0], backRight[0], frontLeft[0], frontRight[0]],
                      [backLeft[1], backRight[1], frontLeft[1], frontRight[1]],
                      [backLeft[2], backRight[2], frontLeft[2], frontRight[2]]])
    return micpos

def create_room_audio(duration,NUM_SIGNALS,minsigduration,maxsigduration,maxsigdelay):
    pol = np.array([[0,0], [0,lengthy], [lengthx,lengthy], [lengthx,0]]).T
    room = pra.Room.from_corners(pol, fs=Fs, max_order=0, absorption=1.0)
    # Create the 3D room by extruding the 2D by height
    room.extrude(height)
        
    # Generate NUM_SIGNALS signals
    sig_freqs = []
    sig_amp = []
    sig_dur = []
    sig_del = []
    sig_pos = []
    sigg = []
    drysigs = []
    
    for i in range(0,NUM_SIGNALS):
#        print('duration: '+str(duration))
        sigdelay = np.floor((np.random.rand()*maxsigdelay)*100)/100 # in seconds
        sig_start = int(sigdelay*Fs) # in samples
#        print('sig_start: '+str(sig_start))    
        
        sigduration = np.floor((np.random.rand()*(maxsigduration-minsigduration)+minsigduration)*100)/100 # in seconds
        sig_dur_smpl = int(sigduration*Fs) # in samples
#        print('sig_dur_smpl: '+str(sig_dur_smpl))
        
        sig = np.zeros(duration*Fs) # empty signal (for the whole duration of the simulation)
        t = np.arange(0,sig_dur_smpl) # time vector (for the signal duration, whichi is random and different for each signal, and is shorter than the duration of the simulation)
        if not f_fixed:
            fsig = np.round((np.random.rand()*(fsigmax-fsigmin)+fsigmin))
        sig[sig_start:(sig_start+sig_dur_smpl)] = np.sin(t/Fs*2*np.pi*fsig) ### GENERATE THE FUCKING SIGNAL HERE ###
        sig = np.array(sig, dtype=float) # Convert signal (of list type) to float array
        sig = pra.normalize(sig) # Normalize signal (is this really needed?)
#        print('signal duration: '+str(len(sig)))
        
        sig_pos_x = round(np.random.rand()*lengthx)
        sig_pos_y = round(np.random.rand()*lengthy)
        sig_pos_z = round(np.random.rand()*height)
        
        sigg.append([fsig,sigduration,sigdelay,[sig_pos_x,sig_pos_y,sig_pos_z]])
        drysigs.append(sig)
        room.add_source([sig_pos_x,sig_pos_y,sig_pos_z],signal=sig)
            
#    R = createSquare(mic_array_center,mic_array_spread)
    R = createTetrahedron(mic_array_center,mic_array_spread)
    room.add_microphone_array(pra.MicrophoneArray(R, room.fs))
    
    room.simulate()
#    trg = targetroom.TargetRoom(room)
#    print('DOING STUFF...')
#    trg.do_stuff_2(STEP) # 0.01 is the step
#    room.plot()
#    plt.savefig('room.png')
#    plt.show()
    
    data = np.transpose(room.mic_array.signals)
    data = np.transpose(data)
    data = data[:,:Fs*duration] # data truncated to simulation duration in samples

    # Parse file (extract channels)
    data_ch = [] # empty list of data
    for i in range(data.shape[0]):
        data_ch.append(data[i,:])
        
  
    return data_ch, room, sigg, np.array(drysigs)


def create_room_rs(rs,Fs,duration,NUM_SIGNALS,lengthx,lengthy,height,order,absorption,mic_array_center,mic_array_spread):
    pol = np.array([[0,0], [0,lengthy], [lengthx,lengthy], [lengthx,0]]).T
    room = pra.Room.from_corners(pol, fs=Fs, max_order=order, absorption=absorption)
    # Create the 3D room by extruding the 2D by height
    room.extrude(height)
        
    # Generate NUM_SIGNALS signals
    sig_freqs = []
    sig_amp = []
    sig_dur = []
    sig_del = []
    sig_pos = []
    sigg = []
    drysigs = []
    
    rs_len = len(rs)
    duration_s = int(duration*Fs)
    
    
    for i in range(0,NUM_SIGNALS):
#       
        sig_start = int(np.floor(np.random.rand()*(rs_len-duration_s))) # in samples
        sig_end = sig_start+duration_s
        
        sig = rs[sig_start:sig_end] ### GENERATE THE FUCKING SIGNAL HERE ###
        sig = np.array(sig, dtype=float) # Convert signal (of list type) to float array
#        sig = pra.normalize(sig) # Normalize signal (is this really needed?)
#        print('signal duration: '+str(len(sig)))
        
        sig_pos_x = round(np.random.rand()*lengthx)
        sig_pos_y = round(np.random.rand()*lengthy)
        sig_pos_z = round(np.random.rand()*height)
        
        sigdelay = 0
        sigg.append([0,duration_s,0,[sig_pos_x,sig_pos_y,sig_pos_z]])
        drysigs.append(sig)
        room.add_source([sig_pos_x,sig_pos_y,sig_pos_z],signal=sig)
            
#    R = createSquare(mic_array_center,mic_array_spread)
    R = createTetrahedron(mic_array_center,mic_array_spread)
    room.add_microphone_array(pra.MicrophoneArray(R, room.fs))
    
    room.simulate()
#    trg = targetroom.TargetRoom(room)
#    print('DOING STUFF...')
#    trg.do_stuff_2(STEP) # 0.01 is the step
#    room.plot()
#    plt.savefig('room.png')
#    plt.show()
    
    data = np.transpose(room.mic_array.signals)
    data = np.transpose(data)
    data = data[:,:duration_s] # data truncated to simulation duration in samples
    
    # Parse file (extract channels)
    data_ch = [] # empty list of data
    for i in range(data.shape[0]):
        data_ch.append(data[i,:])
        
  
    return data_ch, room, sigg, np.array(drysigs)


def create_room_rs_noround(rs,Fs,duration,NUM_SIGNALS,lengthx,lengthy,height,order,absorption,mic_array_center,mic_array_spread):
    pol = np.array([[0,0], [0,lengthy], [lengthx,lengthy], [lengthx,0]]).T
    room = pra.Room.from_corners(pol, fs=Fs, max_order=order, absorption=absorption)
    # Create the 3D room by extruding the 2D by height
    room.extrude(height)
        
    # Generate NUM_SIGNALS signals
    sig_freqs = []
    sig_amp = []
    sig_dur = []
    sig_del = []
    sig_pos = []
    sigg = []
    drysigs = []
    
    rs_len = len(rs)
    duration_s = int(duration*Fs)
    
    
    for i in range(0,NUM_SIGNALS):
#       
        sig_start = int(np.floor(np.random.rand()*(rs_len-duration_s))) # in samples
        sig_end = sig_start+duration_s
        
        sig = rs[sig_start:sig_end] ### GENERATE THE FUCKING SIGNAL HERE ###
        sig = np.array(sig, dtype=float) # Convert signal (of list type) to float array
#        sig = pra.normalize(sig) # Normalize signal (is this really needed?)
#        print('signal duration: '+str(len(sig)))
        
        sig_pos_x = (np.random.rand()*lengthx)
        sig_pos_y = (np.random.rand()*lengthy)
        sig_pos_z = (np.random.rand()*height)
        
        sigdelay = 0
        sigg.append([0,duration_s,0,[sig_pos_x,sig_pos_y,sig_pos_z]])
        drysigs.append(sig)
        room.add_source([sig_pos_x,sig_pos_y,sig_pos_z],signal=sig)
            
#    R = createSquare(mic_array_center,mic_array_spread)
    R = createTetrahedron(mic_array_center,mic_array_spread)
    room.add_microphone_array(pra.MicrophoneArray(R, room.fs))
    
    room.simulate()
#    trg = targetroom.TargetRoom(room)
#    print('DOING STUFF...')
#    trg.do_stuff_2(STEP) # 0.01 is the step
#    room.plot()
#    plt.savefig('room.png')
#    plt.show()
    
    data = np.transpose(room.mic_array.signals)
    data = np.transpose(data)
    data = data[:,:duration_s] # data truncated to simulation duration in samples
    
    # Parse file (extract channels)
    data_ch = [] # empty list of data
    for i in range(data.shape[0]):
        data_ch.append(data[i,:])
        
  
    return data_ch, room, sigg, np.array(drysigs)

def create_room_rs_noround_irs(rs,Fs,duration,NUM_SIGNALS,lengthx,lengthy,height,order,absorption,mic_array_center,mic_array_spread):
    pol = np.array([[0,0], [0,lengthy], [lengthx,lengthy], [lengthx,0]]).T
    room = pra.Room.from_corners(pol, fs=Fs, max_order=order, absorption=absorption)
    # Create the 3D room by extruding the 2D by height
    room.extrude(height)
        
    # Generate NUM_SIGNALS signals
    sig_freqs = []
    sig_amp = []
    sig_dur = []
    sig_del = []
    sig_pos = []
    sigg = []
    drysigs = []
    
    rs_len = len(rs)
    duration_s = int(duration*Fs)
    
    
    for i in range(0,NUM_SIGNALS):
#       
        sig_start = int(np.floor(np.random.rand()*(rs_len-duration_s))) # in samples
        sig_end = sig_start+duration_s
        
        sig = rs[sig_start:sig_end] ### GENERATE THE FUCKING SIGNAL HERE ###
        sig = np.array(sig, dtype=float) # Convert signal (of list type) to float array
#        sig = pra.normalize(sig) # Normalize signal (is this really needed?)
#        print('signal duration: '+str(len(sig)))
        
        sig_pos_x = (np.random.rand()*lengthx)
        sig_pos_y = (np.random.rand()*lengthy)
        sig_pos_z = (np.random.rand()*height)
        
        sigdelay = 0
        sigg.append([0,duration_s,0,[sig_pos_x,sig_pos_y,sig_pos_z]])
        drysigs.append(sig)
        room.add_source([sig_pos_x,sig_pos_y,sig_pos_z],signal=sig)
            
#    R = createSquare(mic_array_center,mic_array_spread)
    R = createTetrahedron(mic_array_center,mic_array_spread)
    room.add_microphone_array(pra.MicrophoneArray(R, room.fs))
    
    room.add_source([mic_array_center[0], 
                 mic_array_center[1], 
                 mic_array_center[2]],signal=np.zeros(room.sources[0].signal.shape[0])) ## for IR of the center of the mic array

    room.simulate()
#    trg = targetroom.TargetRoom(room)
#    print('DOING STUFF...')
#    trg.do_stuff_2(STEP) # 0.01 is the step
#    room.plot()
#    plt.savefig('room.png')
#    plt.show()
    
    data = np.transpose(room.mic_array.signals)
    data = np.transpose(data)
    data = data[:,:duration_s] # data truncated to simulation duration in samples
    
    # Parse file (extract channels)
    data_ch = [] # empty list of data
    for i in range(data.shape[0]):
        data_ch.append(data[i,:])
        
  
    return data_ch, room, sigg, np.array(drysigs)


def create_room_rs_mpos_spos(rs,Fs,duration,lengthx,lengthy,height,order,absorbtion,mpos,spos):
    '''
    mpos, spos format: points x coordinates (4 mics x 3 coordinates etc.)
    '''
    
    pol = np.array([[0,0], [0,lengthy], [lengthx,lengthy], [lengthx,0]]).T
    room = pra.Room.from_corners(pol, fs=Fs, max_order=order, absorption=absorbtion)
    # Create the 3D room by extruding the 2D by height
    room.extrude(height)
        
    # Generate NUM_SIGNALS signals
    sig_freqs = []
    sig_amp = []
    sig_dur = []
    sig_del = []
    sig_pos = []
    sigg = []
    drysigs = []
    
    rs_len = len(rs)
    duration_s = int(duration*Fs)
    
    NUM_SIGNALS = spos.shape[0]
    
    
    for i in range(0,NUM_SIGNALS):
#       
        sig_start = int(np.floor(np.random.rand()*(rs_len-duration_s))) # in samples
        sig_end = sig_start+duration_s
        
        sig = rs[sig_start:sig_end] ### GENERATE THE FUCKING SIGNAL HERE ###
        sig = np.array(sig, dtype=float) # Convert signal (of list type) to float array
#        sig = pra.normalize(sig) # Normalize signal (is this really needed?)
#        print('signal duration: '+str(len(sig)))
        
        sig_pos_x = spos[i,0]
        sig_pos_y = spos[i,1]
        sig_pos_z = spos[i,2]
        
        sigdelay = 0
        sigg.append([0,duration_s,0,[sig_pos_x,sig_pos_y,sig_pos_z]])
        drysigs.append(sig)
        room.add_source([sig_pos_x,sig_pos_y,sig_pos_z],signal=sig)
            
#    R = createSquare(mic_array_center,mic_array_spread)
    R = mpos.T
    room.add_microphone_array(pra.MicrophoneArray(R, room.fs))
    
    room.simulate()
#    trg = targetroom.TargetRoom(room)
#    print('DOING STUFF...')
#    trg.do_stuff_2(STEP) # 0.01 is the step
#    room.plot()
#    plt.savefig('room.png')
#    plt.show()
    
    data = np.transpose(room.mic_array.signals)
    data = np.transpose(data)
    data = data[:,:duration_s] # data truncated to simulation duration in samples
    
    # Parse file (extract channels)
    data_ch = [] # empty list of data
    for i in range(data.shape[0]):
        data_ch.append(data[i,:])
        
  
    return data_ch, room, sigg, np.array(drysigs)

def create_room_rs_mpos_spos_2(rs,Fs,duration,limits,order,absorbtion,mpos,spos):
    '''
    mpos, spos format: points x coordinates (4 mics x 3 coordinates etc.)
    '''
    lengthx = limits[1,0]
    lengthy = limits[1,1]
    height  = limits[1,2]
    
    if spos.ndim < 2:
        spos = np.expand_dims(spos, axis=0)
    if mpos.ndim < 2:
        mpos = np.expand_dims(mpos, axis=0)
    
    pol = np.array([[0,0], [0,lengthy], [lengthx,lengthy], [lengthx,0]]).T
    room = pra.Room.from_corners(pol, fs=Fs, max_order=order, absorption=absorbtion)
    # Create the 3D room by extruding the 2D by height
    room.extrude(height)
        
    # Generate NUM_SIGNALS signals
    sig_freqs = []
    sig_amp = []
    sig_dur = []
    sig_del = []
    sig_pos = []
    sigg = []
    drysigs = []
    
    rs_len = len(rs)
    duration_s = int(duration*Fs)
    
    NUM_SIGNALS = spos.shape[0]
    
    
    for i in range(0,NUM_SIGNALS):
#       
        sig_start = int(np.floor(np.random.rand()*(rs_len-duration_s))) # in samples
        sig_end = sig_start+duration_s
        
        sig = rs[sig_start:sig_end] ### GENERATE THE FUCKING SIGNAL HERE ###
        sig = np.array(sig, dtype=float) # Convert signal (of list type) to float array
#        sig = pra.normalize(sig) # Normalize signal (is this really needed?)
#        print('signal duration: '+str(len(sig)))
        
        sig_pos_x = spos[i,0]
        sig_pos_y = spos[i,1]
        sig_pos_z = spos[i,2]
        
        sigdelay = 0
        sigg.append([0,duration_s,0,[sig_pos_x,sig_pos_y,sig_pos_z]])
        drysigs.append(sig)
        room.add_source([sig_pos_x,sig_pos_y,sig_pos_z],signal=sig)
            
#    R = createSquare(mic_array_center,mic_array_spread)
    R = mpos.T
    room.add_microphone_array(pra.MicrophoneArray(R, room.fs))
    
    room.simulate()
#    trg = targetroom.TargetRoom(room)
#    print('DOING STUFF...')
#    trg.do_stuff_2(STEP) # 0.01 is the step
#    room.plot()
#    plt.savefig('room.png')
#    plt.show()
    
    data = np.transpose(room.mic_array.signals)
    data = np.transpose(data)
    data = data[:,:duration_s] # data truncated to simulation duration in samples
    
    # Parse file (extract channels)
    data_ch = [] # empty list of data
    for i in range(data.shape[0]):
        data_ch.append(data[i,:])
        
  
    return data_ch, room, sigg, np.array(drysigs)


def get_random_n_spos(limits,n):
    lengthx = limits[1,0]
    lengthy = limits[1,1]
    height = limits[1,2]
    spos = []
    for i in range(n):
        spos.append([ 
                (np.random.rand()*(limits[1,0]-limits[0,0])+limits[0,0]),
                (np.random.rand()*(limits[1,1]-limits[0,1])+limits[0,1]),
                (np.random.rand()*(limits[1,2]-limits[0,2])+limits[0,2]) ])
    return np.array(spos)

def create_room_noise_mpos_spos(Fs,duration,lengthx,lengthy,height,order,absorbtion,mpos,spos):
    '''
    mpos, spos format: points x coordinates (4 mics x 3 coordinates etc.)
    '''
    
    pol = np.array([[0,0], [0,lengthy], [lengthx,lengthy], [lengthx,0]]).T
    room = pra.Room.from_corners(pol, fs=Fs, max_order=order, absorption=absorbtion)
    # Create the 3D room by extruding the 2D by height
    room.extrude(height)
        
    # Generate NUM_SIGNALS signals
    sig_freqs = []
    sig_amp = []
    sig_dur = []
    sig_del = []
    sig_pos = []
    sigg = []
    drysigs = []
    
    duration_s = int(duration*Fs) # duration_s is duration in samples
    
    NUM_SIGNALS = spos.shape[0]
    
    for i in range(0,NUM_SIGNALS):
        
        sig = np.random.random(duration_s)*2-1 ### GENERATE THE FUCKING NOISE SIGNAL HERE ###
        sig = np.array(sig, dtype=float) # Convert signal (of list type) to float array
#        sig = pra.normalize(sig) # Normalize signal (is this really needed?)
#        print('signal duration: '+str(len(sig)))
        
        sig_pos_x = spos[i,0]
        sig_pos_y = spos[i,1]
        sig_pos_z = spos[i,2]
        
        sigdelay = 0
        sigg.append([0,duration_s,0,[sig_pos_x,sig_pos_y,sig_pos_z]])
        drysigs.append(sig)
        room.add_source([sig_pos_x,sig_pos_y,sig_pos_z],signal=sig)
            
#    R = createSquare(mic_array_center,mic_array_spread)
    R = mpos.T
    room.add_microphone_array(pra.MicrophoneArray(R, room.fs))
    
    room.simulate()
#    trg = targetroom.TargetRoom(room)
#    print('DOING STUFF...')
#    trg.do_stuff_2(STEP) # 0.01 is the step
#    room.plot()
#    plt.savefig('room.png')
#    plt.show()
    
    data = np.transpose(room.mic_array.signals)
    data = np.transpose(data)
    data = data[:,:duration_s] # data truncated to simulation duration in samples
    
    # Parse file (extract channels)
    data_ch = [] # empty list of data
    for i in range(data.shape[0]):
        data_ch.append(data[i,:])
        
  
    return data_ch, room, sigg, np.array(drysigs)
#%%
def random_file(DATA_DIR):
    file = os.path.join(DATA_DIR, random.choice(os.listdir(DATA_DIR)));
    if os.path.isdir(file):
        return random_file(file)
    else:
        if file.endswith('.flac'):
            return file
        else:
            return random_file(DATA_DIR)
#%%
def create_room_libris_mpos_spos(DATA_DIR,Fs,duration,lengthx,lengthy,height,order,absorbtion,mpos,spos):
    '''
    mpos, spos format: points x coordinates (4 mics x 3 coordinates etc.)
    '''
    
    pol = np.array([[0,0], [0,lengthy], [lengthx,lengthy], [lengthx,0]]).T
    room = pra.Room.from_corners(pol, fs=Fs, max_order=order, absorption=absorbtion)
    # Create the 3D room by extruding the 2D by height
    room.extrude(height)
        
    # Generate NUM_SIGNALS signals
    sig_freqs = []
    sig_amp = []
    sig_dur = []
    sig_del = []
    sig_pos = []
    sigg = []
    drysigs = []
        
    NUM_SIGNALS = spos.shape[0]
    
    
    for i in range(0,NUM_SIGNALS):
#       
        rf = random_file(DATA_DIR)
        with open(rf, 'rb') as f:
            rfdata, rfsr = sf.read(f)
            
        
        ### GENERATE THE FUCKING SIGNAL HERE ###
        sig = rfdata # Convert signal (of list type) to float array
#        sig = pra.normalize(sig) # Normalize signal (is this really needed?)
#        print('signal duration: '+str(len(sig)))
        
        sig_pos_x = spos[i,0]
        sig_pos_y = spos[i,1]
        sig_pos_z = spos[i,2]
        
        sigdelay = 0
        sigg.append([0,0,0,[sig_pos_x,sig_pos_y,sig_pos_z]])
        drysigs.append(sig)
        room.add_source([sig_pos_x,sig_pos_y,sig_pos_z],signal=sig)
            
#    R = createSquare(mic_array_center,mic_array_spread)
    R = mpos.T
    room.add_microphone_array(pra.MicrophoneArray(R, room.fs))
    
    room.simulate()
#    trg = targetroom.TargetRoom(room)
#    print('DOING STUFF...')
#    trg.do_stuff_2(STEP) # 0.01 is the step
#    room.plot()
#    plt.savefig('room.png')
#    plt.show()
    
    data = np.transpose(room.mic_array.signals)
    data = np.transpose(data)
#    data = data[:,:duration_s] # data truncated to simulation duration in samples
    
    # Parse file (extract channels)
    data_ch = [] # empty list of data
    for i in range(data.shape[0]):
        data_ch.append(data[i,:])
        
  
    return data_ch, room, sigg, np.array(drysigs)

#%%
# def create_room_signal_mpos_spos(signal,Fs,duration,lengthx,lengthy,height,order,absorbtion,mpos,spos):
#     '''
#     mpos, spos format: points x coordinates (4 mics x 3 coordinates etc.)
#     signal: array of mono signals, [num_signals x num_audio_samples]
    
#     '''
    
#     pol = np.array([[0,0], [0,lengthy], [lengthx,lengthy], [lengthx,0]]).T
#     room = pra.Room.from_corners(pol, fs=Fs, max_order=order, absorption=absorbtion)
#     # Create the 3D room by extruding the 2D by height
#     room.extrude(height)
        
#     # Generate NUM_SIGNALS signals
#     sig_freqs = []
#     sig_amp = []
#     sig_dur = []
#     sig_del = []
#     sig_pos = []
#     sigg = []
#     drysigs = []
        
#     NUM_SIGNALS = spos.shape[0]
        
#     for i in range(0,NUM_SIGNALS):

        
#         sig = signal[i] # Convert signal (of list type) to float array
# #        sig = pra.normalize(sig) # Normalize signal (is this really needed?)
# #        print('signal duration: '+str(len(sig)))
        
#         sig_pos_x = spos[i,0]
#         sig_pos_y = spos[i,1]
#         sig_pos_z = spos[i,2]
        
#         sigdelay = 0
#         sigg.append([0,0,0,[sig_pos_x,sig_pos_y,sig_pos_z]])
#         drysigs.append(sig)
#         room.add_source([sig_pos_x,sig_pos_y,sig_pos_z],signal=sig)
            
# #    R = createSquare(mic_array_center,mic_array_spread)
#     R = mpos.T
#     room.add_microphone_array(pra.MicrophoneArray(R, room.fs))
    
#     room.simulate()
# #    trg = targetroom.TargetRoom(room)
# #    print('DOING STUFF...')
# #    trg.do_stuff_2(STEP) # 0.01 is the step
# #    room.plot()
# #    plt.savefig('room.png')
# #    plt.show()
    
#     data = np.transpose(room.mic_array.signals)
#     data = np.transpose(data)
# #    data = data[:,:duration_s] # data truncated to simulation duration in samples
    
#     # Parse file (extract channels)
#     data_ch = [] # empty list of data
#     for i in range(data.shape[0]):
#         data_ch.append(data[i,:])
        
  
#     return data_ch, room, sigg, np.array(drysigs)


#%%

def create_room_rs_time_mpos_spos(rs,Fs,time,lengthx,lengthy,height,order,absorbtion,mpos,spos):
    '''
    mpos, spos format: points x coordinates (4 mics x 3 coordinates etc.)
    '''
    
    pol = np.array([[0,0], [0,lengthy], [lengthx,lengthy], [lengthx,0]]).T
    room = pra.Room.from_corners(pol, fs=Fs, max_order=0, absorption=1.0)
    # Create the 3D room by extruding the 2D by height
    room.extrude(height)
        
    # Generate NUM_SIGNALS signals
    sig_freqs = []
    sig_amp = []
    sig_dur = []
    sig_del = []
    sig_pos = []
    sigg = []
    drysigs = []
    
    rs_len = len(rs)
    
    
    NUM_SIGNALS = spos.shape[0]
    
    
    for i in range(0,NUM_SIGNALS):
#       
        duration_s = int(time[i,1]-time[i,0])
        sig_start = time[i,0] # in samples
        sig_end = time[i,1]
        
        sig = rs[sig_start:sig_end] ### GENERATE THE FUCKING SIGNAL HERE ###
        sig = np.array(sig, dtype=float) # Convert signal (of list type) to float array
#        sig = pra.normalize(sig) # Normalize signal (is this really needed?)
#        print('signal duration: '+str(len(sig)))
        
        sig_pos_x = spos[i,0]
        sig_pos_y = spos[i,1]
        sig_pos_z = spos[i,2]
        
        sigdelay = 0
        sigg.append([0,duration_s,0,[sig_pos_x,sig_pos_y,sig_pos_z]])
        drysigs.append(sig)
        room.add_source([sig_pos_x,sig_pos_y,sig_pos_z],signal=sig)
            
#    R = createSquare(mic_array_center,mic_array_spread)
    R = mpos.T
    room.add_microphone_array(pra.MicrophoneArray(R, room.fs))
    
    room.simulate()
#    trg = targetroom.TargetRoom(room)
#    print('DOING STUFF...')
#    trg.do_stuff_2(STEP) # 0.01 is the step
#    room.plot()
#    plt.savefig('room.png')
#    plt.show()
    
    data = np.transpose(room.mic_array.signals)
    data = np.transpose(data)
    data = data[:,:duration_s] # data truncated to simulation duration in samples
    
    # Parse file (extract channels)
    data_ch = [] # empty list of data
    for i in range(data.shape[0]):
        data_ch.append(data[i,:])
        
  
    return data_ch, room, sigg, np.array(drysigs)

def create_room_rs_grid(rs,Fs,duration,NUM_SIGNALS,lengthx,lengthy,height,mic_array_center,mic_array_spread):
    pol = np.array([[0,0], [0,lengthy], [lengthx,lengthy], [lengthx,0]]).T
    room = pra.Room.from_corners(pol, fs=Fs, max_order=0, absorption=1.0)
    # Create the 3D room by extruding the 2D by height
    room.extrude(height)
        
    # Generate NUM_SIGNALS signals
    sig_freqs = []
    sig_amp = []
    sig_dur = []
    sig_del = []
    sig_pos = []
    sigg = []
    drysigs = []
    
    rs_len = len(rs)
    duration_s = int(duration*Fs)
    
    def myround(x, base=2):
        return base * round(x/base)
    
    for i in range(0,NUM_SIGNALS):
#       
        sig_start = int(np.floor(np.random.rand()*(rs_len-duration_s))) # in samples
        sig_end = sig_start+duration_s
        
        sig = rs[sig_start:sig_end] ### GENERATE THE FUCKING SIGNAL HERE ###
        sig = np.array(sig, dtype=float) # Convert signal (of list type) to float array
#        sig = pra.normalize(sig) # Normalize signal (is this really needed?)
#        print('signal duration: '+str(len(sig)))
        
        sig_pos_x = myround(np.random.rand()*lengthx)
        sig_pos_y = myround(np.random.rand()*lengthy)
        sig_pos_z = myround(np.random.rand()*height)
        
        sigdelay = 0
        sigg.append([0,duration_s,0,[sig_pos_x,sig_pos_y,sig_pos_z]])
        drysigs.append(sig)
        room.add_source([sig_pos_x,sig_pos_y,sig_pos_z],signal=sig)
            
#    R = createSquare(mic_array_center,mic_array_spread)
    R = createTetrahedron(mic_array_center,mic_array_spread)
    room.add_microphone_array(pra.MicrophoneArray(R, room.fs))
    
    room.simulate()
#    trg = targetroom.TargetRoom(room)
#    print('DOING STUFF...')
#    trg.do_stuff_2(STEP) # 0.01 is the step
#    room.plot()
#    plt.savefig('room.png')
#    plt.show()
    
    data = np.transpose(room.mic_array.signals)
    data = np.transpose(data)
    data = data[:,:duration_s] # data truncated to simulation duration in samples
    
    # Parse file (extract channels)
    data_ch = [] # empty list of data
    for i in range(data.shape[0]):
        data_ch.append(data[i,:])
        
  
    return data_ch, room, sigg, np.array(drysigs)

#room_rs, room, sigg, drysigs = create_room_rs(rs,duration=duration,
#                                         NUM_SIGNALS=NUM_SIGNALS)


def create_room_rs_test(rs,Fs,duration,NUM_SIGNALS,lengthx,lengthy,height,mic_array_center,mic_array_spread):
    pol = np.array([[0,0], [0,lengthy], [lengthx,lengthy], [lengthx,0]]).T
    room = pra.Room.from_corners(pol, fs=Fs, max_order=0, absorption=1.0)
    # Create the 3D room by extruding the 2D by height
    room.extrude(height)
        
    # Generate NUM_SIGNALS signals
    sig_freqs = []
    sig_amp = []
    sig_dur = []
    sig_del = []
    sig_pos = []
    sigg = []
    drysigs = []
    
    rs_len = len(rs)
    duration_s = int(duration*Fs)
    
    
    for i in range(0,NUM_SIGNALS):
#       
        sig_start = int(np.floor(np.random.rand()*(rs_len-duration_s))) # in samples
        sig_end = sig_start+duration_s
        
        sig = rs[sig_start:sig_end] ### GENERATE THE FUCKING SIGNAL HERE ###
        sig = np.array(sig, dtype=float) # Convert signal (of list type) to float array
#        sig = pra.normalize(sig) # Normalize signal (is this really needed?)
#        print('signal duration: '+str(len(sig)))
        
        sig_pos_x = round(np.random.rand()*lengthx)
        sig_pos_y = round(np.random.rand()*lengthy)
        sig_pos_z = round(np.random.rand()*height)
        
        sigdelay = 0
        sigg.append([0,duration_s,0,[sig_pos_x,sig_pos_y,sig_pos_z]])
        drysigs.append(sig)
        room.add_source([sig_pos_x,sig_pos_y,sig_pos_z],signal=sig)
            
#    R = createSquare(mic_array_center,mic_array_spread)
    R = createTetrahedron(mic_array_center,mic_array_spread)
    room.add_microphone_array(pra.MicrophoneArray(R, room.fs))
    
    room.simulate()
#    trg = targetroom.TargetRoom(room)
#    print('DOING STUFF...')
#    trg.do_stuff_2(STEP) # 0.01 is the step
#    room.plot()
#    plt.savefig('room.png')
#    plt.show()
    
    data = np.transpose(room.mic_array.signals)
    data = np.transpose(data)
    data = data[:,:duration_s] # data truncated to simulation duration in samples
    
    # Parse file (extract channels)
    data_ch = [] # empty list of data
    for i in range(data.shape[0]):
        data_ch.append(data[i,:])
        
  
    return data_ch, room, sigg, np.array(drysigs)

    
#%%    

def plot_audio_data(data_ch):
    plt.figure()
    num_ch = len(data_ch)
    ax1 = plt.subplot(num_ch,1,1)
    for i in range(num_ch):
        plt.subplot(num_ch,1,i+1,sharex=ax1)
        plt.plot(data_ch[i])


def mel_filter_freqs(sample_rate,nfilt):
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
#    bin = np.floor((NFFT + 1) * hz_points / sample_rate)
    hz_points[0] = 1
    hz_points[-1] = hz_points[-1]-1
    return hz_points


def split_sig_to_FBs_2(sig, Fs, hz_points, order=5):
    def butter_bandpass_filterbank(hz_points, fs, order=5):
        def butter_bandpass(lowcut, highcut, fs, order=5):
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            sos = signal.butter(order, [low, high], btype='band', output='sos')
            return sos
        filterbank_sos = []
        for i in range(1,len(hz_points)-1):
            sos = butter_bandpass(hz_points[i-1], hz_points[i+1], 
                                  fs, order=order)
            filterbank_sos.append(sos)
        return np.array(filterbank_sos)
        
    filterbank_sos = butter_bandpass_filterbank(hz_points, Fs, order)
    
    filtered = []
    for i in range(len(filterbank_sos)):
        filtered.append(signal.sosfilt(filterbank_sos[i], sig))
    return np.array(filtered), filterbank_sos


def generate_coords(limits,sets_n):
    '''
    limits format: lower xyz, upper xyz
    '''
    limits = np.array(limits)
    zxc = np.random.rand(sets_n,3)
    low = limits[0,:]
    high = limits[1,:]
    zxc[:,0] *= high[0]+low[0]
    zxc[:,1] *= high[1]+low[1]
    zxc[:,2] *= high[2]+low[2]
    zxc[:,0] += low[0]
    zxc[:,1] += low[1]
    zxc[:,2] += low[2]
    return zxc

#def createTetrahedron(center, side):
#    ''' center is 3D: x, y, z '''
#    half_h = np.sin(60)/2*side
#    ##    x                   y                   z           
#    A = [ center[0] - side/2, center[1] - half_h, center[2] - half_h ]
#    B = [ center[0]         , center[1] + half_h, center[2] - half_h]
#    C = [ center[0] + side/2, center[1] - half_h, center[2] - half_h]
#    D = [ center[0]         , center[1]         , center[2] + half_h]
#    
#    micpos = np.array([[A[0], B[0], C[0], D[0]],
#                      [ A[1], B[1], C[1], D[1]],
#                      [ A[2], B[2], C[2], D[2]] ])
#    micpos = micpos.T
#    return micpos
#room_audio, room, sigg, drysigs = create_room_audio(duration=duration,
#                                         NUM_SIGNALS=NUM_SIGNALS,
#                                         minsigduration=duration/6,
#                                         maxsigduration=duration/3,
#                                         maxsigdelay=(duration-maxsigduration))



def generate_ITD(mics,point_coords):
    v_s = 340 # speed of sound, m/s
    itds = []
    for i in range(point_coords.shape[0]):
        itd_source = []
        for j in range(mics.shape[0]):
            for k in range(mics.shape[0]):
                d1 = np.linalg.norm(point_coords[i]-mics[j])
                d2 = np.linalg.norm(point_coords[i]-mics[k])
                itd_source.append(np.linalg.norm(d1-d2)*v_s)
        itds.append(itd_source)
    itds = np.array(itds)
    return itds

def asSpherical(sxyz,mxyz):
    #takes list xyz (single coord)
    '''
    0, 0 corresponds to positive x axis
    '''
    x       = sxyz[:,0]-mxyz[:,0]
    y       = sxyz[:,1]-mxyz[:,1]
    z       = sxyz[:,2]-mxyz[:,2]
    r       =  np.sqrt(x*x + y*y + z*z)
    theta   =  np.arcsin(z/r)
    phi     =  np.arctan2(y,x)
    return np.array([r,phi,theta]).T

def get_DoAs2(sxyz,mxyz):
    polarcoords = asSpherical(sxyz,mxyz)
    return np.array([np.rad2deg(polarcoords[:,1]),np.rad2deg(polarcoords[:,2])]).T

def get_marray_center(mpos):
    mic_array_center = np.mean(mpos,axis=0)
    return mic_array_center
#%%
def get_DoAs_spos_mpos(spos,mpos):
    MAC = get_marray_center(mpos)
    if spos.ndim < 2:
        spos = np.expand_dims(spos,axis=0)
    x       = spos[:,0]-MAC[0]
    y       = spos[:,1]-MAC[1]
    z       = spos[:,2]-MAC[2]
    r       =  np.sqrt(x*x + y*y + z*z)
    theta   =  np.arcsin(z/r)
    phi     =  np.arctan2(y,x)
    polarcoords = np.array([r,phi,theta]).T
    return np.array([np.rad2deg(polarcoords[:,1]),np.rad2deg(polarcoords[:,2])]).T
#%%
def generate_mxyz_for_each_sample(point_polar_coords,mic_arr_center):
    return np.tile(mic_arr_center, (point_polar_coords.shape[0], 1))


def kernel3D(A,x,y,z,sigma,dx,dy,dz):
    return A*np.exp(-((x-dx)**2 + (y-dy)**2 + (z-dz)**2)/(2*sigma**2))

def kernel1D(A,x,sigma,dx):
    y = A*np.exp(-((x-dx)**2/(2*sigma**2)))
    return np.array(y)

def kernel2D(A,x,y,sigma,dx,dy):
    return A*np.exp(-((x-dx)**2 + (y-dy)**2)/(2*sigma**2))

def generate_field_DoA(doa_resolution):
    x = np.arange(doa_resolution[0,0],doa_resolution[0,1],1)   # coordinate arrays -- make sure they contain 0!
    y = np.arange(doa_resolution[1,0],doa_resolution[1,1],1)
    xx, yy = np.meshgrid(x,y)
    field = np.zeros([len(y),len(x)])
    return xx, yy, field

def populate_field_DoA(xx,yy,field,point_doas,sigma,doa_resolution):
    point_doas[:,0] = np.multiply(np.divide(point_doas[:,0],180),doa_resolution[0,1])
    point_doas[:,1] = np.multiply(np.divide(point_doas[:,1],90),doa_resolution[1,1])
    point_doas = point_doas.tolist()
    for point in point_doas:
#        print(point)
        field += kernel2D(1,xx,yy,sigma,point[0],point[1])
    return field

def populate_field_DoA_amp(xx,yy,field,point_doas,sigma,doa_resolution,amps):
    point_doas[:,0] = np.multiply(np.divide(point_doas[:,0],180),doa_resolution[0,1])
    point_doas[:,1] = np.multiply(np.divide(point_doas[:,1],90),doa_resolution[1,1])
    for p in range(point_doas.shape[0]):
#        print(point)
        field += kernel2D(1,xx,yy,sigma,point_doas[p,0],point_doas[p,1])*amps[p]
    return field


def gen_set(limits,npoints,mic_arr_center, doa_resolution,sigma):
    coords = generate_coords(limits,npoints)
    mxyz = generate_mxyz_for_each_sample(coords, mic_arr_center)
    doas = get_DoAs2(coords,mxyz)
    xx, yy, field = generate_field_DoA(doa_resolution)
    field = populate_field_DoA(xx, yy, field, doas, sigma)
    #plot_volume(field)
    return coords, doas, field

def gen_set_itd(limits,npoints,mic_arr_center, doa_resolution,sigma):
    coords, doas, field = gen_set(limits,npoints,mic_arr_center, doa_resolution,sigma)
    ITDS = generate_ITD(mics,coords)
    return coords, doas, field, ITDS

def extents(f):
    '''
    for imshow ticks
    '''
    delta = f[1] - f[0]
    return [f[0] - delta/2, f[-1] + delta/2]

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def get_quivers_doa_test(TP,TC):
    quivers = []
        
    MX = TC[:,0]
    MY = TC[:,1]
    MZ = TC[:,2]
    
    SX = TP[:,0]
    SY = TP[:,1]
    SZ = TP[:,2]
    U = SX-MX
    V = SY-MY
    W = SZ-MZ
    
    quivers = np.vstack([MX, MY, MZ, U, V, W]).T
    return quivers


#% Show 3D points map
def plot_3D_points(coords,mics,mic_arr_center):
    mxyz = generate_mxyz_for_each_sample(coords, mic_arr_center)
    quivers = get_quivers_doa_test(coords,mxyz)
    X,Y,Z = coords[:,0],coords[:,1],coords[:,2]
    fig = plt.figure()
    ax_room = fig.gca(projection='3d')
    ax_room.set_aspect('equal')
    plot_scatter_3d_append(coords[:,0],coords[:,1],coords[:,2],ax_room)
    set_axes_equal(ax_room)
    plot_scatter_3d_append(mics[:,0],mics[:,1],mics[:,2],ax_room)
    plot_scatter_3d_append(mic_arr_center[0],mic_arr_center[1],mic_arr_center[2],ax_room)
    ax_room.quiver(quivers[:,0],
                   quivers[:,1],
                   quivers[:,2],
                   quivers[:,3],
                   quivers[:,4],
                   quivers[:,5],length=1)
    ax_room.quiver(mic_arr_center[0],mic_arr_center[1],mic_arr_center[2],limits[1,0]-mic_arr_center[0],0,0,color='red',linewidth=3)
    
    
    ax_room.set_xlabel('x',size=14)
    ax_room.set_ylabel('y',size=14)
    ax_room.set_zlabel('z',size=14)
    #ax_room.set_xlim(limits[0,0],limits[1,0])
    #ax_room.set_ylim(limits[0,1],limits[1,1])
    #ax_room.set_zlim(limits[0,2],limits[1,2])
    texts3d = []
    for i in range(len(coords)): #plot each point + it's index as text above
        text3d = ax_room.text(coords[i,0], coords[i,1], coords[i,2], np.int_(coords[i]),ha="center", va="top", color="r")
        texts3d.append(text3d)
        text3d = ax_room.text(coords[i,0], coords[i,1], coords[i,2], np.int_(doas[i]),ha="center", va="bottom", color="magenta")
        texts3d.append(text3d)
    adjust_text(texts3d, autoalign='y', only_move={'text':'y'}, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))


#% Show DoA map
def plot_DoA_map(field,coords,doas):
    fig, ax = plt.subplots()
    im = ax.imshow(field, aspect='auto', interpolation='none',
               extent=[doa_resolution[0,0],doa_resolution[0,1],doa_resolution[1,0],doa_resolution[1,1]], origin='lower')
    texts = []
    for i in range(len(coords)):
        text = ax.text(doas[i,0], doas[i,1], np.int_(coords[i]),ha="center", va="top", color="r")
        texts.append(text)
        text = ax.text(doas[i,0], doas[i,1], np.int_(doas[i]),ha="center", va="bottom", color="magenta")
        texts.append(text)
    adjust_text(texts, autoalign='y', only_move={'text':'y'})
    ax.set_xlabel(r'Azimuth ($ \phi $), degrees',size=14)
    ax.set_ylabel(r'Elevation ($ \theta $), degrees',size=14)
    plt.show()

#%% Training data function definition

def generate_xcorrs_target_data():
    room_audio, room, sigg, drysigs = create_room_audio(duration=duration,
                                         NUM_SIGNALS=NUM_SIGNALS,
                                         minsigduration=duration/6,
                                         maxsigduration=duration/3,
                                         maxsigdelay=(duration-maxsigduration))

    hz_points = mel_filter_freqs(Fs,nfilt)
    filteredaudio, filterbank_sos = split_sig_to_FBs_2(room_audio[0],hz_points, 8)
    
    frame_length = 2**flpow;
    frame_overlap = frame_length/2;
    
    frame_step = frame_length-frame_overlap;
    frames_N = int((room_audio[0].shape[0]-frame_length)/frame_step);
    frames = np.arange(0,frames_N);
    frame_starts = np.arange(0,frames_N*frame_step,frame_step);
    frame_ends = frame_starts+frame_length-1;
    
    filtered_audio_mics = []
    for mic in range(len(room_audio)):
        filteredaudio, filterbank_sos = split_sig_to_FBs_2(room_audio[mic],hz_points, 8)
        filtered_audio_mics.append(filteredaudio)
    
    xcorrs = []
    #% extract coordinates from simulated sigg
    sig_coords = []
    for s in sigg:
        sig_coords.append(s[3])
    sig_coords = np.array(sig_coords)   
       
    mxyz = generate_mxyz_for_each_sample(sig_coords, mic_arr_center)
    doas = get_DoAs2(sig_coords,mxyz)
    xx, yy, field = generate_field_DoA(doa_resolution)
    field = populate_field_DoA(xx, yy, field, doas, sigma)
    quivers = get_quivers_doa_test(sig_coords,mxyz)
    
#    ITDS = generate_ITD(mics,sig_coords) # <-- this we find from xcorr

    ################################################################### FRAMES ####
    target = []
    amps = []
    xcorrs = []
    xcorrsflat = []
    for frame in frames:
        frame_start = int(frame_starts[frame])
        frame_end   = int(frame_ends  [frame])
    #    y1 = interp(squeeze(A_pair_filtered(frame_starts(frame):frame_ends(frame),1)),oversmp);
    #    y2 = interp(squeeze(A_pair_filtered(frame_starts(frame):frame_ends(frame),2)),oversmp);
        
        framexc = []
        for band in range(nfilt):
            y1 = filtered_audio_mics[pair_mic_1][band,frame_start:frame_end]
            y2 = filtered_audio_mics[pair_mic_2][band,frame_start:frame_end]
            fbxcorr = signal.correlate(y1,y2)
            framexc.append(fbxcorr)
        xcorrs.append(framexc)
        xcorrsflat.append(np.array(framexc).flatten())
        
        amps.append([])
        xx, yy, field = generate_field_DoA(doa_resolution)
        for ss in range(NUM_SIGNALS):
            amp = np.sum((drysigs[ss,frame_start:frame_end])**2)/frame_length
            amps[frame].append(amp)
            field += kernel2D(1,xx,yy,sigma,doas[ss,0],doas[ss,1])*amp
        
        target.append(field)
        
    xcorrs = np.array(xcorrs)
    xcorrsflat = np.array(xcorrsflat)
    target = np.array(target)
    return xcorrs, xcorrsflat, target

import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

gmail_user = 'strutistrutis@gmail.com'  
gmail_password = 'strutistrutistrutis'

sent_from = gmail_user  
to = ['strutistrutis@gmail.com']  
subject = 'OMG Super Important Message'  
body = 'DONE!'

email_text = """\  
From: %s  
To: %s  
Subject: %s

%s
""" % (sent_from, ", ".join(to), subject, body)

     
        
def SendMail(ImgFileName,mailtext):
    img_data = open(ImgFileName, 'rb').read()
    msg = MIMEMultipart()
    msg['Subject'] = mailtext
    msg['From'] = gmail_user
    msg['To'] = gmail_user

    text = MIMEText(mailtext)
    msg.attach(text)
    image = MIMEImage(img_data, name=os.path.basename(ImgFileName))
    msg.attach(image)
    
    msg.attach(MIMEText(open("model_summary.txt").read()))
    
    attachment = MIMEText(json.dumps(model.get_config()))
    attachment.add_header('Content-Disposition', 'attachment', 
                          filename="model_config.json")
    msg.attach(attachment)
    
    
    try: 
        s = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        s.ehlo()
#        s.starttls()
        s.ehlo()
        s.login(gmail_user, gmail_password)
        s.sendmail(sent_from, to, msg.as_string())
        s.close()
        print('Email sent!')
    except:  
        print('Something went wrong...')     
        
        
class DynamicUpdate():
    #Suppose we know the x range
    min_x = 0
    max_x = 10

    def on_launch(self,log=False):
        #Set up plot
        self.figure, self.ax = plt.subplots()
        if log==False:
            self.lines, = self.ax.plot([],[], 'o')
            self.lines2, = self.ax.plot([],[], 'x')
        else:
            self.lines, = self.ax.semilogy([],[], 'o')
            self.lines2, = self.ax.semilogy([],[], 'x')
        #Autoscale on unknown axis and known lims on the other
        self.ax.set_autoscale_on(True)
#        self.ax.set_xlim(self.min_x, self.max_x)
        #Other stuff
        self.ax.grid()
        ...

    def on_running(self, xdata, ydata, valdata, cut=0):
        #Update data (with the new _and_ the old points)
        if (len(xdata) < cut) or cut==0:
           self.lines.set_xdata(xdata)
           self.lines.set_ydata(ydata)
           self.lines2.set_xdata(xdata)
           self.lines2.set_ydata(valdata)
        else:
           self.lines.set_xdata(xdata[-50:])
           self.lines.set_ydata(ydata[-50:])
           self.lines2.set_xdata(xdata[-50:])
           self.lines2.set_ydata(valdata[-50:])
        #Need both of these in order to rescale
        self.ax.relim()
        self.ax.autoscale_view()
        #We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

#%%
def generate_xcorrs_target_data_rs(rs,Fs,duration,nfilt,doa_resolution):
    # create room audio from real speech at randomly selected coordinates
    room_audio, room, sigg, drysigs = create_room_rs(rs,Fs,duration,NUM_SIGNALS,
                                                     lengthx,lengthy,height,
                                                     mic_array_center,mic_array_spread)
    
    # calculate sample values for frames
    frame_length = 2**flpow;
    frame_overlap = frame_length/2;
    
    frame_step = frame_length-frame_overlap;
    frames_N = int((room_audio[0].shape[0]-frame_length)/frame_step);
    frames = np.arange(0,frames_N);
    frame_starts = np.arange(0,frames_N*frame_step,frame_step);
    frame_ends = frame_starts+frame_length-1;
    
    # split audio to frequency bands
    hz_points = mel_filter_freqs(Fs,nfilt)
    filtered_audio_mics = []
    for mic in range(len(room_audio)):
        filteredaudio, filterbank_sos = split_sig_to_FBs_2(room_audio[mic],Fs,hz_points, 8)
        filtered_audio_mics.append(filteredaudio)

    #% extract coordinates from simulated sigg
    sig_coords = []
    for s in sigg:
        sig_coords.append(s[3])
    sig_coords = np.array(sig_coords)   
    
    # generate target (DoA field)
    mxyz = generate_mxyz_for_each_sample(sig_coords, mic_arr_center)
    doas = get_DoAs2(sig_coords,mxyz)
    xx, yy, field = generate_field_DoA(doa_resolution)
    field = populate_field_DoA(xx, yy, field, doas, sigma, doa_resolution)
    quivers = get_quivers_doa_test(sig_coords,mxyz)

    ################################################################### FRAMES ####
    target = []
    amps = []
    xcorrs = []
    xcorrsflat = []
    for frame in frames:
        frame_start = int(frame_starts[frame])
        frame_end   = int(frame_ends  [frame])
    #    y1 = interp(squeeze(A_pair_filtered(frame_starts(frame):frame_ends(frame),1)),oversmp);
    #    y2 = interp(squeeze(A_pair_filtered(frame_starts(frame):frame_ends(frame),2)),oversmp);
        
        framexc = []
        for band in range(nfilt):
            y1 = filtered_audio_mics[pair_mic_1][band,frame_start:frame_end]
            y2 = filtered_audio_mics[pair_mic_2][band,frame_start:frame_end]
            fbxcorr = signal.correlate(y1,y2)
            framexc.append(fbxcorr)
        xcorrs.append(framexc)
        xcorrsflat.append(np.array(framexc).flatten())
        
        amps.append([])
        xx, yy, field = generate_field_DoA(doa_resolution)
        for ss in range(NUM_SIGNALS):
            amp = np.sum((drysigs[ss,frame_start:frame_end])**2)/frame_length
            amps[frame].append(amp)
            field += kernel2D(1,xx,yy,sigma,doas[ss,0],doas[ss,1])#*amp
        
        target.append(field)
        
    xcorrs = np.array(xcorrs)
    xcorrsflat = np.array(xcorrsflat)
    target = np.array(target)
    return xcorrs, xcorrsflat, target


def generate_xcorrs_target_data_rs_pairs(rs,Fs,duration,nfilt,doa_resolution):
    # create room audio from real speech at randomly selected coordinates
    room_audio, room, sigg, drysigs = create_room_rs(rs,Fs,duration,NUM_SIGNALS,
                                                     lengthx,lengthy,height,
                                                     mic_array_center,mic_array_spread)
    
    # calculate sample values for frames
    frame_length = 2**flpow;
    frame_overlap = frame_length/2;
    
    frame_step = frame_length-frame_overlap;
    frames_N = int((room_audio[0].shape[0]-frame_length)/frame_step);
    frames = np.arange(0,frames_N);
    frame_starts = np.arange(0,frames_N*frame_step,frame_step);
    frame_ends = frame_starts+frame_length-1;
    
    # split audio to frequency bands
    hz_points = mel_filter_freqs(Fs,nfilt)
    filtered_audio_mics = []
    for mic in range(len(room_audio)):
        filteredaudio, filterbank_sos = split_sig_to_FBs_2(room_audio[mic],Fs,hz_points, 8)
        filtered_audio_mics.append(filteredaudio)

    #% extract coordinates from simulated sigg
    sig_coords = []
    for s in sigg:
        sig_coords.append(s[3])
    sig_coords = np.array(sig_coords)   
    
    # generate target (DoA field)
    mxyz = generate_mxyz_for_each_sample(sig_coords, mic_arr_center)
    doas = get_DoAs2(sig_coords,mxyz)
    xx, yy, field = generate_field_DoA(doa_resolution)
    field = populate_field_DoA(xx, yy, field, doas, sigma, doa_resolution)
    quivers = get_quivers_doa_test(sig_coords,mxyz)

    ################################################################### FRAMES ####
    target = []
    amps = []
    xcorrs = []
    xcorrsflat = []
    target_azm = []
    mic_pairs = list(itertools.combinations(range(0, len(filtered_audio_mics)), r=2))
    for frame in frames:
        frame_start = int(frame_starts[frame])
        frame_end   = int(frame_ends  [frame])
    #    y1 = interp(squeeze(A_pair_filtered(frame_starts(frame):frame_ends(frame),1)),oversmp);
    #    y2 = interp(squeeze(A_pair_filtered(frame_starts(frame):frame_ends(frame),2)),oversmp);
        
        framexc = []
        
        for pair in mic_pairs:
            pairxc = []
            pair_mic_1 = pair[0]
            pair_mic_2 = pair[1]
            for band in range(nfilt):
                y1 = filtered_audio_mics[pair_mic_1][band,frame_start:frame_end]
                y2 = filtered_audio_mics[pair_mic_2][band,frame_start:frame_end]
                fbxcorr = signal.correlate(y1,y2)
                zero_lag = int(fbxcorr.shape[0]/2)+1
                pairxc.append(fbxcorr)
            framexc.append(pairxc)
            
        xcorrs.append(framexc)
        xcorrsflat.append(np.array(framexc).flatten())
        
        amps.append([])
     
        
        azmx = np.arange(doa_resolution[0,0],doa_resolution[0,1],1, dtype=np.float64)
        azm = np.zeros(azmx.shape[0])
        xx, yy, field = generate_field_DoA(doa_resolution)
        for ss in range(NUM_SIGNALS):
            amp = np.sum((drysigs[ss,frame_start:frame_end])**2)/frame_length
            amps[frame].append(amp)
            azm += kernel1D(1,azmx,sigma,doas[ss,0])#*amp
            field += kernel2D(1,xx,yy,sigma,doas[ss,0],doas[ss,1])*amp
        
        target_azm.append(azm)    
        target.append(field)
        
#    xcorrs = np.tanh(np.array(xcorrs)*100)
#    xcorrsflat = np.tanh(np.array(xcorrsflat)*100)
#    target = np.tanh(np.array(target)*10) # empirical value
#    target_azm = np.tanh(np.array(target_azm))
    xcorrs = np.array(xcorrs)
    xcorrsflat = np.array(xcorrsflat)
    target = np.array(target) # empirical value
    target_azm = np.tanh(np.array(target_azm))
    amps = np.array(amps)
    
    return xcorrs, xcorrsflat, target, target_azm, amps


class DynamicUpdate():
    #Suppose we know the x range
    min_x = 0
    max_x = 10

    def on_launch(self,log=False):
        #Set up plot
        self.figure, self.ax = plt.subplots()
        if log==False:
            self.lines, = self.ax.plot([],[], 'o')
            self.lines2, = self.ax.plot([],[], 'x')
        else:
            self.lines, = self.ax.semilogy([],[], 'o')
            self.lines2, = self.ax.semilogy([],[], 'x')
        #Autoscale on unknown axis and known lims on the other
        self.ax.set_autoscale_on(True)
#        self.ax.set_xlim(self.min_x, self.max_x)
        #Other stuff
        self.ax.grid()
        ...

    def on_running(self, xdata, ydata, valdata, cut=0):
        #Update data (with the new _and_ the old points)
        if (len(xdata) < cut) or cut==0:
           self.lines.set_xdata(xdata)
           self.lines.set_ydata(ydata)
           self.lines2.set_xdata(xdata)
           self.lines2.set_ydata(valdata)
        else:
           self.lines.set_xdata(xdata[-50:])
           self.lines.set_ydata(ydata[-50:])
           self.lines2.set_xdata(xdata[-50:])
           self.lines2.set_ydata(valdata[-50:])
        #Need both of these in order to rescale
        self.ax.relim()
        self.ax.autoscale_view()
        #We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        
        
        
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

gmail_user = 'strutistrutis@gmail.com'  
gmail_password = 'strutistrutistrutis'

sent_from = gmail_user  
to = ['strutistrutis@gmail.com']  
subject = 'OMG Super Important Message'  
body = 'DONE!'

email_text = """\  
From: %s  
To: %s  
Subject: %s

%s
""" % (sent_from, ", ".join(to), subject, body)

     
        
def SendMail(ImgFileName,mailtext):
    img_data = open(ImgFileName, 'rb').read()
    msg = MIMEMultipart()
    msg['Subject'] = mailtext
    msg['From'] = gmail_user
    msg['To'] = gmail_user

    text = MIMEText(mailtext)
    msg.attach(text)
    image = MIMEImage(img_data, name=os.path.basename(ImgFileName))
    msg.attach(image)
    
    msg.attach(MIMEText(open("model_summary.txt").read()))
    
    attachment = MIMEText(json.dumps(model.get_config()))
    attachment.add_header('Content-Disposition', 'attachment', 
                          filename="model_config.json")
    msg.attach(attachment)
    
    
    try: 
        s = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        s.ehlo()
#        s.starttls()
        s.ehlo()
        s.login(gmail_user, gmail_password)
        s.sendmail(sent_from, to, msg.as_string())
        s.close()
        print('Email sent!')
    except:  
        print('Something went wrong...')       
        
        
import random
import tables

def get_random_set_of_samples_audio(hdf5_path,n):
    extendable_hdf5_file = tables.open_file(hdf5_path, mode='r')
    random_training_data_batch = []
    rand_audio = []
    rand_target = []
    num_samples = extendable_hdf5_file.root.audio.shape[0]
    xcv = random.sample(range(num_samples), n)
    #    for i in range(m,n):
    #        rand_xcorrs.append(extendable_hdf5_file.root.xcorrs[i])
    #        rand_target.append(extendable_hdf5_file.root.target[i])
    rand_audio = extendable_hdf5_file.root.audio[xcv,:,:]
    rand_target = extendable_hdf5_file.root.target[xcv,:,:]
    extendable_hdf5_file.close()
    return np.array(rand_audio),np.array(rand_target)


def get_contigous_set_of_samples_audio(hdf5_path,m,n):
    extendable_hdf5_file = tables.open_file(hdf5_path, mode='r')
    random_training_data_batch = []
    rand_audio = []
    rand_target = []
    num_samples = extendable_hdf5_file.root.audio.shape[0]
    xcv = np.arange(m,n)
    #    for i in range(m,n):
    #        rand_xcorrs.append(extendable_hdf5_file.root.xcorrs[i])
    #        rand_target.append(extendable_hdf5_file.root.target[i])
    rand_audio = extendable_hdf5_file.root.audio[xcv,:,:]
    rand_target = extendable_hdf5_file.root.target[xcv,:,:]
    extendable_hdf5_file.close()
    return np.array(rand_audio),np.array(rand_target)

def get_number_of_samples_in_dataset_audio(hdf5_path):
    extendable_hdf5_file = tables.open_file(hdf5_path, mode='r')
    num_samples = extendable_hdf5_file.root.audio.shape[0]
    extendable_hdf5_file.close()
    return num_samples


def get_specs(xtrain, 
                  oversample_rate=8,
                  NPS=64,
                  color_gamma=1,Fs = 44100):
    '''
    Uses _audio functions
    '''

    Fsos = Fs*oversample_rate
    shift = 0    
    specs = []
#    mic_pairs = list(itertools.combinations(range(0, xtrain.shape[1]), r=2))
    for frame in range(xtrain.shape[0]):
        if np.mod(frame,100)==0:
            print(frame)
            
        framespecs = []
#        for pair in mic_pairs:    
        for channel in range(xtrain.shape[1]):
            a1 = xtrain[(frame,channel)]
            b1 = signal.resample(a1,a1.shape[0]*oversample_rate)
            
            f1, t1, Zxx1 = signal.stft(b1, Fsos, nperseg=NPS)
            framespecs.append(np.abs(Zxx1))
        specs.append(framespecs)
    return np.array(specs)

import random
def get_random_set_of_samples(hdf5_path,n):
    extendable_hdf5_file = tables.open_file(hdf5_path, mode='r')
    random_training_data_batch = []
    rand_xcorrs = []
    rand_target = []
    num_samples = extendable_hdf5_file.root.xcorrs.shape[0]
    xcv = random.sample(range(num_samples), n)
#    for i in range(m,n):
#        rand_xcorrs.append(extendable_hdf5_file.root.xcorrs[i])
#        rand_target.append(extendable_hdf5_file.root.target[i])
    rand_xcorrs = extendable_hdf5_file.root.xcorrs[xcv,:,:,:]
    rand_target = extendable_hdf5_file.root.target[xcv,:,:]
    extendable_hdf5_file.close()
    return np.array(rand_xcorrs),np.array(rand_target)

def get_random_set_of_samples_STFT(hdf5_path,n):
    extendable_hdf5_file = tables.open_file(hdf5_path, mode='r')
    random_training_data_batch = []
    rand_rax = []
    rand_target = []
    num_samples = extendable_hdf5_file.root.rax.shape[0]
    xcv = random.sample(range(num_samples), n)
#    for i in range(m,n):
#        rand_xcorrs.append(extendable_hdf5_file.root.xcorrs[i])
#        rand_target.append(extendable_hdf5_file.root.target[i])
    rand_rax = extendable_hdf5_file.root.rax[xcv,:,:]
    rand_target = extendable_hdf5_file.root.target[xcv,:]
    extendable_hdf5_file.close()
    return np.array(rand_rax),np.array(rand_target)

def get_random_set_of_samples_irs(hdf5_path,n):
    extendable_hdf5_file = tables.open_file(hdf5_path, mode='r')
    random_training_data_batch = []
    rand_xcorrs = []
    rand_irs = []
    rand_target = []
    num_samples = extendable_hdf5_file.root.xcorrs.shape[0]
    xcv = random.sample(range(num_samples), n)
#    for i in range(m,n):
#        rand_xcorrs.append(extendable_hdf5_file.root.xcorrs[i])
#        rand_target.append(extendable_hdf5_file.root.target[i])
    rand_xcorrs = extendable_hdf5_file.root.xcorrs[xcv,:,:,:]
    rand_target = extendable_hdf5_file.root.target[xcv,:,:]
    rand_irs = extendable_hdf5_file.root.irs[xcv,:,:]
    extendable_hdf5_file.close()
    return np.array(rand_xcorrs),np.array(rand_irs),np.array(rand_target)


def get_contigous_set_of_samples(hdf5_path,m,n):
    extendable_hdf5_file = tables.open_file(hdf5_path, mode='r')
    random_training_data_batch = []
    rand_xcorrs = []
    rand_target = []
    num_samples = extendable_hdf5_file.root.xcorrs.shape[0]
    xcv = np.arange(m,n)
#    for i in range(m,n):
#        rand_xcorrs.append(extendable_hdf5_file.root.xcorrs[i])
#        rand_target.append(extendable_hdf5_file.root.target[i])
    rand_xcorrs = extendable_hdf5_file.root.xcorrs[xcv,:,:,:]
    rand_target = extendable_hdf5_file.root.target[xcv,:,:]
    extendable_hdf5_file.close()
    return np.array(rand_xcorrs),np.array(rand_target)

def get_number_of_samples_in_dataset(hdf5_path):
    extendable_hdf5_file = tables.open_file(hdf5_path, mode='r')
    num_samples = extendable_hdf5_file.root.xcorrs.shape[0]
    extendable_hdf5_file.close()
    return num_samples

def get_number_of_samples_in_dataset_STFT(hdf5_path):
    extendable_hdf5_file = tables.open_file(hdf5_path, mode='r')
    num_samples = extendable_hdf5_file.root.rax.shape[0]
    extendable_hdf5_file.close()
    return num_samples


def generate_xcorrs_target_data_rs_pairs_mpos_spos(rs,Fs,time,nfilt,order,
                                                   absorbtion,
                                                   doa_resolution,mpos,spos,sigma,limits,flpow,mic_arr_center):
    # create room audio from real speech at randomly selected coordinates
    '''
    time is audio signal samples (real speech signal)
    '''
    lengthx = limits[1,0]
    lengthy = limits[1,1]
    height = limits[1,2]
    NUM_SIGNALS = spos.shape[0]
    room_audio, room, sigg, drysigs = create_room_rs_time_mpos_spos(rs,Fs,time,
                                                                    lengthx,
                                                                    lengthy,
                                                                    height,
                                                                    order,
                                                                    absorbtion,
                                                                    mpos,
                                                                    spos)

    room_audio = [x * 100 for x in room_audio] # empirical
    # calculate sample values for frames
    frame_length = 2**flpow;
    frame_overlap = frame_length/2;
    
    frame_step = frame_length-frame_overlap;
    frames_N = int((room_audio[0].shape[0]-frame_length)/frame_step);
    frames = np.arange(0,frames_N);
    frame_starts = np.arange(0,frames_N*frame_step,frame_step);
    frame_ends = frame_starts+frame_length-1;
    
    # split audio to frequency bands
    hz_points = mel_filter_freqs(Fs,nfilt)
    filtered_audio_mics = []
    for mic in range(len(room_audio)):
        filteredaudio, filterbank_sos = split_sig_to_FBs_2(room_audio[mic],Fs,hz_points, 8)
        filtered_audio_mics.append(filteredaudio)

    #% extract coordinates from simulated sigg
    sig_coords = []
    for s in sigg:
        sig_coords.append(s[3])
    sig_coords = np.array(sig_coords)   
    
    # generate target (DoA field)
    mxyz = generate_mxyz_for_each_sample(sig_coords, mic_arr_center)
    doas = get_DoAs2(sig_coords,mxyz)
    xx, yy, field = generate_field_DoA(doa_resolution)
    field = populate_field_DoA(xx, yy, field, doas, sigma, doa_resolution)
    quivers = get_quivers_doa_test(sig_coords,mxyz)

    ################################################################### FRAMES ####
    target = []
    amps = []
    xcorrs = []
    xcorrsflat = []
    target_azm = []
    mic_pairs = list(itertools.combinations(range(0, len(filtered_audio_mics)), r=2))
    for frame in frames:
        frame_start = int(frame_starts[frame])
        frame_end   = int(frame_ends  [frame])
    #    y1 = interp(squeeze(A_pair_filtered(frame_starts(frame):frame_ends(frame),1)),oversmp);
    #    y2 = interp(squeeze(A_pair_filtered(frame_starts(frame):frame_ends(frame),2)),oversmp);
        
        framexc = []
        
        for pair in mic_pairs:
            pairxc = []
            pair_mic_1 = pair[0]
            pair_mic_2 = pair[1]
            for band in range(nfilt):
                y1 = filtered_audio_mics[pair_mic_1][band,frame_start:frame_end]
                y2 = filtered_audio_mics[pair_mic_2][band,frame_start:frame_end]
                fbxcorr = signal.correlate(y1,y2)
                zero_lag = int(fbxcorr.shape[0]/2)+1
                pairxc.append(fbxcorr[zero_lag-64:zero_lag+64])
            framexc.append(pairxc)
            
        xcorrs.append(framexc)
        xcorrsflat.append(np.array(framexc).flatten())
        
        amps.append([])
     
        
        azmx = np.arange(doa_resolution[0,0],doa_resolution[0,1],1, dtype=np.float64)
        azm = np.zeros(azmx.shape[0])
        xx, yy, field = generate_field_DoA(doa_resolution)
        for ss in range(NUM_SIGNALS):
            amp = np.sum((drysigs[ss,frame_start:frame_end])**2)
            amps[frame].append(amp)
            azm += kernel1D(1,azmx,sigma,doas[ss,0])#*amp
            field += kernel2D(1,xx,yy,sigma,doas[ss,0],doas[ss,1])*amp
        
        target_azm.append(azm)    
        target.append(field)
        
#    xcorrs = np.tanh(np.array(xcorrs)*100)
#    xcorrsflat = np.tanh(np.array(xcorrsflat)*100)
#    target = np.tanh(np.array(target)*10) # empirical value
#    target_azm = np.tanh(np.array(target_azm))
    xcorrs = np.array(xcorrs)
#    xcorrs /= np.max(xcorrs)
    xcorrsflat = np.array(xcorrsflat)
#    xcorrsflat /= np.max(xcorrsflat)
    target = np.array(target) # empirical value
#    target /= np.max(target)
    
    target_azm = np.array(target_azm)
#    target_azm /= np.max(target_azm)
    amps = np.array(amps)
#    amps /= np.max(amps)
    
    return xcorrs, xcorrsflat, target, target_azm, amps, doas

def get_centroids(Z,threshold=1):
    #Set everything below the threshold to zero:
    Z_thresh = np.copy(Z)
    Z_thresh[Z_thresh<threshold] = 0
    
    
    #now find the objects
    labeled_image, number_of_objects = scipy.ndimage.label(Z_thresh)
    peak_slices = scipy.ndimage.find_objects(labeled_image)
    
    def centroid(data):
        h,w = np.shape(data)   
        x = np.arange(0,w)
        y = np.arange(0,h)
    
        X,Y = np.meshgrid(x,y)
    
        cx = np.sum(X*data)/np.sum(data)
        cy = np.sum(Y*data)/np.sum(data)
    
        return cx,cy
    
    centroids = []
    
    for peak_slice in peak_slices:
        dy,dx  = peak_slice
        x,y = dx.start, dy.start
        cx,cy = centroid(Z_thresh[peak_slice])
        centroids.append((x+cx,y+cy))
    
    return centroids

from matplotlib.colors import LogNorm

import scipy
def get_N_centroids(ZEST,N):
    threshest = np.max(ZEST)
    threshd = threshest/100
#    # find lowest threshold (for max centroids)    
#    while len(get_centroids(ZEST,threshest)) < len(get_centroids(ZEST,threshest+threshd)):
#        threshest += threshd
#        print(threshest)
#        print(len(get_centroids(ZEST,threshest)))
    # find threshold for N centroids        
    while len(get_centroids(ZEST,threshest))<N and threshest > 0:
        threshest-=threshd
#        print('threshold: {}'.format(threshest))
#        print('number of centroids: {}'.format(len(get_centroids(ZEST,threshest))))
    centroidsEST = get_centroids(ZEST,threshest)
    return centroidsEST


def getcoordscanlist(limits,ni,nj,nk,margin):
    coordscanlist = []
    ispace = np.linspace(limits[0,0]+margin,limits[1,0]-margin,ni)
    jspace = np.linspace(limits[0,1]+margin,limits[1,1]-margin,nj)
    kspace = np.linspace(limits[0,2]+margin,limits[1,2]-margin,nk)
    
    for k in range(len(kspace)):
        for j in range(len(jspace)):
            for i in range(len(ispace)):
                coordscanlist.append([[ispace[i],jspace[j],kspace[k]]])
    return np.array(coordscanlist)
#%%
def getcoordscanlist2src(limits,ni,nj,nk,margin):
    coordscanlist = []
    ispace = np.linspace(limits[0,0]+margin,limits[1,0]-margin,ni)
    jspace = np.linspace(limits[0,1]+margin,limits[1,1]-margin,nj)
    kspace = np.linspace(limits[0,2]+margin,limits[1,2]-margin,nk)
    
    for k in range(len(kspace)):
        for j in range(len(jspace)):
            for i in range(len(ispace)):
                coordscanlist.append([[ispace[i],jspace[j],kspace[k]],
                                      [ispace[int(np.mod(i+len(ispace)/2,len(kspace)))],
                                       jspace[int(np.mod(j+len(jspace)/2,len(kspace)))],
                                       kspace[int(np.mod(k+len(kspace)/2,len(kspace)))]]])
    return np.array(coordscanlist)


def adjust_centroids(centroids):
    '''
    empirically found from the imshow; needs validation
    also the doas are something completely else
    '''
    return (np.array(centroids)+[0.25,0.5])*[10]+[-180,-90]

def get_centroids_err(centroidst,centroidsy,centroids_N):
    '''
    calculates euclidean distances between all points, returns centroids_N 
    smallest values, as we expect those to be the distances between the ground
    truth and the estimation; we can't be sure, because centroid lists are 
    unsorted
    '''
    errdist = scipy.spatial.distance.cdist(centroidst,centroidsy)
    errdistmin = np.sort(errdist.flatten())[0:centroids_N]
    return errdistmin
#%%
def RT602Absorbtion(limits,RT60mean):
    lengthx = limits[1,0]-limits[0,0]
    lengthy = limits[1,1]-limits[0,1]
    height  = limits[1,2]-limits[0,2]
    S = ((lengthx*lengthx)+(lengthx*height)+(lengthy*height))*2
    V = lengthx*lengthx*height
    absorbtion = V/(S*RT60mean)*0.1611 # wiki reverberation ir     # http://hyperphysics.phy-astr.gsu.edu/hbase/Acoustic/revmod.html
    return absorbtion, S, V


#%%
def add_zeros_column(pts):
    pts = np.array(pts)
    pts = np.hstack((pts,np.zeros((pts.shape[0],1))))
    return pts

def get_max_radius(limits,mac,percentage=0.9):
    return np.min(np.abs(limits[:,0:2]-mac[0:2]))*percentage

def spos_circle(r = 1, n = 100, mac=0):    
    pts = []
    for x in range(1,n+1):
        pts.append([np.cos(2*np.pi/n*x)*r, np.sin(2*np.pi/n*x)*r])    
    pts = add_zeros_column(pts)    
    pts = pts+mac    
    return pts

#%%
def snr_noise(s=None,SNR_dB=None):
    snr = 10.0**(SNR_dB/10.0)
    p1 = np.mean(np.power(s,2))
    n = p1/snr
    w = np.sqrt(n)*np.random.rand(s.shape[0])
    pw = np.mean(np.power(w,2))
    return w