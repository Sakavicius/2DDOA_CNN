# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 18:16:44 2021

@author: sauli
"""
import pyroomacoustics as pra
import numpy as np
from hyperopt import Trials, STATUS_OK, tpe
import keras
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.models import Model
from keras.utils import np_utils
from keras.layers import Input
from keras.layers.core import Dense, Activation, Dropout, Flatten, Reshape
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import DepthwiseConv2D, Conv2D
from keras.layers.pooling import MaxPooling2D
# from keras.layers.normalization import BatchNormalization
# from keras.utils import plot_model
from keras import optimizers
import scipy
from scipy.signal import fftconvolve
from scipy.io import wavfile
from scipy.interpolate import griddata
from scipy.signal import fftconvolve
from scipy.io import wavfile
from scipy import signal
import random
import wandb
from wandb.keras import WandbCallback
import tensorflow as tf
from tqdm import tqdm
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage import data, img_as_float

from sklearn.metrics.pairwise import euclidean_distances

from ssfunctions import *
# from AAA_dataset8_FUNC import *
# from AAA_dataset8_CONST import *
import os

import tables
from datetime import datetime


#%% FUNCTIONS
# =============================================================================
# FUNCTIONS
# =============================================================================
def get_number_of_samples_in_dataset_STFT(hdf5_path):
    extendable_hdf5_file = tables.open_file(hdf5_path, mode='r')
    num_samples = extendable_hdf5_file.root.rax.shape[0]
    extendable_hdf5_file.close()
    return num_samples

def kernel2D(A,x,y,sigma,dx,dy):
    return A*np.exp(-((x-dx)**2 + (y-dy)**2)/(2*sigma**2))

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

def generate_mxyz_for_each_sample(point_polar_coords,mic_arr_center):
    return np.tile(mic_arr_center, (point_polar_coords.shape[0], 1))

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

def get_random_n_spos(limits,n):
    '''
    Returns an array of n random 3D positions
    '''
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

def get_random_mpos(limits,mpos):
    '''
    Returns randomly shifted (within limits) mpos array
    '''
    mic_array_center = np.mean(mpos,axis=0)
    mextent = mpos - mic_array_center
    minx = limits[0,0] - np.min(mextent[:,0]) 
    maxx = limits[1,0] - np.max(mextent[:,0]) 
    miny = limits[0,1] - np.max(mextent[:,1]) 
    maxy = limits[1,1] - np.max(mextent[:,1]) 
    minz = limits[0,2] - np.max(mextent[:,2]) 
    maxz = limits[1,2] - np.max(mextent[:,2]) 
    mposrnd = mpos+np.array([ 
            (np.random.rand()*(maxx-minx)+minx),
            (np.random.rand()*(maxy-miny)+miny),
            (np.random.rand()*(maxz-minz)+minz) ])
    return np.array(mposrnd)

def random_file(DATA_DIR):
    '''
    Returns random flac file from a directory tree with the root directory being
    DATA_DIR.
    
    '''
    file = os.path.join(DATA_DIR, random.choice(os.listdir(DATA_DIR)));
    if os.path.isdir(file):
        return random_file(file)
    else:
        if file.endswith('.flac'):
            return file
        else:
            return random_file(file)

def RA2raxFt(RA):
    RA = np.array(RA)

    #% Extract T, F and STFT
    F = RA[0,0,0]
    t = []
    t.append(RA[0,0,1])
    for tt in range(1,RA.shape[0]):
        t.append(RA[tt,0,1]+RA[tt-1,0,1][-1])
    t = np.array(t).flatten()
    #%
    rax = []
    for N in range(RA.shape[0]):
        raxrax = []
        for M in range(RA.shape[1]):
            raxrax.append(RA[N,M,2])
        rax.append(np.array(raxrax))
    
    rax = np.array(rax)
    rax = np.moveaxis(rax,3,1)
    rax = np.reshape(rax,(-1,rax.shape[2],rax.shape[3]))
    return rax, F, t

def generate_doas_and_field(SIG_COORDS,mic_arr_center,rax,
                            resolution_x=10,resolution_y=10,
                            sigma_x=1,sigma_y=1,sigma_res=True):
    # generate target (DoA field)
    mxyz = generate_mxyz_for_each_sample(SIG_COORDS, mic_arr_center)
    doas = get_DoAs2(SIG_COORDS,mxyz)
    
    if sigma_res:        
        sigma_x = resolution_x
        sigma_y = resolution_y
    
    field_xrange = [-180,180]
    field_yrange = [-90,90]
    x = np.arange(field_xrange[0],field_xrange[1],resolution_x)   # coordinate arrays -- make sure they contain 0!
    y = np.arange(field_yrange[0],field_yrange[1],resolution_y)
    xx, yy = np.meshgrid(x,y)
    field = np.zeros([len(y),len(x)])
    for point in doas:
        field += kernel2D(1,xx,yy,sigma_x,point[0],point[1])
    
    target = np.tile(field,(rax.shape[0],1,1))
    return target, doas

def generate_azim_elev(SIG_COORDS,mic_arr_center,rax,
                       resolution_x=10,resolution_y=10,
                       sigma_x=1,sigma_y=1,sigma_res=True):

    # generate target (DoA field)
    mxyz = generate_mxyz_for_each_sample(SIG_COORDS, mic_arr_center)
    doas = get_DoAs2(SIG_COORDS,mxyz)
    
    # NEW, Gaussian way
    # resolution_x = 10 # degrees
    # resolution_y = 10 # degrees
    if sigma_res:        
        sigma_x = resolution_x
        sigma_y = resolution_y
    
    field_xrange = [-180,180]
    field_yrange = [-90,90]
    x = np.arange(field_xrange[0],field_xrange[1],resolution_x)   # coordinate arrays -- make sure they contain 0!
    y = np.arange(field_yrange[0],field_yrange[1],resolution_y)
    
    field_x = np.zeros([len(x)])
    field_y = np.zeros([len(y)])
    for point in doas:
        field_x += kernel1D(1,x,sigma_x,point[0])
        field_y += kernel1D(1,y,sigma_y,point[1])

    azim = np.tile(field_x,(rax.shape[0],1,1))
    elev = np.tile(field_y,(rax.shape[0],1,1))
    # azimelev = [azim, elev]
    
    # target = azimelev

    return azim, elev, doas

def get_lengthxyheight_from_limits(limits):
    lengthx = limits[1,0]
    lengthy = limits[1,1]
    height = limits[1,2]
    return lengthx, lengthy, height


def generate_STFT_noise_training_sample(limits,
                                        sposlimits,
                                        mposlimits,
                                        NUM_SIGNALS_TO_CONCATENATE,
                                        mposs,
                                        mpos,
                                        duration,
                                        absorbtion,
                                        order,azel=True,
                                        doa_res_x=10,doa_res_y=10,
                                        sigma_x=1,sigma_y=1,sigma_res=True):
    '''
    mposs is array geometry which will be pushed around using get_random_mpos()
    function
    '''

    lengthx, lengthy, height = get_lengthxyheight_from_limits(limits)
    
    RA = []
    SIGG = []
    #% Generate NUM_SIGNALS_TO_CONCATENATE signals at different DoA
    for i in range(NUM_SIGNALS_TO_CONCATENATE):
        # positions
        
        # mpos = get_random_mpos(mposlimits,mposs) # RANDOM MPOS
        mic_array_center = np.mean(mpos,axis=0)
        mic_arr_center = mic_array_center
        
        spos = get_random_n_spos(sposlimits,1) # RANDOM SPOS
        
        # room audio
        room_audio, room, sigg, drysigs = create_room_noise_mpos_spos(Fs,duration,lengthx,lengthy,height,order,absorbtion,mpos,spos)
        
        # STFT
        ra = []
        for rx in room_audio:
            ra.append(scipy.signal.stft(rx,**stft_args))
        RA.append(ra)
        SIGG.append(sigg)
    
    rax, F, t = RA2raxFt(RA)
        
    #% Shuffle!
    for k in range(rax.shape[2]):
        np.random.shuffle(rax[:,:,k])
    
    #% Extract DoAs
    #% extract coordinates from simulated sigg
    SIG_COORDS = []
    for S in SIGG:
        sig_coords = []
        for s in S:
            sig_coords.append(s[3])
        sig_coords = np.array(sig_coords[0])   
        SIG_COORDS.append(sig_coords)
    SIG_COORDS = np.array(SIG_COORDS)    
    

    target, doas = generate_doas_and_field(
        SIG_COORDS,mic_arr_center,rax,
        resolution_x=doa_res_x,resolution_y=doa_res_y,
        sigma_x=sigma_x,sigma_y=sigma_y,sigma_res=sigma_res)
    return rax, target, SIG_COORDS, mpos, doas


def generate_STFT_noise_sample_2D_spos_mpos(limits=None,
                                        spos=None,
                                        mpos=None,
                                        duration=None,
                                        absorbtion=None,
                                        order=None,Fs=None,azel=True,
                                        doa_res_x=10,doa_res_y=10,
                                        sigma_x=1,sigma_y=1,sigma_res=True,stft_args=None,
                                        shuffle=True):
    '''
    DATA_DIR :   LIBRI data root directory
    spos :       positions of sources (2D array, N_SRCx[x,y,z]), number of sources is
                  inferred from the length of the N_SRC dimension
    mpos :       positions of the microphones of the array; the center of the 
                  array (needed for DoA calculation) is inferred from mpos as
                  the arithmetic mean in all three dimensions
    duration :   duration in seconds of the simulated room audio segment
    absorbtion : room absorbtion
    order :      order of the image-source model simulation
    
    '''

    NUM_MIC = mpos.shape[0]
    mic_array_center = np.mean(mpos,axis=0)
    mic_arr_center = mic_array_center
    
    lengthx, lengthy, height = get_lengthxyheight_from_limits(limits)
    
    RA = []
    SIGG = []    
    # room audio
    room_audio, room, sigg, drysigs = create_room_noise_mpos_spos(
        Fs,duration,lengthx,lengthy,height,order,absorbtion,mpos,spos)
        
    # STFT
    ra = []
    for rx in room_audio:
        ra.append(scipy.signal.stft(rx,**stft_args))
    RA.append(ra)
    SIGG.append(sigg)
    
    rax, F, t = RA2raxFt(RA)
    
    if shuffle:
            #% Shuffle!
        for k in range(rax.shape[2]):
            np.random.shuffle(rax[:,:,k])
        
    #% Extract DoAs
    #% extract coordinates from simulated sigg
    sig_coords = []
    for s in sigg:
        sig_coords.append(s[3])
    sig_coords = np.array(sig_coords)      
    SIG_COORDS = sig_coords
        

    target, doas = generate_doas_and_field(
        SIG_COORDS,mic_arr_center,rax,
        resolution_x=doa_res_x,resolution_y=doa_res_y,
        sigma_x=sigma_x,sigma_y=sigma_y,sigma_res=sigma_res)
    return rax, target, SIG_COORDS, mpos, doas

def generate_STFT_LIBRI_speech_sample_2D_spos_mpos(DATA_DIR=None,
                                                   limits=None,
                                                   spos=None,
                                                   mpos=None,
                                                   duration=None,
                                                   absorbtion=None,
                                                   order=None,Fs=None,azel=True,
                                                   doa_res_x=10,doa_res_y=10,
                                                   sigma_x=1,sigma_y=1,sigma_res=True,stft_args=None,
                                                   shuffle=True):
    '''
    DATA_DIR :   LIBRI data root directory
    spos :       positions of sources (2D array, N_SRCx[x,y,z]), number of sources is
                  inferred from the length of the N_SRC dimension
    mpos :       positions of the microphones of the array; the center of the 
                  array (needed for DoA calculation) is inferred from mpos as
                  the arithmetic mean in all three dimensions
    duration :   duration in seconds of the simulated room audio segment
    absorbtion : room absorbtion
    order :      order of the image-source model simulation
    
    '''

    NUM_MIC = mpos.shape[0]
    mic_array_center = np.mean(mpos,axis=0)
    mic_arr_center = mic_array_center
    
    lengthx, lengthy, height = get_lengthxyheight_from_limits(limits)
    
    RA = []
    SIGG = []    
    # room audio
    room_audio, room, sigg, drysigs = create_room_libris_mpos_spos(DATA_DIR,Fs,duration,lengthx,lengthy,height,order,absorbtion,mpos,spos)
    
    # STFT
    ra = []
    for rx in room_audio:
        ra.append(scipy.signal.stft(rx,**stft_args))
    RA.append(ra)
    SIGG.append(sigg)
    
    rax, F, t = RA2raxFt(RA)
    
    if shuffle:
            #% Shuffle!
        for k in range(rax.shape[2]):
            np.random.shuffle(rax[:,:,k])
        
    #% Extract DoAs
    #% extract coordinates from simulated sigg
    sig_coords = []
    for s in sigg:
        sig_coords.append(s[3])
    sig_coords = np.array(sig_coords)      
    SIG_COORDS = sig_coords
        

    target, doas = generate_doas_and_field(
        SIG_COORDS,mic_arr_center,rax,
        resolution_x=doa_res_x,resolution_y=doa_res_y,
        sigma_x=sigma_x,sigma_y=sigma_y,sigma_res=sigma_res)
    return rax, target, SIG_COORDS, mpos, doas


# def generate_STFT_signal_sample_2D_spos_mpos(signal=None,
#                                                    limits=None,
#                                                    spos=None,
#                                                    mpos=None,
#                                                    duration=None,
#                                                    absorbtion=None,
#                                                    order=None,Fs=None,azel=True,
#                                                    doa_res_x=10,doa_res_y=10,
#                                                    sigma_x=1,sigma_y=1,sigma_res=True,stft_args=None,
#                                                    shuffle=True):
#     '''
#     signal :   array of single channel audio data array, [num_signals x num_audio_samples]
#     spos :       positions of sources (2D array, N_SRCx[x,y,z]), number of sources is
#                   inferred from the length of the N_SRC dimension
#     mpos :       positions of the microphones of the array; the center of the 
#                   array (needed for DoA calculation) is inferred from mpos as
#                   the arithmetic mean in all three dimensions
#     duration :   duration in seconds of the simulated room audio segment
#     absorbtion : room absorbtion
#     order :      order of the image-source model simulation
    
#     '''

#     NUM_MIC = mpos.shape[0]
#     mic_array_center = np.mean(mpos,axis=0)
#     mic_arr_center = mic_array_center
    
#     lengthx, lengthy, height = get_lengthxyheight_from_limits(limits)
    
#     RA = []
#     SIGG = []    
#     # room audio
#     room_audio, room, sigg, drysigs = create_room_signal_mpos_spos(signal,Fs,duration,lengthx,lengthy,height,order,absorbtion,mpos,spos)
    
#     # STFT
#     ra = []
#     for rx in room_audio:
#         ra.append(scipy.signal.stft(rx,**stft_args))
#     RA.append(ra)
#     SIGG.append(sigg)
    
#     rax, F, t = RA2raxFt(RA)
    
#     if shuffle:
#             #% Shuffle!
#         for k in range(rax.shape[2]):
#             np.random.shuffle(rax[:,:,k])
        
#     #% Extract DoAs
#     #% extract coordinates from simulated sigg
#     sig_coords = []
#     for s in sigg:
#         sig_coords.append(s[3])
#     sig_coords = np.array(sig_coords)      
#     SIG_COORDS = sig_coords
        

#     target, doas = generate_doas_and_field(
#         SIG_COORDS,mic_arr_center,rax,
#         resolution_x=doa_res_x,resolution_y=doa_res_y,
#         sigma_x=sigma_x,sigma_y=sigma_y,sigma_res=sigma_res)
#     return rax, target, SIG_COORDS, mpos, doas

#%%
def centroid_coords_to_doas(coordinates,doa_res_x,doa_res_y):
    doa_resolution = np.array([[-180/doa_res_x,180/doa_res_x],[-90/doa_res_y,90/doa_res_x]])
    c = []
    for coord in coordinates:
        cc = ((coord-doa_resolution[::-1,1])/doa_resolution[::-1,1]*np.array([90,180]))
        c.append(cc[::-1])
    return np.array(c)


def get_random_set_of_samples_STFT(hdf5_path,n):
    # extendable_hdf5_file = tables.open_file(hdf5_path, mode='r')
    with tables.open_file(hdf5_path, mode='r') as extendable_hdf5_file:
        random_training_data_batch = []
        rand_rax = []
        rand_target = []
        # rand_azim = []
        # rand_elev = []
        num_samples = extendable_hdf5_file.root.rax.shape[0]
        xcv = random.sample(range(num_samples), n)
    #    for i in range(m,n):
    #        rand_xcorrs.append(extendable_hdf5_file.root.xcorrs[i])
    #        rand_target.append(extendable_hdf5_file.root.target[i])
        rand_rax = extendable_hdf5_file.root.rax[xcv,:,:]
        rand_target = extendable_hdf5_file.root.target[xcv,:,:]
        # rand_azim = extendable_hdf5_file.root.azimuth[xcv,:,:]
        # rand_elev = extendable_hdf5_file.root.elevation[xcv,:,:]
        
    return np.array(rand_rax),np.array(rand_target)

def get_all_set_of_samples_STFT(hdf5_path,return_spos=False):
    # extendable_hdf5_file = tables.open_file(hdf5_path, mode='r')
    with tables.open_file(hdf5_path, mode='r') as extendable_hdf5_file:
        # random_training_data_batch = []
        # rand_rax = []
        # rand_target = []
        # rand_azim = []
        # rand_elev = []
        # num_samples = extendable_hdf5_file.root.rax.shape[0]
        # xcv = random.sample(range(num_samples), n)
    #    for i in range(m,n):
    #        rand_xcorrs.append(extendable_hdf5_file.root.xcorrs[i])
    #        rand_target.append(extendable_hdf5_file.root.target[i])
        rand_rax = extendable_hdf5_file.root.rax[:,:,:]
        rand_target = extendable_hdf5_file.root.target[:,:,:]
        if return_spos:
            rand_spos = extendable_hdf5_file.root.spos[:,:]
        # rand_azim = extendable_hdf5_file.root.azimuth[xcv,:,:]
        # rand_elev = extendable_hdf5_file.root.elevation[xcv,:,:]
    if return_spos:           
        return np.array(rand_rax),np.array(rand_target),np.array(rand_spos)
    else:
        return np.array(rand_rax),np.array(rand_target)
    
def cube_show_slider(cube_test, cube_pred, axis=2, title='', **kwargs):
    """
    Display a 3d ndarray with a slider to move along the third dimension.

    Extra keyword arguments are passed to imshow
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button, RadioButtons

    # generate figure
    fig = plt.figure()
    ax_test = plt.subplot(211)
    ax_pred = plt.subplot(212)
    fig.subplots_adjust(left=0.25, bottom=0.25)

    # select first image
    sample = 0
    time = 0
    im_test = cube_test[sample,time,:,:]
    im_pred = cube_pred[sample,time,:,:]
    # display image
    l_test = ax_test.imshow(im_test, **kwargs)
    l_pred = ax_pred.imshow(im_pred, **kwargs)

    # define slider
    ax_slider_sample = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    ax_slider_time = fig.add_axes([0.25, 0.15, 0.65, 0.03])

    slider_sample = Slider(ax_slider_sample, 'Sample', 0, cube_test.shape[0] - 1,
                    valinit=0, valfmt='%i')
    slider_time = Slider(ax_slider_time, 'Time', 0, cube_test.shape[1] - 1,
                    valinit=0, valfmt='%i')

    def update(val):
        sample = int(slider_sample.val)
        time = int(slider_time.val)
        im_test = cube_test[sample,time,:,:]
        im_pred = cube_pred[sample,time,:,:]
        l_test.set_data(im_test, **kwargs)
        l_pred.set_data(im_pred, **kwargs)

    slider_sample.on_changed(update)
    slider_time.on_changed(update)
    
    plt.title(title)
    plt.show()
    return slider_sample, slider_time


def cube_show_slider_pcolormesh(cube_test, cube_pred, axis=2, title='', **kwargs):
    """
    Display a 3d ndarray with a slider to move along the third dimension.

    Extra keyword arguments are passed to imshow
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button, RadioButtons

    # generate figure
    fig = plt.figure()
    ax_test = plt.subplot(211)
    ax_pred = plt.subplot(212)
    fig.subplots_adjust(left=0.25, bottom=0.25)

    # select first image
    sample = 0
    time = 0
    im_test = cube_test[sample,time,:,:]
    im_pred = cube_pred[sample,time,:,:]
    # display image
    l_test = ax_test.pcolormesh(im_test, **kwargs)
    l_pred = ax_pred.pcolormesh(im_pred, **kwargs)

    # define slider
    ax_slider_sample = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    ax_slider_time = fig.add_axes([0.25, 0.15, 0.65, 0.03])

    slider_sample = Slider(ax_slider_sample, 'Sample', 0, cube_test.shape[0] - 1,
                    valinit=0, valfmt='%i')
    slider_time = Slider(ax_slider_time, 'Time', 0, cube_test.shape[1] - 1,
                    valinit=0, valfmt='%i')

    def update(val):
        sample = int(slider_sample.val)
        time = int(slider_time.val)
        im_test = cube_test[sample,time,:,:]
        im_pred = cube_pred[sample,time,:,:]
        l_test.set_array(im_test.ravel())
        l_pred.set_array(im_pred.ravel())

    slider_sample.on_changed(update)
    slider_time.on_changed(update)
    
    plt.title(title)
    plt.show()
    return slider_sample, slider_time

def cube_show_slider_multiple(cube_test, cube_pred, axis=2, title='', **kwargs):
    """
    Display a 3d ndarray with a slider to move along the third dimension.

    Extra keyword arguments are passed to imshow
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button, RadioButtons

    # generate figure
    fig = plt.figure()
    ax_test = plt.subplot(211)
    ax_pred = plt.subplot(212)
    fig.subplots_adjust(left=0.25, bottom=0.25)

    # select first image
    sample = 0
    time = 0
    im_test = cube_test[sample,time,:,:]
    im_pred = cube_pred[sample,time,:,:]
    # display image
    l_test = ax_test.imshow(im_test, **kwargs)
    l_pred = ax_pred.imshow(im_pred, **kwargs)

    # define slider
    ax_slider_sample = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    ax_slider_time = fig.add_axes([0.25, 0.15, 0.65, 0.03])

    slider_sample = Slider(ax_slider_sample, 'Sample', 0, cube_test.shape[0] - 1,
                    valinit=0, valfmt='%i')
    slider_time = Slider(ax_slider_time, 'Time', 0, cube_test.shape[1] - 1,
                    valinit=0, valfmt='%i')

    def update(val):
        sample = int(slider_sample.val)
        time = int(slider_time.val)
        im_test = cube_test[sample,time,:,:]
        im_pred = cube_pred[sample,time,:,:]
        l_test.set_data(im_test, **kwargs)
        l_pred.set_data(im_pred, **kwargs)

    slider_sample.on_changed(update)
    slider_time.on_changed(update)
    
    plt.title(title)
    plt.show()
    return slider_sample, slider_time

def create_tetrahedron(th_side=1):
    th_h = np.sqrt(th_side**2-(th_side/2)**2)
    th_ch = np.sqrt(  (((th_side)/2)**2)/3  )
    th0 = np.array([0,0,0])
    th1 = np.array([th_side,0,0])
    th2 = np.array([th_side/2,th_h,0])
    th3 = np.array([th_side/2,th_ch,th_h])
    th = np.array([th0, 
                   th1,
                   th2,
                   th3])
    return th


#%%
def calculate_prediction_errors(PRED=None,GT=None,num_peaks=None,doa_res_x=None,doa_res_y=None):
    
    PRED = np.array(PRED)
    GT = np.array(GT)
    
    ERRS = []
    GT_COORDS = []
    EST_COORDS = []
    GT_COORDS_IDX = []
    EST_COORDS_IDX = []
    
    if PRED.shape[0] != GT.shape[0]:
        print("PRED and GT has not the same number of samples")
    else:
        for i in tqdm(range(PRED.shape[0])):
            coordinatesGT = peak_local_max(GT[i,0,:,:], min_distance=1,num_peaks=num_peaks)
            coordinatesEST = peak_local_max(PRED[i,0,:,:], min_distance=1,num_peaks=num_peaks)
            GT_COORDS_IDX.append(coordinatesGT)  
            EST_COORDS_IDX.append(coordinatesEST)  
            
            try:
                # print("Trying")
                gt_coords = centroid_coords_to_doas(coordinatesGT, doa_res_x, doa_res_y)
                est_coords = centroid_coords_to_doas(coordinatesEST, doa_res_x, doa_res_y) 
        
        #  %%            
                D = []
                for gt in gt_coords:
                    for est in est_coords:
                        d = gt-est
                        # d[d>180] = d[d>180]-180
                        if d[0] > 180:
                            d[0] -= 180
                        if d[1] > 90:
                            d[1] -= 90
                        D.append(d)
                D = np.array(D)
        #  %%                    print(d)
                err = np.linalg.norm(D,axis=1)
                errs = np.sort(err, axis=None)[0:num_peaks]
                
                while errs.shape[0] < 3:
                    errs = np.append(errs,np.NaN)
                
                while est_coords.shape[0] < 3:
                    est_coords = np.vstack((est_coords,np.array([np.NaN]*num_peaks)))
        
                ERRS.append(errs)        
                GT_COORDS.append(gt_coords)
                EST_COORDS.append(est_coords)
                # print('written')
                
            except:
                print("Not even trying..")
                err = float('NaN')
                print("est coords is zero")
    return np.array(ERRS), np.array(GT_COORDS), np.array(EST_COORDS), np.array(GT_COORDS_IDX), np.array(EST_COORDS_IDX)


def plot_3D_points2(coords,mics,mic_arr_center):
    mxyz = generate_mxyz_for_each_sample(coords, mic_arr_center)
    quivers = get_quivers_doa_test(coords,mxyz)
    X,Y,Z = coords[:,0],coords[:,1],coords[:,2]
    fig = plt.figure()
    ax_room = fig.gca(projection='3d')
    # ax_room.set_aspect('equal')
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
    # texts3d = []
    for i in range(len(coords)): #plot each point + it's index as text above
        text3d = ax_room.text(coords[i,0], coords[i,1], coords[i,2], int(i), ha="center", va="top", color="r")
        # texts3d.append(text3d)
    #     text3d = ax_room.text(coords[i,0], coords[i,1], coords[i,2], np.int_(doas[i]),ha="center", va="bottom", color="magenta")
    #     texts3d.append(text3d)
    # adjust_text(texts3d, autoalign='y', only_move={'text':'y'}, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
    
    
def dataset_spos_to_SPOS(spos_all=None):
    '''
    converts spos information stored in the training/testing datasets to 
    array of format N_samples x N_sources x N_spatial_dims
    '''
    prev_spos = np.array([0,0,0])
    UNIQUE_SPOS = []
    for s in spos_all:
        if (s != prev_spos).any():
            prev_spos = s
            UNIQUE_SPOS.append(s)
            
    SPOS_TESTSPEECH
    UNIQUE_SPOS = np.array(UNIQUE_SPOS)
    SPOS_RESHAPED = np.reshape(UNIQUE_SPOS,(-1,3,3))
    return SPOS_RESHAPED

# SPOS_RESHAPED = dataset_spos_to_SPOS(spos_all=spos_all)

#%%
def dataset_spos_to_SPOS_rep(spos_all=None,num_src=None):
    '''
    converts spos information stored in the training/testing datasets to 
    array of format N_samples x N_sources x N_spatial_dims
    '''
    prev_spos = np.array([0,0,0])
    UNIQUE_SPOS = []
    sample_counter = 0
    src_counter = 0
    S = []
    
    for s in spos_all:
        sample_counter += 1
        
        if (s != prev_spos).any():
            # print(s)
            prev_spos = s

            S.append(s)
            src_counter += 1
            if src_counter == num_src:
                # print(np.array(S))
                # break
                # UNIQUE_SPOS.append(np.repeat(np.array(S),sample_counter,axis=0))   
                UNIQUE_SPOS.append(np.repeat(np.array([S]),sample_counter,axis=0))
                S = []
                src_counter = 0
            sample_counter = 0
    UNIQUE_SPOS = np.array(UNIQUE_SPOS)
    UNIQUE_SPOS = np.vstack(UNIQUE_SPOS)    
    return UNIQUE_SPOS


#%% 
def calculate_prediction_errors_GT_DOAS(PRED=None,GT_DOAS=None,num_peaks=None,doa_res_x=None,doa_res_y=None):
    '''
    same as calculate_prediction_errors but uses 
    '''
    #  %%
    PRED = np.array(PRED)
    GT_DOAS = np.array(GT_DOAS)
    
    ERRS = []
    GT_COORDS = []
    EST_COORDS = []
    GT_COORDS_IDX = []
    EST_COORDS_IDX = []
    
    if PRED.shape[0] != GT_DOAS.shape[0]:
        print("PRED and GT has not the same number of samples")
    else:
        for i in tqdm(range(PRED.shape[0])):
            # coordinatesGT = peak_local_max(GT[i,0,:,:], min_distance=1,num_peaks=num_peaks)
            coordinatesEST = peak_local_max(PRED[i,0,:,:], min_distance=1,num_peaks=num_peaks)
            # GT_COORDS_IDX.append(coordinatesGT)  
            EST_COORDS_IDX.append(coordinatesEST)  
            
            try:
                # print("Trying")
                # gt_coords = centroid_coords_to_doas(coordinatesGT, doa_res_x, doa_res_y)
                gt_coords = GT_DOAS[i]
                est_coords = centroid_coords_to_doas(coordinatesEST, doa_res_x, doa_res_y) 
        
        #  %%            
                D = []
                for gt in gt_coords:
                    for est in est_coords:
                        d = gt-est
                        # d[d>180] = d[d>180]-180
                        if d[0] > 180:
                            d[0] -= 180
                        if d[1] > 90:
                            d[1] -= 90
                        D.append(d)
                D = np.array(D)
        #  %%                    print(d)
                err = np.linalg.norm(D,axis=1)
                errs = np.sort(err, axis=None)[0:num_peaks]
                
                while errs.shape[0] < 3:
                    errs = np.append(errs,np.NaN)
                
                while est_coords.shape[0] < 3:
                    est_coords = np.vstack((est_coords,np.array([np.NaN]*num_peaks)))
        
                ERRS.append(errs)        
                GT_COORDS.append(gt_coords)
                EST_COORDS.append(est_coords)
                # print('written')
                
            except:
                print("Not even trying..")
                err = float('NaN')
                print("est coords is zero")
                
                #  %%
    return np.array(ERRS), np.array(GT_COORDS), np.array(EST_COORDS), np.array(EST_COORDS_IDX)



def calculate_prediction_errors_GT_DOAS_fill(PRED=None,GT_DOAS=None,num_peaks=None,doa_res_x=None,doa_res_y=None,min_distance=1):
    '''
    same as calculate_prediction_errors but 
    adds nan if for some reason coordinates or errors could not be determined
    '''
    #  %%
    PRED = np.array(PRED)
    GT_DOAS = np.array(GT_DOAS)
    
    ERRS = []
    GT_COORDS = []
    EST_COORDS = []
    GT_COORDS_IDX = []
    EST_COORDS_IDX = []
    
    if PRED.shape[0] != GT_DOAS.shape[0]:
        print("PRED and GT has not the same number of samples")
    else:
        for i in tqdm(range(PRED.shape[0])):
            # coordinatesGT = peak_local_max(GT[i,0,:,:], min_distance=1,num_peaks=num_peaks)

            gt_coords = GT_DOAS[i]
            GT_COORDS.append(gt_coords)
            try:
                coordinatesEST = peak_local_max(PRED[i,0,:,:], min_distance=min_distance,num_peaks=num_peaks)
                # GT_COORDS_IDX.append(coordinatesGT)  
                # print("Trying")
                # gt_coords = centroid_coords_to_doas(coordinatesGT, doa_res_x, doa_res_y)
                est_coords = centroid_coords_to_doas(coordinatesEST, doa_res_x, doa_res_y) 
        
        #  %%            
                D = []
                for gt in gt_coords:
                    for est in est_coords:
                        d = gt-est
                        # d[d>180] = d[d>180]-180
                        if d[0] > 180:
                            d[0] -= 180
                        if d[1] > 90:
                            d[1] -= 90
                        D.append(d)
                D = np.array(D)
        #  %%                    print(d)
                err = np.linalg.norm(D,axis=1)
                errs = np.sort(err, axis=None)[0:num_peaks]
                
                while errs.shape[0] < num_peaks:
                    errs = np.append(errs,np.NaN)
                
                while est_coords.shape[0] < num_peaks:
                    est_coords = np.vstack((est_coords,np.array([np.NaN]*num_peaks)))
        


                # print('written')
                
            except:
                print('exception, inserted NaNs')
                errs = np.array([np.NaN]*num_peaks)
                # print("Not even trying..")
                # err = float('NaN')
                # print("est coords is zero")
                est_coords = np.repeat(np.array([[np.NaN]*2]),num_peaks,axis=0)
                coordinatesEST = np.repeat(np.array([[np.NaN]*2]),num_peaks,axis=0)
            
            ERRS.append(errs)        
            EST_COORDS.append(est_coords)    
            EST_COORDS_IDX.append(coordinatesEST)  
                #  %%
    return np.array(ERRS), np.array(GT_COORDS), np.array(EST_COORDS), np.array(EST_COORDS_IDX)


#%%
