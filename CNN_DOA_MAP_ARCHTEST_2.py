# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 15:11:07 2020

@author: sauli
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 16:35:55 2019

STFT CNN constants

@author: MWS
"""

#%% INITIALIZATION
from ssfunctions import *
from CNN_DOA_MAP_WANDB_CLEAN_FUNCTIONS import *

curr_file = os.path.splitext(__file__)[0]
path = os.path.dirname(__file__)

plt.close('all')
# sys.exit("Stopped for manual execution") 

#  %% CONSTANTS

MAKE_NEW_DATASET = False
MAKE_NEW_DATASET_SPEECHTEST = False

shuffle = False

hyperparameter_defaults = dict(
    fs = 16000,  
    lengthx = 5.40,
    lengthy = 5.86,
    height = 2.84,
    th_side = 0.4,
    mpos_center = [3,2,2],
    duration  = 0.1, # duration of simulation in seconds
    NUM_SIGNALS = 1, # per one doa simulation
    NUM_SIGNALS_TO_CONCATENATE = 3,
    absorbtion = 0.8,
    order = 7,
    window = 'hann', 
    nperseg = 512, 
    noverlap = 256, 
    nfft = 512,
    detrend = False, 
    return_onesided = True, 
    boundary = 'zeros', 
    padded = True,
    axis = -1,
    num_samples = 100000,
    conv1_units = 128,
    conv2_units = 64,
    conv3_units = 32,
    conv1_kernel = (2,1),
    conv2_kernel = (2,1),
    conv3_kernel = (2,1),
    conv1_activ = 'elu',
    conv2_activ = 'elu',
    conv3_activ = 'elu',
    conv_dropout = 0.125,
    dense1_units = 512,
    dense2_units = 512,
    dense3_units = 16,
    dense1_activ = 'elu',
    dense2_activ = 'elu',
    dense3_activ = 'elu',
    dense_dropout = 0.125,
    lr = 0.001,
    loss = 'binary_crossentropy',
    optimizer_str = 'adam',
    batch_size = 512,
    epochs = 5,
    doa_res_x = 10,
    doa_res_y = 10,
    sigma_x = 10,
    sigma_y = 10
    )

wandb.init(project="cnn_doa_map",config=hyperparameter_defaults,dir="wandb")
config = wandb.config

print(config)

fs = config.fs
Fs = fs

#  %%% Room parameters

lengthx = config.lengthx
lengthy = config.lengthy
height = config.height

limits = np.array([[0,0,0],[lengthx,lengthy,height]])

#  %%% Microphone array parameters

# actual antennae-based thetrahedral microphone array
th_side = config.th_side

th = create_tetrahedron(th_side)
# plot_scatter_3d(th[:,0],th[:,1],th[:,2])

mpos_center = config.mpos_center # fixed microphone array position
mpos0 = th
mpos = mpos0 + mpos_center # shift microphone array to a position within a room

NUM_MIC = mpos0.shape[0]
mic_array_center = np.mean(mpos0,axis=0)
mic_arr_center = mic_array_center

mposlimits = np.copy(limits)

#  %%% Signal parameters

sposlimits = np.copy(limits)
#sposlimits[:,2] = np.array([mic_array_center[2], mic_array_center[2]])

duration  = config.duration # duration of simulation in seconds
NUM_SIGNALS = config.NUM_SIGNALS # per one doa simulation
NUM_SIGNALS_TO_CONCATENATE = config.NUM_SIGNALS_TO_CONCATENATE
absorbtion = config.absorbtion
order = config.order

#  %%% Training sample generation parameters

sample_gen_args = {'limits':limits,
                   'sposlimits':sposlimits,
                   'mposlimits':mposlimits,
                   'NUM_SIGNALS_TO_CONCATENATE':NUM_SIGNALS_TO_CONCATENATE,
                   'mpos0':mpos0,
                   'mpos':mpos,
                   'duration':duration,
                   'absorbtion':absorbtion,
                   'order':order,
                   'doa_res_x':config.doa_res_x,
                   'doa_res_y':config.doa_res_y,
                   'sigma_x':config.sigma_x,
                   'sigma_y':config.sigma_y}

stft_args = {'fs':Fs, 
             'window':config.window, 
             'nperseg':config.nperseg, 
             'noverlap':config.noverlap, 
             'nfft':config.nfft,
             'detrend':config.detrend, 
             'return_onesided':config.return_onesided, 
             'boundary':config.boundary, 
             'padded':config.padded,
             'axis':config.axis}


DATA_DIR = "E:\DISSERTATION\PY\LIBRISpeech\dev-clean\LibriSpeech\dev-clean"
# hdf5_path = "AAA_dataset8.hdf5"
hdf5_dir = "D:\CNNDOAMAP_DATASETS"
# hdf5_dir = os.path.join('D:','CNNDOAMAP_DATASETS')
hdf5_filename = "{}_doaresxy_{}_{}_sigmaxy_{}_{}_dataset.hdf5".format(os.path.split(curr_file)[-1],config.doa_res_x,config.doa_res_y,config.sigma_x,config.sigma_y)
hdf5_path = os.path.join(hdf5_dir,hdf5_filename)
model_name = curr_file+"_model.h5"


#%% DATASET CREATION

# if we would like to create a new set of SPOS, uncomment
# SPOS = []
# for i in range(100):
#     SPOS.append(get_random_n_spos(sposlimits,3))
# SPOS = np.array(SPOS)
# spos_ds = xr.DataArray(SPOS, dims=["sample", "source", "xyz"])
# spos_ds.to_netcdf('CONSISTENT_3SRC_SPOS_100_TESTSPEECH.nc')

import xarray as xr

for doares in [5,10,20]:
    for sigma in [5, 10, 15, 20]:
        hdf5_filename = "{}_doaresxy_{}_{}_sigmaxy_{}_{}_dataset.hdf5".format(os.path.split(curr_file)[-1],doares,doares,sigma,sigma)
        hdf5_path = os.path.join(hdf5_dir,hdf5_filename)
        # SPOS = np.loadtxt('CONSISTENT_SPOS_100k.csv')
        print("Now creating dataset: "+hdf5_path)
        SPOS = xr.load_dataarray('CONSISTENT_3SRC_SPOS_100k.nc').values
        
 
#  %%% Training dataset noise
        if MAKE_NEW_DATASET:
            
            spos = get_random_n_spos(sposlimits,3) # one random spos to get this started
            x, target, s, m, d = generate_STFT_noise_sample_2D_spos_mpos(limits=limits,
                                                                         spos=spos,
                                                                         mpos=mpos,
                                                                         duration=duration,
                                                                         absorbtion=absorbtion,
                                                                         order=order,
                                                                         Fs=Fs,
                                                                         doa_res_x=doares,
                                                                         doa_res_y=doares,
                                                                         sigma_x=sigma,
                                                                         sigma_y=sigma,
                                                                         sigma_res=False,
                                                                         stft_args=stft_args,
                                                                         shuffle=True)
            print("spos: \n",s)
            print("mpos: \n",m)
            print("doas: \n",d)
        
            num_samples = config.num_samples # samples to generate to the dataset
            samples_per_sim = x.shape[0]
            n_sims = int(num_samples/samples_per_sim)
            print(n_sims) # number of simulations to get num_samples samples
                
            #% GENERATE DATASET
            with tables.open_file(hdf5_path, mode='w') as hdf5_file:
                filters = tables.Filters(complevel=5, complib='blosc')
                
                x, t, s, m, d = generate_STFT_noise_sample_2D_spos_mpos(limits=limits,
                                                                        spos=spos,
                                                                        mpos=mpos,
                                                                        duration=duration,
                                                                        absorbtion=absorbtion,
                                                                        order=order,
                                                                        Fs=Fs,
                                                                        doa_res_x=doares,
                                                                        doa_res_y=doares,
                                                                        sigma_x=sigma,
                                                                        sigma_y=sigma,
                                                                        sigma_res=False,
                                                                        stft_args=stft_args,
                                                                        shuffle=True)
                ss = np.repeat(s,x.shape[0],axis=0); # spos is generated once per position, but x anr t are generated per frame, and there are multiple frames per position
        
                table_rax = hdf5_file.create_earray(hdf5_file.root, 'rax',
                                                    tables.Atom.from_dtype(x.dtype),
                                                    shape=(0, x.shape[1], x.shape[2]),
                                                    filters=filters,
                                                    expectedrows=len(x))
                
                table_spos = hdf5_file.create_earray(hdf5_file.root, 'spos',
                                                     tables.Atom.from_dtype(ss.dtype),
                                                     shape=(0, ss.shape[1]),
                                                     filters=filters,
                                                     expectedrows=len(ss))
                
                table_target = hdf5_file.create_earray(hdf5_file.root, 'target',
                                                        tables.Atom.from_dtype(t.dtype),
                                                        shape=(0, t.shape[1], t.shape[2]),
                                                        filters=filters,
                                                        expectedrows=len(t))
                
                tick = datetime.now()
                
                for n in tqdm(range(n_sims), position=0, leave=True):
                    ticky = datetime.now()
                    
                    x, t, s, m, d = generate_STFT_noise_sample_2D_spos_mpos(limits=limits,
                                                                            spos=SPOS[n],
                                                                            mpos=mpos,
                                                                            duration=duration,
                                                                            absorbtion=absorbtion,
                                                                            order=order,
                                                                            Fs=Fs,
                                                                            doa_res_x=doares,
                                                                            doa_res_y=doares,
                                                                            sigma_x=sigma,
                                                                            sigma_y=sigma,
                                                                            sigma_res=False,
                                                                            stft_args=stft_args,
                                                                            shuffle=True)
                    
                    ss = np.repeat(s,x.shape[0],axis=0);
                    
                    table_rax.append(x)
                    table_spos.append(ss)
                    table_target.append(t)
                    # table_azim.append(a)
                    # table_elev.append(e)
                
                    tocky = datetime.now()
                    diffy = tocky - ticky    # the result is a datetime.timedelta object
                    # print("set no. {} generated in {} seconds".format(n,diffy.total_seconds()))
            
            tock = datetime.now()
            diff = tock - tick    # the result is a datetime.timedelta object
            print("WHOLE dataset of {} sets of {} samples ({} samples in total) generated in {} seconds".format(n,x.shape[0],n*x.shape[0],diff.total_seconds()))

#  %%% Testing dataset speech

        if MAKE_NEW_DATASET_SPEECHTEST:
            SPOS_TESTSPEECH = xr.load_dataarray('CONSISTENT_3SRC_SPOS_100_TESTSPEECH.nc').values
            hdf5_speechtest_filename = "{}_doaresxy_{}_{}_sigmaxy_{}_{}_dataset_speechtest.hdf5".format(os.path.split(curr_file)[-1],doares,doares,sigma,sigma)
            hdf5_speechtest_path = os.path.join(hdf5_dir,hdf5_speechtest_filename)
            
            spos = get_random_n_spos(sposlimits,3) # one random spos to get this started

            x, target, s, m, d = generate_STFT_LIBRI_speech_sample_2D_spos_mpos(
                                                        DATA_DIR=DATA_DIR,                                                    
                                                        limits=limits,
                                                        spos=spos,
                                                        mpos=mpos,
                                                        duration=duration,
                                                        absorbtion=absorbtion,
                                                        order=order,
                                                        Fs=Fs,
                                                        doa_res_x=doares,
                                                        doa_res_y=doares,
                                                        sigma_x=sigma,
                                                        sigma_y=sigma,
                                                        sigma_res=False,
                                                        stft_args=stft_args,
                                                        shuffle=False)
            
            
            print("spos: \n",s)
            print("mpos: \n",m)
            print("doas: \n",d)
        
            num_samples = SPOS_TESTSPEECH.shape[0] # samples to generate to the dataset
            samples_per_sim = x.shape[0]
            n_sims = int(num_samples/samples_per_sim)
            print(n_sims) # number of simulations to get num_samples samples
                
            #% GENERATE DATASET
            with tables.open_file(hdf5_speechtest_path, mode='w') as hdf5_file:
                filters = tables.Filters(complevel=5, complib='blosc')
                
                x, t, s, m, d = generate_STFT_LIBRI_speech_sample_2D_spos_mpos(
                                                        DATA_DIR=DATA_DIR,                                                    
                                                        limits=limits,
                                                        spos=spos,
                                                        mpos=mpos,
                                                        duration=duration,
                                                        absorbtion=absorbtion,
                                                        order=order,
                                                        Fs=Fs,
                                                        doa_res_x=doares,
                                                        doa_res_y=doares,
                                                        sigma_x=sigma,
                                                        sigma_y=sigma,
                                                        sigma_res=False,
                                                        stft_args=stft_args,
                                                        shuffle=False)
                ss = np.repeat(s,x.shape[0],axis=0); # spos is generated once per position, but x anr t are generated per frame, and there are multiple frames per position
        
                table_rax = hdf5_file.create_earray(hdf5_file.root, 'rax',
                                                    tables.Atom.from_dtype(x.dtype),
                                                    shape=(0, x.shape[1], x.shape[2]),
                                                    filters=filters,
                                                    expectedrows=len(x))
                
                table_spos = hdf5_file.create_earray(hdf5_file.root, 'spos',
                                                     tables.Atom.from_dtype(ss.dtype),
                                                     shape=(0, ss.shape[1]),
                                                     filters=filters,
                                                     expectedrows=len(ss))
                
                table_target = hdf5_file.create_earray(hdf5_file.root, 'target',
                                                        tables.Atom.from_dtype(t.dtype),
                                                        shape=(0, t.shape[1], t.shape[2]),
                                                        filters=filters,
                                                        expectedrows=len(t))
                
                tick = datetime.now()
                
                for n in tqdm(range(SPOS_TESTSPEECH.shape[0]), position=0, leave=True):
                    ticky = datetime.now()
                    
                    x, t, s, m, d = generate_STFT_LIBRI_speech_sample_2D_spos_mpos(
                                                        DATA_DIR=DATA_DIR,                                                    
                                                        limits=limits,
                                                        spos=SPOS_TESTSPEECH[n],
                                                        mpos=mpos,
                                                        duration=duration,
                                                        absorbtion=absorbtion,
                                                        order=order,
                                                        Fs=Fs,
                                                        doa_res_x=doares,
                                                        doa_res_y=doares,
                                                        sigma_x=sigma,
                                                        sigma_y=sigma,
                                                        sigma_res=False,
                                                        stft_args=stft_args,
                                                        shuffle=False)
                    
                    ss = np.repeat(s,x.shape[0],axis=0);
                    
                    table_rax.append(x)
                    table_spos.append(ss)
                    table_target.append(t)
                    # table_azim.append(a)
                    # table_elev.append(e)
                
                    tocky = datetime.now()
                    diffy = tocky - ticky    # the result is a datetime.timedelta object
                    # print("set no. {} generated in {} seconds".format(n,diffy.total_seconds()))
            
            tock = datetime.now()
            diff = tock - tick    # the result is a datetime.timedelta object
            print("WHOLE dataset of {} sets of {} samples ({} samples in total) generated in {} seconds".format(n,x.shape[0],n*x.shape[0],diff.total_seconds()))



#%% TRAINING

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# directory = 'D:\CNNDOAMAP_DATASETS'

for conv_layers in [1,2,3,4]:
    print(conv_layers)
    # for hdf5_filename in os.listdir(hdf5_dir):
    for hdf5_filename in os.listdir(hdf5_dir):
        # if hdf5_filename.startswith('CNN_DOA_MAP_WANDB_CLEAN__3src_TESTBASELINE_COMPACT_SHUFFLETRAIN') and hdf5_filename.endswith("dataset.hdf5"): 
        if hdf5_filename == 'CNN_DOA_MAP_WANDB_CLEAN__3src_TESTBASELINE_COMPACT_SHUFFLETRAIN_doaresxy_10_10_sigmaxy_10_10_dataset.hdf5':
            print(hdf5_filename)
            # hdf5_filename = "{}_doaresxy_{}_{}_sigmaxy_{}_{}_dataset.hdf5".format(os.path.split(curr_file)[-1],doares,doares,sigma,sigma)
            hdf5_path = os.path.join(hdf5_dir,hdf5_filename)
            model_path = hdf5_path[:-13]+"_model_archtest_convlayers_{}_50epochs.h5".format(conv_layers)
            print(model_path)
            if os.path.isfile(model_path):
                print('ALREADY TRAINED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            else:
                print('model not trained, training')
                #  %%
                try:
                    print('trying')
                    num_of_samples = get_number_of_samples_in_dataset_STFT(hdf5_path)
                    
                    print('loading...')
                    tick = datetime.now()  
                    # xx,t = get_random_set_of_samples_STFT(hdf5_path,num_of_samples)
                    xx,t = get_all_set_of_samples_STFT(hdf5_path)
                    print('loaded!')
                    tock = datetime.now()
                    diff = tock - tick    # the result is a datetime.timedelta object
                    print("{} random samples taken in {} seconds".format(
                            num_of_samples,
                            diff.total_seconds()))
                    
                    #  %%% extract phase data
                    xphase = np.angle(xx)
                    
                    #  %%% split dataset
                    trtst_split = 0.9
                    trvl_split = 0.8
                    
                    tr_samples = int(num_of_samples*trtst_split)
                    trvl_samples = int(tr_samples*trvl_split)
                    
                    x1 = np.expand_dims(xphase, axis=-1)
                    
                    x_train = x1[0:trvl_samples]
                    t_train = t[0:trvl_samples]
                    
                    
                    x_val = x1[trvl_samples:tr_samples]
                    t_val = t[trvl_samples:tr_samples]
                    
                    
                    x_test = x1[tr_samples:-1]
                    t_test = t[tr_samples:-1]
                    
                    
                    #  %%% define model
                    
                    inputs = layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
                    
                    x = layers.Conv2D(config.conv1_units, kernel_size=config.conv1_kernel, strides=1, 
                                     data_format='channels_last', 
                                     padding="valid", 
                                     activation = config.conv1_activ, 
                                     name='Conv_first')(inputs)
                    
                    if conv_layers>1:
                        x = layers.Conv2D(config.conv2_units, kernel_size=config.conv2_kernel, strides=1, 
                                         padding="valid", 
                                         activation = config.conv2_activ, 
                                         name='Conv_second',
                                         data_format='channels_last')(x)
                    if conv_layers>2:
                        x = layers.Conv2D(config.conv3_units, kernel_size=config.conv3_kernel, strides=1, 
                                         padding="valid", 
                                         activation = config.conv3_activ, 
                                         name='Conv_third',
                                         data_format='channels_last')(x)
                    if conv_layers>3:
                        x = layers.Conv2D(config.conv3_units, kernel_size=config.conv3_kernel, strides=1, 
                                         padding="valid", 
                                         activation = config.conv3_activ, 
                                         name='Conv_third',
                                         data_format='channels_last')(x)                              
                    
                    x = layers.Dropout(config.conv_dropout)(x)
                    
                    a = layers.Dense(config.dense1_units,
                              activation = config.dense1_activ,
                              name='Dense_first_a')(x) # was 512
                    a = layers.Dense(config.dense2_units,
                              activation = config.dense2_activ,
                              name='Dense_second_a')(a) # was 512
                    a = layers.Dropout(config.dense_dropout)(a)
                    a = layers.Flatten()(a)
                    a = layers.Dense(t_train.shape[1]*t_train.shape[2],
                              activation = 'sigmoid',name='Output_a')(a)
                    out_a = layers.Reshape((t_train.shape[1],t_train.shape[2]))(a)
                    #  %%
                    # e = Dense(512,activation = 'relu',name='Dense_first_e')(x) # was 512
                    # e = Dense(512,activation = 'relu',name='Dense_second_e')(e) # was 512
                    # e = Dropout(0.5)(e)
                    # e = Flatten()(e)
                    # e = Dense(e_train.shape[1]*e_train.shape[2],
                    #           activation = 'sigmoid',name='Output_e')(e)
                    # out_e = Reshape((e_train.shape[1],e_train.shape[2]))(e)
                    
                    # model = Model(inputs=[inputs], outputs=[out_a, out_e])
                    model = keras.Model(inputs=inputs, outputs=out_a)
                    
                    print('compiling the model')
                    
                    optimizer = None
                    if config.optimizer_str == 'adam':
                        optimizer = keras.optimizers.Adam(lr=config.lr)
                    elif config.optimizer_str == 'nadam':
                        optimizer = keras.optimizers.Nadam(lr=config.lr)
                    elif config.optimizer_str == 'sgd':
                        optimizer = keras.optimizers.SGD(lr=config.lr)
                    
                    model.compile(loss=config.loss,
                                  optimizer=config.optimizer_str,
                                  metrics=['accuracy']) 
                    # plot_model(model, to_file=curr_file+'_model.png', show_shapes=True)
                    model.summary()
                    #model = keras.models.load_model('CONV2_OK_fast_model_2.h5')
                    
                    #  %%% fit model
                    
                    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
                    sess_config = tf.compat.v1.ConfigProto()
                    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.90
                    #  %%
                    with tf.device('/gpu:0'):
                        tick = datetime.now()    
                    
                        # lh = model.fit([x_train], [a_train, e_train],
                        lh = model.fit(x_train, t_train,
                                  batch_size=config.batch_size,
                                  epochs=50,
                                  verbose=1,
                                  shuffle=True,
                                  # validation_data=([x_val], [a_val,e_val]))
                                  # validation_data=([x_val], [t_val]),
                                  callbacks=[WandbCallback(save_model=False)])
                    
                    tock = datetime.now()
                    diff = tock - tick    # the result is a datetime.timedelta object
                    print("Training took {} seconds ({} hours)".format(diff.total_seconds(),diff.total_seconds()/3600))
                    
                    #  %%% save model
                    
                    # model.save(os.path.split(hdf5_path)[-1][:-5]+"_model.h5")
                    model.save(model_path)
    
                except:
                    pass
#%% TESTING

plt.close('all')

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
import xarray as xr

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import pandas as pd
import numpy as np

#%%% Define testing data paths
doares_sigma_string = 'doaresxy_10_10_sigmaxy_10_10'
model_filename = 'CNN_DOA_MAP_WANDB_CLEAN__3src_'+doares_sigma_string+'_model.h5'
model_path = os.path.join(hdf5_dir,model_filename)
testdataset_filename = 'CNN_DOA_MAP_WANDB_CLEAN__3src_TESTBASELINE_COMPACT_SHUFFLETRAIN_'+doares_sigma_string+'_dataset_speechtest.hdf5'
testdataset_path = os.path.join(hdf5_dir,testdataset_filename)

#%%% Load dataset samples

print('loading...')
tick = datetime.now()  
xx_all,t_all = get_all_set_of_samples_STFT(testdataset_path)

print('loaded!')
tock = datetime.now()
diff = tock - tick    # the result is a datetime.timedelta object

print("{} samples loaded in {} seconds".format(
        xx.shape[0],
        diff.total_seconds()))

#%%% Select samples for testing
# create list of indices, in sequences of k elements l elements apart from 0 to m
k = 20
l = 4000
m = xx_all.shape[0]
test_samples_idx = [item for sublist in [list(range(n, n+k)) for n in range(0,m,l)] for item in sublist]

xx = xx_all[test_samples_idx]
t = t_all[test_samples_idx]

#%%% Test Model

# extract phase data
xphase = np.angle(xx)

x1 = np.expand_dims(xphase, axis=-1)

x_test = x1
t_test = t


model = keras.models.load_model(model_path)


X_TEST = []
Y_TEST = []
Y_PRED = []
for i in tqdm(range(xx.shape[0])):

    raxx = np.array([x_test[i]])
    tt = np.array([t_test[i]])
    pt = model.predict(raxx) 

    X_TEST.append(raxx)
    Y_TEST.append(tt)
    Y_PRED.append(pt)
    # print('written')
        
ERRS, GT_COORDS, EST_COORDS = calculate_prediction_errors(PRED=Y_PRED,
                                                          GT=Y_TEST,
                                                          num_peaks=config.NUM_SIGNALS_TO_CONCATENATE,
                                                          doa_res_x=config.doa_res_x,
                                                          doa_res_y=config.doa_res_x)
    

ds = xr.Dataset()
# ds['X_TEST'] = (("sample", "time", "channel", "frequency", "complexreim"), 
                # np.transpose(np.array([np.array(X_TEST).real,np.array(X_TEST).imag]),(1,2,3,4,0)))
ds['Y_TEST'] = (("sample", "time", "elevation", "azimuth"), np.array(Y_TEST))
ds['Y_PRED'] = (("sample", "time", "elevation", "azimuth"), np.array(Y_PRED))

ds['ERRS'] = (("sample", "error"), np.array(ERRS))
# ds['GT_COORDS'] = (("sample", "source", "azel"), np.array(GT_COORDS))
ds['EST_COORDS'] = (("sample", "source", "azel"), np.array(EST_COORDS))


# wandb.log({'MEAN_ERRS': np.mean(np.array(ERRS))})
# ds.to_netcdf(hdf5_path[:-13]+'_eval_ds.nc')
# wandb.save(os.path.split(curr_file)[-1]+'_ds.nc')
            
#%%%% Visualize Model result

slider_sample, slider_time = cube_show_slider( np.array(Y_TEST), np.array(Y_PRED), title='zxc')

        
#%%% Test Baseline

# doa = pra.doa.MUSIC(mpos.T, config.fs, config.nfft, num_src=config.NUM_SIGNALS_TO_CONCATENATE, dim=3, n_grid=t_test.shape[-2]*t_test.shape[-1])
doa_SRP = pra.doa.SRP(mpos.T, config.fs, config.nfft, num_src=config.NUM_SIGNALS_TO_CONCATENATE, dim=3, n_grid=t_test.shape[-2]*t_test.shape[-1])
doa_MUSIC = pra.doa.MUSIC(mpos.T, config.fs, config.nfft, num_src=config.NUM_SIGNALS_TO_CONCATENATE, dim=3, n_grid=t_test.shape[-2]*t_test.shape[-1])
baseline_doa = {'SRP': doa_SRP, 'MUSIC': doa_MUSIC}

baseline_results_all_pos_samples = {}

for doa in baseline_doa:
    BASELINE_PRED_ALL_POS_SAMPLES = []
    for i in tqdm(range(xx.shape[0])):
        # print(i)
        try:
            # xxx = xx_test[i:i+num_samples_per_position,:,:]
            # print(xxx.shape)
            # doa.locate_sources(np.transpose(xxx,(1,2,0)), freq_bins=np.arange(20, 40))
            baseline_doa[doa].locate_sources(np.transpose(np.array([xx[i]]),(1,2,0)),  freq_range=[500.0, 4000.0])
            zxc = baseline_doa[doa].grid.regrid()
            BASELINE_PRED_ALL_POS_SAMPLES.append(zxc)
        except:
            print('exception')
    
    baseline_pred_f = np.expand_dims(np.transpose(np.array(BASELINE_PRED_ALL_POS_SAMPLES)[:,-1,:,:],(0,2,1)),axis=1) 
    # baseline_pred_f = np.repeat(baseline_pred_f,num_samples_per_position,axis=1)
    baseline_pred_f = np.vstack(baseline_pred_f)
    baseline_pred_f = np.expand_dims(baseline_pred_f,axis=1)
    baseline_results_all_pos_samples[doa] = baseline_pred_f
    
#%%%% Visualize Baseline
BASELINE_ERRS = {}
for b in baseline_results_all_pos_samples:
    slider_sample, slider_time = cube_show_slider(np.expand_dims(t_test,axis=1), baseline_results_all_pos_samples[b], title=b)
    BASELINE_GT = np.expand_dims(t_test,axis=1)
    BASELINE_PRED = baseline_results_all_pos_samples[b]
    ERRS, GT_COORDS, EST_COORDS = calculate_prediction_errors(PRED = BASELINE_PRED,
                                                              GT   = BASELINE_GT,
                                                              num_peaks = config.NUM_SIGNALS_TO_CONCATENATE,
                                                              doa_res_x = config.doa_res_x,
                                                              doa_res_y = config.doa_res_x)
    BASELINE_ERRS[b] = ERRS

#%% Test on custom spos

# create spos

# spos = np.array([[1,1,1],[5,4,1]])
spos = np.array([[5,4,1]])

# try smaller array aperture

mpos_orig = True
mpos_baseline_test_side = 0.1
if mpos_orig:
    mpos_baseline = mpos
else:
    th_side = mpos_baseline_test_side
    th = create_tetrahedron(th_side)
    mpos_center = config.mpos_center # fixed microphone array position
    mpos0 = th
    mpos_baseline = mpos0 + mpos_center # shift microphone array to a position within a room

# generate sample (noise or speech)
signal = 'speech'
if signal == 'speech':

    x_demo, t_demo, s_demo, m_demo, d_demo = generate_STFT_LIBRI_speech_sample_2D_spos_mpos(
                                                            DATA_DIR=DATA_DIR,                                                    
                                                            limits=limits,
                                                            spos=spos,
                                                            mpos=mpos_baseline,
                                                            duration=duration,
                                                            absorbtion=absorbtion,
                                                            order=config.order,
                                                            Fs=Fs,
                                                            doa_res_x=config.doa_res_x,
                                                            doa_res_y=config.doa_res_x,
                                                            sigma_x=config.sigma_x,
                                                            sigma_y=config.sigma_x,
                                                            sigma_res=False,
                                                            stft_args=stft_args)
    nsmpls = 16
    x_demo = x_demo[:nsmpls]
    t_demo = t_demo[:nsmpls]
    s_demo = s_demo[:nsmpls]
    m_demo = m_demo[:nsmpls]
    d_demo = d_demo[:nsmpls]
    
    
if signal == 'noise':
    x_demo, t_demo, s_demo, m_demo, d_demo = generate_STFT_noise_sample_2D_spos_mpos(limits=limits,
                                                            spos=spos,
                                                            mpos=mpos_baseline,
                                                            duration=duration,
                                                            absorbtion=absorbtion,
                                                            order=config.order,
                                                            Fs=Fs,
                                                            doa_res_x=config.doa_res_x,
                                                            doa_res_y=config.doa_res_x,
                                                            sigma_x=config.sigma_x,
                                                            sigma_y=config.sigma_x,
                                                            sigma_res=False,
                                                            stft_args=stft_args)

num_samples_per_position = x_demo.shape[0]
xxx = x_demo


#%%%% Model prediction custom spos
rax_demo = x_demo
raxx_demo = np.angle(rax_demo)
# raxx_demo = rax_demo

PT = []
for r in tqdm(raxx_demo):
    pt = model.predict(np.expand_dims(r,axis=(0,-1)))
    PT.append(pt)
PT = np.array(PT)
#%%%%% Visualize model prediction custom spos
slider_sample, slider_time = cube_show_slider( np.expand_dims(t_demo,axis=1), PT, title='zxc')

#%%%% Baseline prediction custom spos

# doa = pra.doa.MUSIC(mpos.T, config.fs, config.nfft, num_src=config.NUM_SIGNALS_TO_CONCATENATE, dim=3, n_grid=t_test.shape[-2]*t_test.shape[-1])
doa_SRP_manual_spos = pra.doa.SRP(mpos.T, config.fs, config.nfft, num_src=config.NUM_SIGNALS_TO_CONCATENATE, dim=3, n_grid=t_test.shape[-2]*t_test.shape[-1])
doa_MUSIC_manual_spos = pra.doa.MUSIC(mpos.T, config.fs, config.nfft, num_src=config.NUM_SIGNALS_TO_CONCATENATE, dim=3, n_grid=t_test.shape[-2]*t_test.shape[-1])
baseline_doa_manual_spos = {'SRP': doa_SRP_manual_spos, 'MUSIC': doa_MUSIC_manual_spos}

baseline_results_all_pos_samples_manual_spos = {}

for doa in baseline_doa:
    BASELINE_PRED_ALL_POS_SAMPLES = []
    
    X_TEST = []
    Y_TEST = []
    Y_PRED = []
    ERRS = []
    GT_COORDS = []
    EST_COORDS = []
    for i in tqdm(range(xxx.shape[0])):
        # print(i)
        try:
            # xxx = xx_test[i:i+num_samples_per_position,:,:]
            # print(xxx.shape)
            # doa.locate_sources(np.transpose(xxx,(1,2,0)), freq_bins=np.arange(20, 40))
            baseline_doa_manual_spos[doa].locate_sources(np.transpose(np.array([xxx[i]]),(1,2,0)),  freq_range=[500.0, 4000.0])
            zxc = baseline_doa_manual_spos[doa].grid.regrid()
            BASELINE_PRED_ALL_POS_SAMPLES.append(zxc)
        except:
            print('exception')
    
    baseline_pred_f = np.expand_dims(np.transpose(np.array(BASELINE_PRED_ALL_POS_SAMPLES)[:,-1,:,:],(0,2,1)),axis=1) 
    # baseline_pred_f = np.repeat(baseline_pred_f,num_samples_per_position,axis=1)
    baseline_pred_f = np.vstack(baseline_pred_f)
    baseline_pred_f = np.expand_dims(baseline_pred_f,axis=1)
    baseline_results_all_pos_samples_manual_spos[doa] = baseline_pred_f
    
#%%%%% Visualize Baseline custom spos
for b in baseline_results_all_pos_samples_manual_spos:
    slider_sample, slider_time = cube_show_slider(np.expand_dims(t_demo,axis=1), baseline_results_all_pos_samples_manual_spos[b], title=b)

#%%  TODO: write speech testing datasets for each doa_res, sigma configuration and test on these datasets
#%% TODO: test if training on noise stft phase and testing on speech stft phase is better than testing on speech stft magnitude

'''
maybe baseline results are so bad because of large array apperture
'''



#%%%% Baseline prediction custom spos with error estimation only SRP

# doa = pra.doa.MUSIC(mpos.T, config.fs, config.nfft, num_src=config.NUM_SIGNALS_TO_CONCATENATE, dim=3, n_grid=t_test.shape[-2]*t_test.shape[-1])
doa_SRP_manual_spos = pra.doa.SRP(mpos.T, config.fs, config.nfft, num_src=config.NUM_SIGNALS_TO_CONCATENATE, dim=3, n_grid=t_test.shape[-2]*t_test.shape[-1])
doa_MUSIC_manual_spos = pra.doa.MUSIC(mpos.T, config.fs, config.nfft, num_src=config.NUM_SIGNALS_TO_CONCATENATE, dim=3, n_grid=t_test.shape[-2]*t_test.shape[-1])
baseline_doa_manual_spos = {'SRP': doa_SRP_manual_spos}

baseline_results_all_pos_samples_manual_spos = {}

for doa in baseline_doa:
    BASELINE_PRED_ALL_POS_SAMPLES = []
    for i in tqdm(range(xxx.shape[0])):

        try:

            baseline_doa_manual_spos[doa].locate_sources(np.transpose(np.array([xxx[i]]),(1,2,0)),  freq_range=[500.0, 4000.0])
            zxc = baseline_doa_manual_spos[doa].grid.regrid()
            BASELINE_PRED_ALL_POS_SAMPLES.append(zxc)
        except:
            print('exception')
        
        coordinatesGT = peak_local_max(tt[0,:,:], min_distance=1,num_peaks=config.NUM_SIGNALS_TO_CONCATENATE)
        coordinatesEST = peak_local_max(pt[0,:,:], min_distance=1,num_peaks=config.NUM_SIGNALS_TO_CONCATENATE)
        
        tt = np.array([t_demo[i]])
        
        try:
            # print("Trying")
            gt_coords = centroid_coords_to_doas(coordinatesGT, config.doa_res_x, config.doa_res_y)
            est_coords = centroid_coords_to_doas(coordinatesEST, config.doa_res_x, config.doa_res_y) 
    
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
            errs = np.sort(err, axis=None)[0:config.NUM_SIGNALS_TO_CONCATENATE]
            
            while errs.shape[0] < 3:
                errs = np.append(errs,np.NaN)
            
            while est_coords.shape[0] < 3:
                est_coords = np.vstack((est_coords,np.array([np.NaN]*config.NUM_SIGNALS_TO_CONCATENATE)))
    
            ERRS.append(errs)        
            GT_COORDS.append(gt_coords)
            EST_COORDS.append(est_coords)
            X_TEST.append(raxx)
            Y_TEST.append(tt)
            Y_PRED.append(pt)
            # print('written')
        
    except:
        print("Not even trying..")
        err = float('NaN')
        print("est coords is zero")
    
    baseline_pred_f = np.expand_dims(np.transpose(np.array(BASELINE_PRED_ALL_POS_SAMPLES)[:,-1,:,:],(0,2,1)),axis=1) 
    # baseline_pred_f = np.repeat(baseline_pred_f,num_samples_per_position,axis=1)
    baseline_pred_f = np.vstack(baseline_pred_f)
    baseline_pred_f = np.expand_dims(baseline_pred_f,axis=1)
    baseline_results_all_pos_samples_manual_spos[doa] = baseline_pred_f
    
#%%%%% Visualize Baseline custom spos
for b in baseline_results_all_pos_samples_manual_spos:
    slider_sample, slider_time = cube_show_slider(np.expand_dims(t_demo,axis=1), baseline_results_all_pos_samples_manual_spos[b], title=b)


#%%

ERRS, GT_COORDS, EST_COORDS = calculate_prediction_errors(PRED,GT)
