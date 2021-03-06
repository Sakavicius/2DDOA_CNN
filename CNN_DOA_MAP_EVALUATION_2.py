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


import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
import xarray as xr

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import pandas as pd
import numpy as np

#%% CONSTANTS

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


#%% TESTING
#%%% Define testing data paths

DATA_DIR = "E:\DISSERTATION\PY\LIBRISpeech\dev-clean\LibriSpeech\dev-clean"
hdf5_dir = "D:\CNNDOAMAP_DATASETS"
SPOS_TESTSPEECH = xr.load_dataarray('CONSISTENT_3SRC_SPOS_100_TESTSPEECH.nc').values
# TESTFILE_ROOT = "CNN_DOA_MAP_WANDB_CLEAN__3src_TESTBASELINE_COMPACT_NOTSHUFFLE_"
# TESTFILE_ROOT = "CNN_DOA_MAP_WANDB_CLEAN__3src_"
TESTFILE_ROOT = 'CNN_DOA_MAP_WANDB_CLEAN__3src_TESTBASELINE_COMPACT_SHUFFLETRAIN_'

'CNN_DOA_MAP_WANDB_CLEAN__3src_TESTBASELINE_doaresxy_2_2_sigmaxy_5_5_dataset_speechtest'
'CNN_DOA_MAP_WANDB_CLEAN__3src_doaresxy_5_5_sigmaxy_10_10_model'
'CNN_DOA_MAP_WANDB_CLEAN__3src_doaresxy_5_5_sigmaxy_5_5_dataset'

#%%% Get model filenames
items_to_test = []
for filename in os.listdir(hdf5_dir):
    # print(filename)
    # if filename.endswith("_model.h5") and filename.startswith(TESTFILE_ROOT): 
    if '_model_archtest_convlayers_' in filename  and filename.startswith(TESTFILE_ROOT): 
        # if not filename.endswith('NOTSHUFFLED_model.h5'):
        # if not 'NOTSHUFFLE' in filename:
        model_filename = filename
        items_to_test.append(model_filename)


#%%% Iterate through all models in 'items_to_test'

ERRORS = {}

ERRDS = []

PLOTTING = False

for model_filename in tqdm(items_to_test):
    doares_sigma_string = model_filename[model_filename.find('doaresxy'):-9]
    doares = int(doares_sigma_string.split('_')[1])
    sigma = int(doares_sigma_string.split('_')[4])
    # model_filename = 'CNN_DOA_MAP_WANDB_CLEAN__3src_'+doares_sigma_string+'_model.h5'
    model_path = os.path.join(hdf5_dir,model_filename)
    # testdataset_filename = 'CNN_DOA_MAP_WANDB_CLEAN__3src_TESTBASELINE_'+doares_sigma_string+'_dataset_speechtest.hdf5'
    # testdataset_filename = TESTFILE_ROOT+doares_sigma_string+'_dataset_speechtest.hdf5'
    # testdataset_filename = 'CNN_DOA_MAP_WANDB_CLEAN__3src_TESTBASELINE_COMPACT_SHUFFLETRAIN_doaresxy_10_10_sigmaxy_10_10_model_archtest_convla_dataset_speechtest.hdf5'
    # testdataset_path = os.path.join(hdf5_dir,testdataset_filename)
    testdataset_path = 'D:\\CNNDOAMAP_DATASETS\\CNN_DOA_MAP_WANDB_CLEAN__3src_TESTBASELINE_COMPACT_SHUFFLETRAIN_doaresxy_10_10_sigmaxy_10_10_dataset.hdf5'
    # print(model_path)
    # print(testdataset_path)
    if (not os.path.isfile(model_path)):
        print('NOFILE!!!!!!!!!!!!!!!!!!!! '+model_filename)
        continue
    if (not os.path.isfile(testdataset_path)):
        print('NOFILE!!!!!!!!!!!!!!!!!!!! '+testdataset_path)
        continue
    print("TESTING MODEL {} with dataset {}".format(model_filename, testdataset_path))
    #  %%
    
    # doares_sigma_string = 'doaresxy_10_10_sigmaxy_10_10'
    # model_filename = 'CNN_DOA_MAP_WANDB_CLEAN__3src_'+doares_sigma_string+'_model.h5'
    # model_path = os.path.join(hdf5_dir,model_filename)
    # testdataset_filename = 'CNN_DOA_MAP_WANDB_CLEAN__3src_TESTBASELINE_'+doares_sigma_string+'_dataset_speechtest.hdf5'
    # testdataset_path = os.path.join(hdf5_dir,testdataset_filename)
        
    #  %%% Load dataset samples
    
    print('loading...')
    tick = datetime.now()  
    # xx_all,t_all = get_all_set_of_samples_STFT(testdataset_path)
    xx_all,t_all,spos_all = get_all_set_of_samples_STFT(testdataset_path,return_spos=True)
    
    # format spos to proper shape
    spos_all_shaped = dataset_spos_to_SPOS_rep(spos_all=spos_all,num_src=config.NUM_SIGNALS_TO_CONCATENATE)
    
    
    print('loaded!')
    tock = datetime.now()
    diff = tock - tick    # the result is a datetime.timedelta object
    
    print("{} samples loaded in {} seconds".format(
            xx_all.shape[0],
            diff.total_seconds()))
    
    #  %%% Select samples for testing
    # create list of indices, in sequences of k elements l elements apart from 0 to m
    k = 20
    l = 4000
    m = xx_all.shape[0]
    test_samples_idx = [item for sublist in [list(range(n, n+k)) for n in range(0,m,l)] for item in sublist]
    
    xx = xx_all[test_samples_idx]
    t = t_all[test_samples_idx]
    spos = spos_all_shaped[test_samples_idx]
    
    #  %%
    # sss = spos_all[]
    # s = spos_all[test_samples_idx]
    #  %%% Test Model
    
    # extract phase data
    xphase = np.angle(xx)
    
    x1 = np.expand_dims(xphase, axis=-1)
    
    x_test = x1
    t_test = t
    
    model = keras.models.load_model(model_path)
    
    X_TEST = []
    Y_TEST = []
    Y_PRED = []
    GT_DOAS = []
    
    print('MODEL PREDICTING')
    for i in tqdm(range(xx.shape[0])):
    
        raxx = np.array([x_test[i]])
        tt = np.array([t_test[i]])
        pt = model.predict(raxx) 
    
        X_TEST.append(raxx)
        Y_TEST.append(tt)
        Y_PRED.append(pt)
        GT_DOAS.append(get_DoAs_spos_mpos(spos[i],mpos))
        # print('written')
            
    ERRS, GT_COORDS, EST_COORDS, EST_COORDS_IDX = calculate_prediction_errors_GT_DOAS_fill(PRED=Y_PRED,
                                                            GT_DOAS=GT_DOAS,
                                                            num_peaks=config.NUM_SIGNALS_TO_CONCATENATE,
                                                            doa_res_x=doares,
                                                            doa_res_y=doares)
    # STFT STE for STE/ERR correlation plot
    stft_xx = np.angle(xx)
    energy_stft = np.sum(np.power(stft_xx,2),axis = (1,2))
    energy_ypred = np.sum(np.power(np.squeeze(np.array(Y_PRED)),2),axis = (1,2))
    
    # fig, ax1 = plt.subplots()
    # ax1.plot(energy_stft,label='STFT per-frame STE', color='g')
    # ax1.legend(loc=1)
    # ax2 = ax1.twinx()
    # ax2.plot(energy_ypred,label='prediction per-frame STE', color = 'b')
    # ax2.legend(loc=0)
    # ax3 = ax1.twinx()
    # ax3.plot(ERRS,label='prediction per-frame error', color = 'r')
    # ax3.legend(loc=2)
    
    # fig, ax1 = plt.subplots()
    # ax1.scatter(energy_ypred,ERRS[:,0])
    # ax1.scatter(energy_ypred,ERRS[:,1])
    # ax1.scatter(energy_ypred,ERRS[:,2])
    
    # ERRORS[doares_sigma_string]['CNN'] = ERRS   
    
    # ERRORS[doares_sigma_string] = {}

    # errds.
    
    ds = xr.Dataset()
    ds['X_TEST'] = (("sample", "time", "channel", "frequency", "complexreim"), 
                    np.transpose(np.array([np.array(X_TEST)[:,:,:,:,0].real,np.array(X_TEST)[:,:,:,:,0].imag]),(1,2,3,4,0)))
    ds['Y_TEST'] = (("sample", "time", "elevation", "azimuth"), np.array(Y_TEST))
    ds['Y_PRED'] = (("sample", "time", "elevation", "azimuth"), np.array(Y_PRED))
    ds['ENERGY_STFT'] = (("sample", "time", "STE"), np.expand_dims(np.expand_dims(np.array(energy_stft),1),1))
    ds['ENERGY_YPRED'] = (("sample", "time", "STE"), np.expand_dims(np.expand_dims(np.array(energy_ypred),1),1))
    ds['ERRS'] = (("sample", "source"), np.array(ERRS))
    ds['GT_COORDS'] = (("sample", "source", "azel"), np.array(GT_COORDS))
    ds['EST_COORDS'] = (("sample", "source", "azel"), np.array(EST_COORDS))
    ds['EST_COORDS_IDX'] = (("sample", "source", "azel"), np.array(EST_COORDS_IDX))
    ds['METHOD'] = 'CNN'
    ds['DOARES'] = doares
    ds['SIGMA'] = sigma
    ds['ARCH'] = model_filename.split('_')[-1].split('.')[0]
    ERRDS.append(ds)
    # wandb.log({'MEAN_ERRS': np.mean(np.array(ERRS))})
    # ds.to_netcdf(hdf5_path[:-13]+'_eval_ds.nc')
    # wandb.save(os.path.split(curr_file)[-1]+'_ds.nc')
                
    #  %%%% Visualize Model result
    if PLOTTING:
        slider_sample, slider_time = cube_show_slider( np.array(Y_TEST), np.array(Y_PRED), title='zxc')
    
    #  %%% Test Baseline
    
    # doa = pra.doa.MUSIC(mpos.T, config.fs, config.nfft, num_src=config.NUM_SIGNALS_TO_CONCATENATE, dim=3, n_grid=t_test.shape[-2]*t_test.shape[-1])
    doa_SRP = pra.doa.SRP(mpos.T, config.fs, config.nfft, num_src=config.NUM_SIGNALS_TO_CONCATENATE, dim=3, n_grid=t_test.shape[-2]*t_test.shape[-1])
    # doa_MUSIC = pra.doa.MUSIC(mpos.T, config.fs, config.nfft, num_src=config.NUM_SIGNALS_TO_CONCATENATE, dim=3, n_grid=t_test.shape[-2]*t_test.shape[-1])
    baseline_doa = {'SRP': doa_SRP}
    
    baseline_results_all_pos_samples = {}
    
    for doa in baseline_doa:
        BASELINE_PRED_ALL_POS_SAMPLES = []
        print(doa+' PREDICTING')
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


        BASELINE_GT = np.expand_dims(t_test,axis=1)
        BASELINE_PRED = baseline_results_all_pos_samples[doa]
        ERRS, GT_COORDS, EST_COORDS, EST_COORDS_IDX = calculate_prediction_errors_GT_DOAS_fill(PRED = BASELINE_PRED,
                                                                  GT_DOAS   = GT_DOAS,
                                                                  num_peaks = config.NUM_SIGNALS_TO_CONCATENATE,
                                                                  doa_res_x = doares,
                                                                  doa_res_y = doares)
        # BASELINE_ERRS[doa] = ERRS
        
        # ERRORS[doares_sigma_string]['BASELINE'] = BASELINE_ERRS
        
        ds = xr.Dataset()
        # ds['X_TEST'] = (("sample", "time", "channel", "frequency", "complexreim"), 
        #                 np.transpose(np.array([np.array(X_TEST).real,np.array(X_TEST).imag]),(1,2,3,4,0)))
        # ds['Y_TEST'] = (("sample", "time", "elevation", "azimuth"), np.array(Y_TEST))
        ds['Y_PRED'] = (("sample", "time", "elevation", "azimuth"), np.array(BASELINE_PRED))
        
        ds['ERRS'] = (("sample", "source"), np.array(ERRS))
        ds['GT_COORDS'] = (("sample", "source", "azel"), np.array(GT_COORDS))
        ds['EST_COORDS'] = (("sample", "source", "azel"), np.array(EST_COORDS))
        ds['EST_COORDS_IDX'] = (("sample", "source", "azel"), np.array(EST_COORDS_IDX))
        ds['METHOD'] = doa
        ds['DOARES'] = doares
        ds['SIGMA'] = sigma
        ERRDS.append(ds)
        
    #  %%%% Visualize Baseline
    BASELINE_ERRS = {}

    for b in baseline_results_all_pos_samples:
        if PLOTTING:
            slider_sample, slider_time = cube_show_slider(np.expand_dims(t_test,axis=1), baseline_results_all_pos_samples[b], title=b)
        
#%% COMBINE ERRDS to single dataset

ERRDSNOYPRED = [DS.drop('Y_PRED') for DS in ERRDS]

ERRDSNOYPRED = [DS.drop('X_TEST') for DS in [DS for DS in ERRDSNOYPRED if 'X_TEST' in DS.keys()]]
        

ERRDS_ALL = xr.concat(ERRDSNOYPRED,dim='sample')
ds.to_netcdf('ERRDS_ALL_ARCH_2.nc')

#%% 
ERRDS_ALL = xr.open_dataset('ERRDS_ALL_ARCH_2.nc')

#%%


df = ERRDS_ALL.where(ERRDS_ALL['SIGMA']==10,drop=True)[['ERRS','DOARES','METHOD','ARCH']].to_dataframe()
df.groupby('METHOD').boxplot(by='DOARES')
#%%
import seaborn as sns
for s in np.unique(ERRDS_ALL.SIGMA.values):
    df = ERRDS_ALL.where(ERRDS_ALL['SIGMA']==s,drop=True)[['ERRS','DOARES','METHOD','ARCH']].to_dataframe()
    fig, ax = plt.subplots(figsize=(3,4))
    z = sns.catplot(ax=ax,x='ARCH', y='ERRS',
                     hue="METHOD",
                    data=df, kind="box");
    z.axes[0][0].grid(color='gray', linestyle='--', linewidth=0.5, axis='y')
    # plt.title('MAE of Predicted Source DoA, sigma='+str(sigma))
    plt.title('Errors of Predicted Source DoA')
    plt.xlabel('Number of Convolutional Layers')
    plt.ylabel('DoA Prediction Error,  ??')
    z._legend.remove()
    # plt.savefig('arch.png')
    # plt.savefig('arch.pdf')
    # labels = [item.get_text() for item in z.axes[0][0].get_xticklabels()]
    # z.axes[0][0].set_xticklabels([str(-float(i)) for i in labels])
   
#%%
df.groupby('ARCH').mean().to_excel('doares_10_sigma_10_arch_123_MAE.xlsx')
   
#%%
fig, axs = plt.subplots(1,2,sharey=True,figsize=(8,4))
fig.suptitle("DoA Prediction Errors")
import seaborn as sns
sns.set_style("whitegrid")
for idx, m in enumerate(np.unique(ERRDS_ALL.METHOD.values)):
    print(idx)
    df = ERRDS_ALL.where(ERRDS_ALL['METHOD']==m,drop=True)[['ERRS','DOARES','SIGMA','ARCH']].to_dataframe()
    z = sns.boxplot(x='ARCH', y='ERRS',
                     hue="SIGMA",
                    data=df, ax=axs[idx]);
    axs[idx].set_ylim([0,180])
    axs[idx].set_xlabel('DoA map Resolution, ??')    
    axs[idx].set_ylabel('DoA Prediction Error, ??')
    axs[idx].title.set_text(m) # method name
    axs[idx].grid(color='gray', linestyle='--', linewidth=0.5, axis='y')
    # plt.title('MAE of Predicted Source DoA, method='+str(m))
    z.get_legend().remove()

#%%% PICKLE PREDICTED ERRORS

import pickle
import pandas as pd
with open('ERRORS_SHUFFLE_GT_DOA.p', 'wb') as fp:
    pickle.dump(ERRORS, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    #%% pickle ERRDS
import pickle
import pandas as pd
with open('ERRDS_NOTSHUFFLED.p', 'wb') as fp:
    pickle.dump(ERRDS, fp, protocol=pickle.HIGHEST_PROTOCOL)

#%% TEST MANUALLY ON CUSTOM SPOS
#%%% Create custom spos and mpos

# spos = np.array([[1,1,1],[5,4,1]])
spos = np.array([[3,5,2]])

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


#%%% Model prediction custom spos
rax_demo = x_demo
raxx_demo = np.angle(rax_demo)
# raxx_demo = rax_demo

PT = []
for r in tqdm(raxx_demo):
    pt = model.predict(np.expand_dims(r,axis=(0,-1)))
    PT.append(pt)
PT = np.array(PT)

#%%%% Visualize model prediction custom spos
slider_sample, slider_time = cube_show_slider( np.expand_dims(t_demo,axis=1), PT, title='zxc')

#%%%% Visualize ground truth, model prediction and STFT frame
stft_demo = np.abs(rax_demo)
slider_sample, slider_time = cube_show_slider_pcolormesh( np.expand_dims(stft_demo,axis=1), PT, title='zxc')

#%%%%% check the energy of stft and prediction frames
PTsqueezed = np.squeeze(PT)

energy_stft = np.sum(np.power(stft_demo,2),axis = (1,2))
energy_PT = np.sum(np.power(PTsqueezed,2),axis = (1,2))

fig, ax1 = plt.subplots()
ax1.plot(energy_stft,label='STFT per-frame STE', color='g')
ax1.legend(loc=1)
ax2 = ax1.twinx()
ax2.plot(energy_PT,label='prediction per-frame STE', color = 'r')
ax2.legend(loc=0)


#%% compare the energy with per-frame DoA error


gt_doa_demo = get_DoAs_spos_mpos(spos,mpos)
GT_DOAS_DEMO = [gt_doa_demo for i in range(PT.shape[0])]
Y_PRED_DEMO = [PT[i,:] for i in range(PT.shape[0])]


ERRS_DEMO, GT_COORDS_DEMO, EST_COORDS_DEMO, EST_COORDS_IDX_DEMO \
    = calculate_prediction_errors_GT_DOAS_fill(PRED=Y_PRED_DEMO,
                                                GT_DOAS=GT_DOAS_DEMO,
                                                num_peaks=1,
                                                doa_res_x=doares,
                                                doa_res_y=doares)


fig, ax1 = plt.subplots()
ax1.plot(energy_stft,label='STFT per-frame STE', color='g')
ax1.legend(loc=1)
ax2 = ax1.twinx()
ax2.plot(energy_PT,label='prediction per-frame STE', color = 'b')
ax2.legend(loc=0)
ax3 = ax1.twinx()
ax3.plot(ERRS_DEMO,label='prediction per-frame error', color = 'r')
ax3.legend(loc=2)






#%%% Baseline prediction custom spos

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
    
#%%%% Visualize Baseline custom spos
for b in baseline_results_all_pos_samples_manual_spos:
    slider_sample, slider_time = cube_show_slider(np.expand_dims(t_demo,axis=1), baseline_results_all_pos_samples_manual_spos[b], title=b)







#%% //////////////////////////////////


#%%  TODO: write speech testing datasets for each doa_res, sigma configuration and test on these datasets
#%% TODO: test if training on noise stft phase and testing on speech stft phase is better than testing on speech stft magnitude

'''
maybe baseline results are so bad because of large array apperture
'''

#%% //////////////////////////////////

#%% PLOT DOA HEATMAPS WITH GROUND TRUTH OVERLAY (FROM TARGET)

ERRS, GT_COORDS, EST_COORDS, EST_COORDS_IDX = calculate_prediction_errors_GT_DOAS(PRED = PRED,
                                                              GT_DOAS  = GT_DOAS,
                                                              num_peaks = config.NUM_SIGNALS_TO_CONCATENATE,
                                                              doa_res_x = config.doa_res_x,
                                                              doa_res_y = config.doa_res_x)

frame = 281
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.pcolormesh(PRED[frame,0,:,:])
for c in EST_COORDS_IDX[frame]:
    ax1.scatter(c[1]+0.5,c[0]+0.5,c=np.array([[1,0,0]]),marker='x',s=50)
for c in GT_COORDS_IDX[frame]:
    ax1.scatter(c[1]+0.5,c[0]+0.5,c=np.array([[1,0,1]]),marker='+',s=70)

# #   %% get real doas from sig gen spos
# doas = get_DoAs2(SPOS_TESTSPEECH[frame,:,:],np.array([mpos_center]))
# for c in doas/10:
#     ax1.scatter(c[1]+0.5,c[0]+0.5,c=np.array([[1,1,1]]),marker='o',s=70)

ax1.set_xlim([-0.5,PRED[frame,0,:,:].shape[-1]+0.5])
ax1.set_ylim([-0.5,PRED[frame,0,:,:].shape[0]+0.5])

new_tick_locations = np.arange(0,PRED[frame,0,:,:].shape[-1]+1,3)
ax1.set_xticks(new_tick_locations+0.5)
ax1.set_xticklabels(new_tick_locations)
new_tick_locations = np.arange(0,PRED[frame,0,:,:].shape[0]+1,3)
ax1.set_yticks(new_tick_locations+0.5)
ax1.set_yticklabels(new_tick_locations)

plt.grid()

new_tick_locations = np.arange(0,360+config.doa_res_x,config.doa_res_x*3)
ax2 = ax1.twiny()
ax2.set_xticks(new_tick_locations+0.5*config.doa_res_x)
ax2.set_xticklabels(new_tick_locations)
ax2.set_xlim(np.array(ax1.get_xlim())*config.doa_res_x)

new_tick_locations = np.arange(0,180+config.doa_res_x,config.doa_res_x*3)
ax3 = ax1.twinx()
ax3.set_yticks(new_tick_locations+0.5*config.doa_res_x)
ax3.set_yticklabels(new_tick_locations)
ax3.set_ylim(np.array(ax1.get_ylim())*config.doa_res_x)



#%%
    
import collections
CNN_MAE = collections.defaultdict(dict)
SRP_MAE = collections.defaultdict(dict)

for e in ERRORS:
    try:
        doares = e.split('_')[1]
        sigma = e.split('_')[-1]
    except:
        pass
    CNN_MAE[doares][sigma] = np.mean(ERRORS[e]['CNN'])
    SRP_MAE[doares][sigma] = np.mean(ERRORS[e]['BASELINE']['SRP'])
    
#%% PLOT PLOT AND BAR CHART of PREDICTION MAE per DOARES, fixed SIGMA

lists = sorted(SRP_MAE['10'].items()) # sorted by key, return a list of tuples
x, y = zip(*lists) # unpack a list of pairs into two tuples
x = [int(e) for e in x]
xs,ys = zip(*sorted(zip(x, y)))

df = pd.DataFrame()
df['doares'] = xs
df['SRP MAE'] = ys

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(xs, ys)
plt.grid()
ax1.set_xlim([xs[0],xs[-1]])
new_tick_locations = np.arange(xs[0],xs[-1]+5,5)
ax1.set_xticks(new_tick_locations)
ax1.set_xticklabels(new_tick_locations)

lists = sorted(CNN_MAE['10'].items()) # sorted by key, return a list of tuples
x, y = zip(*lists) # unpack a list of pairs into two tuples
x = [int(e) for e in x]
xs,ys = zip(*sorted(zip(x, y)))
ax1.plot(xs, ys)


df['CNN MAE'] = ys

df = df.set_index('doares')
df.plot(kind='bar')

#%% Errors dict to dataframe

with open('ERRORS_NOSHUFFLE_GT_DOA.p', 'rb') as pickle_file:
    ERRORS = pickle.load(pickle_file)

for e in ERRORS:
    try:
        doares = e.split('_')[1]
        sigma = e.split('_')[-1]
    except:
        pass
    CNN_MAE[doares][sigma] = ERRORS[e]['CNN']
    SRP_MAE[doares][sigma] = ERRORS[e]['BASELINE']['SRP']


def pretty(d, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('\t' * (indent+1))
#%% flatten data of single sigma multiple doares for boxplot vis
CNN_MAE_BOXPLOT_DATA = {}
for d in CNN_MAE['10']:
    CNN_MAE_BOXPLOT_DATA[d] = CNN_MAE['10'][d].flatten()


lists = sorted(CNN_MAE['10'].items()) # sorted by key, return a list of tuples
x, y = zip(*lists) # unpack a list of pairs into two tuples
x = [int(e) for e in x]
xs,ys = zip(*sorted(zip(x, y)))

df = pd.DataFrame()
df['doares'] = xs
df['CNN MAE'] = ys
df = df.set_index('doares')
plt.figure()
df.boxplot()


#%%

#%% DOA GROUND TRUTH FUCKERY

#%%% get GT DoAs from dataset - saved in spos_all in a weird manner
testdataset_path = 'D:\CNNDOAMAP_DATASETS\CNN_DOA_MAP_WANDB_CLEAN__3src_TESTBASELINE_doaresxy_10_10_sigmaxy_10_10_dataset_speechtest.hdf5'
xx_all,t_all,spos_all = get_all_set_of_samples_STFT(testdataset_path,return_spos=True)
# spos_all_reshaped = np.reshape(spos_all,(-1,3,3)) # reshape to same format as xx_all and t_all (set of value per sample)
# SPOS_TESTSPEECH = xr.load_dataarray('CONSISTENT_3SRC_SPOS_100_TESTSPEECH.nc').values

#%%% reshape spos_all to a correct format
SPOS_RESHAPED = dataset_spos_to_SPOS_rep(spos_all=spos_all,num_src=config.NUM_SIGNALS_TO_CONCATENATE)

#%%% take same elements as were taken for predictio 
spos = SPOS_RESHAPED[test_samples_idx]

#%%% obtain doas from gt source positions
DOAS = get_DoAs_spos_mpos(spos[sample_n],mpos)

#%%% plot predicted doa heatmap and gt source positions
ypred = np.array(Y_PRED) # Y_PRED is obtained from CNN pediction
doares = 10
sample_n = 0
plt.figure();
# plt.pcolormesh(t[sample_n]) # TARGET
plt.pcolormesh(ypred[sample_n,0,:,:])

for doa in DOAS:
    plt.scatter((doa[0]+180)/doares+0.5,(doa[1]+90)/doares+0.5)

#%% Plot spos and mpos in 3D space

plot_3D_points2(SPOS_RESHAPED[sample_n],mpos,mpos_center)
print(DOAS)