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
from CNN_3D_FIELD_FUNCTIONS import *
#  %%
curr_file = os.path.splitext(__file__)[0]
path = os.path.dirname(__file__)

plt.close('all')
# sys.exit("Stopped for manual execution") 

#  %% CONSTANTS

MAKE_TESTSPEECH_SPOS = False
MAKE_NEW_DATASET = False
MAKE_NEW_DATASET_SPEECHTEST = False
CREATE_STFTS = False
CREATE_FIELDS = False

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
    epochs=100,#epochs = 5,
    resolution = 0.25,
    sigma = 0.25,
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
hdf5_dir = "D:\CNNDOAMAP_DATASETS"
hdf5_filename = "{}_src_{}_resolution_{}_sigma_{}_dataset.hdf5".format(config.NUM_SIGNALS_TO_CONCATENATE,
                                                                             os.path.split(curr_file)[-1],
                                                                             config.resolution,
                                                                             config.sigma)
hdf5_path = os.path.join(hdf5_dir,hdf5_filename)
model_name = curr_file+"_model.h5"

speech_dry_files_subdir = 'speech_dry_audio_100_files_5_seconds'
noise_dry_files_subdir = 'noise_dry_audio_100_files_5_seconds'
#  %%
import xarray as xr
#%%
# if we would like to create a new set of SPOS, uncomment
if MAKE_TESTSPEECH_SPOS:
    SPOS = []
    for i in range(100):
        SPOS.append(get_random_n_spos(sposlimits,3))
    SPOS = np.array(SPOS)
    spos_ds = xr.DataArray(SPOS, dims=["sample", "source", "xyz"])
    spos_ds.to_netcdf('CONSISTENT_TEST_3SRC_SPOS_100.nc')


#%% Create speech dataset
# from speech dataset dir select n_spos files
CREATE_DRY_SPEECH_DATASET = False
if CREATE_DRY_SPEECH_DATASET:
    DATA_DIR = "E:\DISSERTATION\PY\LIBRISpeech\dev-clean\LibriSpeech\dev-clean"
    
    n_files = 1000
    n_seconds = 5
    
    filenames = []
    for path, subdirs, files in os.walk(DATA_DIR):
        for name in files:
            if name.endswith('.flac'):
                filenames.append(os.path.join(path, name))
                print(os.path.join(path, name))
    
    
    SECONDS = []
    for rf in tqdm(filenames):
        with open(rf, 'rb') as f:
            rfdata, rfsr = sf.read(f)
        
        seconds = rfdata.shape[0]/rfsr
        SECONDS.append(seconds)
    SECONDS = np.array(SECONDS)
    
    sns.distplot(SECONDS,kde=False)
    # cut n_seconds from each of the files
    # if file is shorter, select next file
    # duration is n_seconds; noise signal was 0.1 second. 
    # the distribution of the duration of the files is:
    '''
                        ,                                                           
                        ,,,                                                         
                        ,,,                                                         
                        ,*,                                                         
       ,.,.             (,,                                                         
                        ,,,,,                                                       
                        ,,,,,                                                       
       (*#*            (,,,,,,,.                                                    
                        ,,,,,,,,,(,                                                 
                        ,,,,,,,,,,*                                                 
       ,,#,           / ,,,,,,,,,,,*                                                
                      . ,,,,,,,,,,,,#                                               
                      ,,,,,,,,,,,,,,,.                                              
       (. .          ,,,,,,,,,,,,,,,,                                               
                     *,,,,,,,,,,,,,,,,,                                             
                      ,,,,,,,,,,,,,,,,,,,#,                                         
       ,  ,         / ,,,,,,,,,,,,,,,,,,,,,..(                                      
                      ,,,,,,,,,,,,,,,,,,,,,,,,,,,/,,                                
                   .  ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, .                            
       .                   @                                         #              
                   0       5       10      15       20      30      35       %& 
    '''
    #%%
    # so we might select files over 5 seconds.
    FILES = []
    n_speech_files = 100
    i = 0
    while len(FILES)<n_speech_files:
                
        rf = filenames[i]
        with open(rf, 'rb') as f:
            rfdata, rfsr = sf.read(f)
        
        seconds = rfdata.shape[0]/rfsr
        if (seconds > n_seconds):
            FILES.append(rf)
        
        i+=1
        
    #%%
    # take 5 seconds from each file and write all files to dataset dir, dry speech subdir

    SECONDS = []
    file_counter = 0
    for file in tqdm(FILES):
        with open(file, 'rb') as f:
            rfdata, rfsr = sf.read(f)
        
        seconds = rfdata.shape[0]/rfsr
        
        speech_filepath = os.path.join(hdf5_dir,speech_dry_files_subdir,str(file_counter)+'.wav')
        sf.write(speech_filepath,rfdata[0:int(n_seconds*rfsr)],rfsr)
        
        file_counter += 1
        
        SECONDS.append(seconds)
    SECONDS = np.array(SECONDS)
    
    plt.figure()
    sns.distplot(SECONDS,kde=False)


#%% CREATE NOISE SIGNALS, save to wav



n_files = 100
for i in tqdm(range(n_files)):
    n_seconds = 5

    sig = np.random.random(n_seconds*config.fs) ### GENERATE THE FUCKING NOISE SIGNAL HERE ###
    sig = np.array(sig, dtype=float) # Convert signal (of list type) to float array
    
    noise_filepath = os.path.join(hdf5_dir,noise_dry_files_subdir,str(i)+'.wav')
    sf.write(noise_filepath,sig,config.fs)

#%% CREATE SPEECH STFTS

# only needs to be recreated if stft params or speech files change

# open 5 second files created in last step and load to RAM

def create_stft_ds_from_files(input_dir=None,
                              SPOS_TEST=None,
                              limits=None,
                              mpos=None,
                              absorbtion=None,
                              order=None,
                              Fs=None,
                              stft_args=None,
                              ds_stft_filename=None,
                              ds_spos_filename=None,
                              train=False,
                              snr=None):
    SIGNALS = []
    for file in os.listdir(input_dir):
        print(file)
        if file.endswith('.wav'):
            rfdata, rfsr = sf.read(os.path.join(input_dir,file))
            SIGNALS.append(rfdata)
    SIGNALS = np.array(SIGNALS)

    num_sigfiles = SIGNALS.shape[0]
    
    if train:
        divider = 8   
    else:
        divider = 1
    
    STFTS = []
    for n in tqdm(range(int(len(SPOS_TEST)/divider))):
        
        if train:
            signal = np.array([SIGNALS[np.random.randint(num_sigfiles)][0:int(fs*duration)],
                               SIGNALS[np.random.randint(num_sigfiles)][0:int(fs*duration)],
                               SIGNALS[np.random.randint(num_sigfiles)][0:int(fs*duration)]])

        else:
            idx1 = int((num_sigfiles/3*0 + n)%num_sigfiles)
            idx2 = int((num_sigfiles/3*1 + n)%num_sigfiles)
            idx3 = int((num_sigfiles/3*2 + n)%num_sigfiles)
            signal = np.array([SIGNALS[idx1],
                               SIGNALS[idx2],
                               SIGNALS[idx3]])

        x = generate_STFT_signal_sample_spos_mpos(signal=signal,
                                            limits=limits,
                                            spos=SPOS_TEST[n],
                                            mpos=mpos,
                                            absorbtion=absorbtion,
                                            order=order,
                                            Fs=Fs,
                                            stft_args=stft_args,
                                            shuffle=False,
                                            snr=snr)
        STFTS.append(x)
    STFTS = np.array(STFTS)
    
    ds_stft = xr.Dataset()
    STFTS_magphase = np.transpose(np.array([np.abs(STFTS),np.angle(STFTS)]),(1,2,3,4,0))
    ds_stft['STFTS'] = (("spos", "time", "channel", "frequency", "magphase"), STFTS_magphase)

    ds_stft.to_netcdf(ds_stft_filename)
    
    
    # SS = []
    # for spos in tqdm(SPOS_TESTSPEECH):    
    #     ss = np.repeat(spos,x.shape[0],axis=0);
    #     SS.append(ss)
    # SS = np.array(SS)
    SS = np.repeat(np.expand_dims(SPOS_TESTSPEECH,1),x.shape[0],axis=1)
    
    ds_spos = xr.Dataset()
    ds_spos['SS'] = (("spos", "time", "source", "xyz"), SS)

    ds_spos.to_netcdf(ds_spos_filename)
    
    return ds_stft, ds_spos


SPOS_TESTSPEECH = xr.load_dataset('CONSISTENT_TEST_3SRC_SPOS_100.nc').to_array().values[0]
#%%
if CREATE_STFTS:
    stft_calc_input_dir = os.path.join(hdf5_dir,speech_dry_files_subdir) # SPEECH
    ds_stft_filename = 'REVISION_TEST_DATASET_3SRC_STFTS_SPEECH_100.nc' # SPEECH
    ds_spos_filename = 'REVISION_TEST_DATASET_3SRC_SPOS_SPEECH_100.nc' # SPEECH

    # stft_calc_input_dir = os.path.join(hdf5_dir,noise_dry_files_subdir) # NOISE
    # ds_stft_filename = 'REVISION_TEST_DATASET_3SRC_STFTS_NOISE_100.nc' # NOISE
    # ds_spos_filename = 'REVISION_TEST_DATASET_3SRC_SPOS_NOISE_100.nc' # NOISE
    
    create_stft_ds_from_files(input_dir=stft_calc_input_dir,
                              SPOS_TEST=SPOS_TESTSPEECH,
                              limits=limits,
                              mpos=mpos,
                              absorbtion=absorbtion,
                              order=order,
                              Fs=Fs,
                              stft_args=stft_args,
                              ds_stft_filename=ds_stft_filename,
                              ds_spos_filename=ds_spos_filename,
                              train=False)

#%%

SS = xr.load_dataset('REVISION_DATASET_3SRC_SPOS_100k.nc')
SS = SS.to_array().values[0]

STFTS = xr.load_dataset('REVISION_DATASET_3SRC_STFTS_NOISE_100k.nc')
x = STFTS.to_array().values[0][0]
#%%
if CREATE_FIELDS:
    for resolution in [5,10,20]:
        for sigma in [5,10,15,20]:
    
            FIELDS = []
            for i in tqdm(range(int(len(SS)/8))):
                # fields = generate_spos_field(spos=spos[0],
                #                              sposlimits=sposlimits,
                #                              resolution=resolution,
                #                              sigma=sigma,
                #                              x=x)
                fields, doas = generate_doas_and_field(
                            SS[i][0],mic_array_center,x,
                            resolution_x=resolution,resolution_y=resolution,
                            sigma_x=sigma,sigma_y=sigma,sigma_res=False)
                FIELDS.append(fields)
            FIELDS = np.array(FIELDS)
            
            ds = xr.Dataset()
            ds['FIELDS'] = (("spos", "time", "azim", "elev"), FIELDS)
            # ds_filename = 'DATASET_2SRC_FIELDS_TESTSPEECH_100_resolution_{}_sigma_{}.pickle'.format(resolution,sigma) # SPEECH
            ds_filename = 'REVISION_DATASET_3src_100k_resolution_{}_sigma_{}.pickle'.format(resolution,sigma) # NOISE
            # ds.to_netcdf(ds_filename,format='NETCDF3_64BIT ')
            with open(ds_filename, 'wb') as handle:
                pickle.dump(ds, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
#%% test generated fields

sigma = 10
resolution = 5

# ds_filename = 'DATASET_FIELDS_TESTSPEECH_100_resolution_{}_sigma_{}.pickle'.format(resolution,sigma)
ds_filename = 'REVISION_DATASET_3src_100k_resolution_{}_sigma_{}.pickle'.format(resolution,sigma)
with open(ds_filename, "rb") as input_file:
    FIELDS = pickle.load(input_file)
FIELDS = FIELDS.to_array().values[0]
#%%
spos_idx = 45
frame = 4
fig, ax = plt.subplots(1)
ax.pcolormesh(FIELDS[spos_idx,frame])
#%%
ss = SS[spos_idx,frame]/resolution
for s in ss:
    ax.scatter(s[1],s[0],s[2],color='b')


#%% TRAINING

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# directory = 'D:\CNNDOAMAP_DATASETS'
#%%
cur_dir = os.path.split(curr_file)[0]
for fields_filename in os.listdir(cur_dir):
    if fields_filename.startswith('REVISION_DATASET_3src') and fields_filename.endswith(".pickle"): 
#  %%
        # hdf5_filename = 'CNN_3D_FIELD_1src_3_resolution_0.25_0.25_sigmaxy_0.5_0.5_dataset.hdf5'
        print(fields_filename)
        #  %%
        resolution, sigma = extract_resolution_and_sigma_values_from_filename(fields_filename,reso_string='resolution',sigma_string='sigma')
            
        
        # hdf5_filename='dataset_2src_CNN_3D_FIELD_2src_100epochs_resolution_0.25_0.25_sigmaxy_0.5_0.5_dataset.hdf5'
        stft_filename = 'REVISION_DATASET_3SRC_STFTS_NOISE_100k.nc'
        stft_path = os.path.join(cur_dir,stft_filename)
        model_path = fields_filename[:-7]+"_model_100epochs.h5"
        print(model_path)
        if os.path.isfile(model_path):
            print('ALREADY TRAINED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        else:
            print('model not trained, training')
            #  %%
            try:
                                
                STFTS = xr.load_dataset('REVISION_DATASET_3SRC_STFTS_NOISE_100k.nc')
                STFTS = np.vstack(STFTS.to_array().values[0])
                
                ds_filename = 'REVISION_DATASET_3src_100k_resolution_{}_sigma_{}.pickle'.format(int(resolution),int(sigma))
                with open(ds_filename, "rb") as input_file:
                    FIELDS = pickle.load(input_file)
                
                FIELDS = np.vstack(FIELDS.to_array().values[0])
                
                t = FIELDS
                
                
                num_of_samples = STFTS.shape[0]
                #  %%% extract phase data
                # xphase = np.angle(xx)
                xphase = STFTS[:,:,:,1]
                
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
                
                x = layers.Conv2D(config.conv2_units, kernel_size=config.conv2_kernel, strides=1, 
                                 padding="valid", 
                                 activation = config.conv2_activ, 
                                 name='Conv_second',
                                 data_format='channels_last')(x)
                
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
# #%% TEST LOADED TARGET AND SPOS
# field = t[0,:,:,:]

# fig = plt.figure()
# ax = fig.add_subplot(111,projection='3d')

# colors = plt.cm.plasma(field.flatten())
# alpha = field.flatten()



# norm_alpha = rescale_linear(alpha,0,1)
# colors[:,3] = norm_alpha


# field_xrange = [sposlimits[0][0],sposlimits[1][0]]
# field_yrange = [sposlimits[0][1],sposlimits[1][1]]
# field_zrange = [sposlimits[0][2],sposlimits[1][2]]

# resolution_x = resolution
# resolution_y = resolution
# resolution_z = resolution

# x = np.arange(field_xrange[0],field_xrange[1],resolution*2)   # coordinate arrays -- make sure they contain 0!
# y = np.arange(field_yrange[0],field_yrange[1],resolution*2)
# z = np.arange(field_zrange[0],field_zrange[1],resolution*2)
# xf, yf, zf = np.meshgrid(x,y,z)

# ax.scatter(xf, yf, zf, c=colors, s=20)
# ax.scatter(xf, yf, zf, c=field.flatten(), s=20)
# plt.show()
                
                
                
                 #  %%% fit model
 #  %%               
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
                sess_config = tf.compat.v1.ConfigProto()
                sess_config.gpu_options.per_process_gpu_memory_fraction = 0.90
                #  %%
                with tf.device('/gpu:0'):
                    tick = datetime.now()    
                
                    # lh = model.fit([x_train], [a_train, e_train],
                    lh = model.fit(x_train, t_train,
                              batch_size=config.batch_size,
                              epochs=config.epochs,
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
#  %%     
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

for signal in ['speech']:    
    print(signal)

    if signal == 'speech':
        signal_str = 'SPEECH'
    elif signal == 'noise':
        signal_str = 'NOISE'


    # Load STFTs
    xxr = xr.load_dataset('REVISION_TEST_DATASET_3SRC_STFTS_'+signal_str+'_100.nc')
    xphase = xxr.to_array().values[0,:,:,:,:,1]
    x_test = np.expand_dims(xphase, axis=-1)
    
    # Load spos
    # SS = xr.load_dataset('DATASET_2SRC_SPOS_TEST'+signal_str+'_100.nc').to_array().values[0]
    SS =   xr.load_dataset('REVISION_TEST_DATASET_3SRC_SPOS_'+signal_str+'_100.nc').to_array().values[0]

    
    for filename in os.listdir(hdf5_dir):
    # for filename in ['dataset_2src_CNN_3D_FIELD_2src_100epochs_resolution_0.25_0.25_sigmaxy_0.5_0.5_model_100epochs.h5']:
        'CNN_DOA_MAP_WANDB_CLEAN__3src_TESTBASELINE_COMPACT_SHUFFLETRAIN_doaresxy_5_5_sigmaxy_5_5_model.h5'
        if filename.startswith('CNN_DOA_MAP_WANDB_CLEAN__3src_TESTBASELINE_COMPACT_SHUFFLETRAIN_') and filename.endswith("_model.h5"): 
        # if filename=='dataset_2src_CNN_3D_FIELD_2src_100epochs_resolution_0.25_0.25_sigmaxy_0.5_0.5_model_100epochs.h5': 
            print(filename)
            
            # Load model
            # doares_sigma_string = 'doaresxy_10_10_sigmaxy_10_10'
            # model_filename = 'CNN_3D_FIELD_2src_3_resolution_0.25_0.25_sigmaxy_0.5_0.5_model.h5'
            model_filename = filename
            model_path = os.path.join(hdf5_dir,model_filename)
            if not os.path.isfile(model_path):
                print('No model file')
                continue

            
            resolution, sigma = extract_resolution_and_sigma_values_from_filename(model_filename,
                                                                                  reso_string='doaresxy',
                                                                                  sigma_string='sigmaxy')
            
            # Load test fields
            
            # testdataset_filename = model_filename.replace('model.h5','dataset'+signal_str+'.hdf5')
            # testdataset_filename = 'CNN_3D_FIELD_1src_CLEAN_resolution_0.25_0.25_sigmaxy_0.5_0.5_dataset_speech.hdf5'
            # testdataset_filename = 'CNN_3D_FIELD_1src_3_resolution_0.25_0.25_sigmaxy_0.5_0.5_dataset.hdf5'
                                
            # fields_ds_filename = 'DATASET_2SRC_FIELDS_TESTNOISE_100_resolution_{:g}_sigma_{:g}.pickle'.format(resolution,sigma)            
            # # fields_ds_filename =   'DATASET_2SRC_FIELDS_TESTNOISE_100_resolution_0.25_sigma_0.5.pickle'
            # if not os.path.isfile(fields_ds_filename):
            #     print('No test dataset file')
            #     continue
            
            # with open(fields_ds_filename, "rb") as input_file:
            #     Y_TEST = pickle.load(input_file)

            model = keras.models.load_model(model_path)

            X_TEST = []
            # Y_TEST = []
            Y_PRED = []
            PRED_SPOS = []
            GT_FIELD_SPOS = []
            GT_SPOS = SS

            # GT_SPOS.append(np.array([ss[i] for ptt in pt]))
            
            pred_centers = []
            PRED_CENTERS = []
            
            for i in tqdm(range(xphase.shape[0]),position=0,leave=True):
            
                raxx = np.array([x_test[i]])
                # tt = np.array([t_test[i]])
                # pt = model.predict(raxx) 
                pt = model.predict(x_test[i]) 
            
                X_TEST.append(raxx)
                # Y_TEST.append(tt)
                Y_PRED.append(pt)
                
                # cc = []
                # CC = []
                # for pptt in pt:
                #     C, c, X, l = get_blob_centers_coordinates(field=pptt,
                #                                        resolution_clust=resolution,
                #                                        thrdiv=[2,3,4],
                #                                        n_clusters=2,
                #                                        return_mean=True)
                #     c = c[:,[1,0,2]] # because function actually return in field axis order, and we need the normal axis order
                #     cc.append(c)
                #     CC.append(C)
                # pred_centers.append(cc)
                # PRED_CENTERS.append(CC)
                
         

            ds = xr.Dataset()
            
            ds['X_TEST'] = (("sample", "time", "channel", "frequency", "complexmagphase"), 
                            xxr.to_array().values[0])
            # ds['Y_TEST'] = (("sample", "time", "x", "y", "z"), Y_TEST.to_array().values[0])
            ds['Y_PRED'] = (("sample", "time", "azim", "elev"), np.array(Y_PRED))
            
            # ds['PRED_SPOS'] = (("sample", "time", "source", "xyz"), np.array(pred_centers))
            # ds['PRED_IDX'] = (("sample", "time", "thresh", "source", "xyz"), np.array(PRED_CENTERS))
            # ds['GT_FIELD_SPOS'] = (("sample", "time", "xyz"), np.array(GT_FIELD_SPOS))
            ds['GT_SPOS'] = (("sample", "time", "source", "azimelev"), np.array(GT_SPOS))
            
            
            #  %% global dataset variables
            ds['resolution'] = (("sample", "time"), np.tile(resolution,np.array(Y_PRED).shape[0:2]))
            ds['sigma'] = (("sample", "time"), np.tile(sigma,np.array(Y_PRED).shape[0:2]))
            ds['numsrc'] = (("sample", "time"), np.tile(2,np.array(Y_PRED).shape[0:2]))

            evaluation_ds_file_path = os.path.join(hdf5_dir,"REVISION_3_EVAL_3src_resolution_{}_sigma_{}_{}.nc".format(resolution,sigma,signal_str))
            ds.to_netcdf(evaluation_ds_file_path)
            

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

#%% 3D blob center finding 
# k-means clustering of multiple blobs 

sigma_clust = 0.25
resolution_clust = 0.25
custom_spos = np.array([[1.2,2.1,1.3],[3.1,3.2,2.3],[2.1,1.2,2.3]])
field = create_field(spos=custom_spos,sposlimits=sposlimits,resolution=resolution_clust,sigma=sigma_clust)
ax = plot_3D_field(field)
plt.legend()
#%%

# or we can use a predicted field
frame = 33
field = fields[frame,0,:,:,:]
custom_spos = np.array([GT_SPOS[frame]])
#%%




def get_blob_centers_coordinates(field=None,
                                 resolution_clust=None,
                                 thrdiv=[2,3,4],
                                 n_clusters=3,
                                 return_mean=True):
    '''
    returns 
        CENTERS - the element index of field where the blob centers are
        centers - metric (scaled by resolution) of the blob centers
        X - field points above last threshold,
        labels - cluster labels
    '''
    CENTERS, X, labels = get_blob_centers_multiple_thr(field=field,thrdiv=thrdiv,n_clusters=n_clusters)
    centers = np.array(CENTERS)*resolution_clust
    # centers = centers[:,[1,0,2]]
    # CENTERS = CENTERS[:,[1,0,2]]
    # cluster_center = np.mean(np.mean(centers,axis=0),axis=0) # in case we wish only one center
    if return_mean:
        centers = np.mean(centers,axis=0)
    return CENTERS, centers, X, labels

CENTERS, centers, X, labels = get_blob_centers_coordinates(field=field,
                                                   resolution_clust=resolution_clust,
                                                   thrdiv=[2,3,4],
                                                   n_clusters=3,
                                                   return_mean=True)

# ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
ax = plot_3D_field(field)
ax.scatter(X[:, 0], X[:, 1], X[:, 2],c=labels.astype(float), edgecolor='k',label='thresholded 3D field elements')

s = custom_spos/resolution_clust
ax.scatter(s[:, 1], s[:, 0], s[:, 2],c='g', edgecolor='k',s=100,label='GT source positions')

m = mpos/resolution_clust
ax.scatter(m[:, 1], m[:, 0], m[:, 2],c='r', edgecolor='k',s=100,label='microphone positions')

for center in CENTERS:
    ax.scatter(center[:, 1], center[:, 0], center[:, 2],c=[1,0,1], edgecolor='k',s=50,label='cluster centers')

plt.legend()
#%%

# plt_title = 'dummy field with 3 sources, source position location estimation \n using k-means clustering'
# plt.title(plt_title)
# plt.savefig(os.path.join(hdf5_dir,plt_title.replace("\n", "")+'.pdf'))


#%% gradient descent local maxima finding
# TODO


#%% check 3D gradient 
gradient = np.gradient(field)

for g in gradient:
    plot_3D_field(g)
    
    
    
#%% check dataset fields
dataset_filename = 'CNN_3D_FIELD_1src_CLEAN_test_speech_auto_resolution_0.25_0.25_sigmaxy_1_1_dataset_speech.hdf5'
dataset_path = os.path.join(hdf5_dir,dataset_filename)

xx_all,t_all,spos_all = get_all_set_of_samples_STFT(dataset_path,return_spos=True)
#%%
plot_3D_field(t_all[1000])

#%% test input feature STE and thresholding
# xx,t = get_all_set_of_samples_STFT(hdf5_path)

xphase = np.angle(xx)
xabs = np.abs(xx)
#%%%

XSTE = []
for x in xabs:
    XSTE.append(np.sum(np.power(x,2)))
    
XSTE = np.array(XSTE)
#%%
plt.figure()
sns.distplot(XSTE)
