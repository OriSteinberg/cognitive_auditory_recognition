import torch
import numpy as np
import os
from glob import glob
import sys
from importlib import reload


np.random.seed(0)     # for reproducibility tests
torch.manual_seed(0)  # for reproducibility tests
random_seed_flag = True

Traning      = True
Eval         = True
save_flag    = True
random_idx   = True
ONE_MAT_FILE = True

eval_modulo = 50
output_size = 1
hidden_size = 100
speaker_num = 1
hidden_channels = 64

batch_size   = int(128)  # int(1024)
num_epoch    = 300  # 2400
num_context  = 1000
idx_keep_num = 250

learning_rate = 1e-3
weight_decay  = 0.1

bade_adddr              = 'C:/my_programs/python_projects/thesis/'
# bade_adddr              = '/mnt/dsi_vol1/users/steinbo/code/'
file_path_net           = bade_adddr
file_path_name_get_data = bade_adddr + 'get_binary_conv'  # XXX_path_and_name_to_get_data
file_path_training      = bade_adddr + 'training'
subject_dir             = bade_adddr + 'data/two_audio_selective/'
file_path_save          = bade_adddr + 'results/'  
# file_name_net           = 'binary_conv'
file_name_net           = 'binary_conv_2'
loss_type               = 'bce'
file_path_name_net      = os.path.join(file_path_net, file_name_net)
subj_folder_list        = [subject_dir + subj for subj in sorted(os.listdir(subject_dir))]

sys.path.append(os.path.split(file_path_name_get_data)[0])
module = __import__(os.path.split(file_path_name_get_data)[1])
reload(module)
load_data = getattr(module, 'load_data')
get_data = getattr(module, 'get_data')

sys.path.append(os.path.split(file_path_training)[0])
module = __import__(os.path.split(file_path_training)[1])
reload(module)
big_node = getattr(module, 'big_node')

sys.path.append(os.path.split(file_path_name_net)[0])
module = __import__(os.path.split(file_path_name_net)[1])
reload(module)  # handle case of making changes to the module- forces reload
NN = getattr(module, 'NN')

# ******************** gathering the data into lists ********************
ONE_AUDIO = False if speaker_num > 1 else True
file_path_name_audio_list = []
file_path_name_eeg_list = []
for subj_folder in subj_folder_list:
    try:
        if ONE_MAT_FILE:
            file_path_name_audio_list.append(sorted(glob(os.path.join(subj_folder, '*s.*')))[-1])
            file_path_name_eeg_list.append(sorted(glob(os.path.join(subj_folder, '*s.*')))[-1])
        else:
            file_path_name_audio_list.append(sorted(glob(os.path.join(subj_folder, '*Envelope.*')))[-1])
            file_path_name_eeg_list.append(sorted(glob(os.path.join(subj_folder, '*EEG*.*')))[-1])
    except:
        print('-- missing --')
        print(subj_folder)
print(file_path_name_audio_list)
print(file_path_name_eeg_list)

if random_idx:
    idx_keep_audioTime = np.sort(np.random.permutation(num_context)[:idx_keep_num])
else:
    idx_keep_audioTime = np.linspace(1, num_context, idx_keep_num, dtype=np.int32)
dct_params = {'idx_keep_audioTime': idx_keep_audioTime}

# ******************** Load data ********************
# this part is only for getting parameters from the data, we dont use the data itself
eval_list = []

audio, eeg, audio_unatt = load_data(file_path_name_audio_list[0])
# eeg - [Trails, Channel, Time]   audio/audio_unatt - [Trails, Time]
x, y = get_data(audio, eeg, audio_unatt, idx_sample=0, num_context=num_context, dct_params=dct_params, ONE_AUDIO_PER_SAMPLE=ONE_AUDIO)
# x - [num_time(windows) * 2, channel + 1, num_context\np.size(idx_keep_audioTime)]
# y - [num_time(windows) * 2, 1] two sets - one good and one bad
print(f'shape of sample - {x.size()}')
set_size = x.size(0)
try:
    example_num = audio.shape[0]  # num of trails
    channel_num = eeg.shape[1]
except:
    example_num = len(audio)
    channel_num = eeg[0].shape[0]
    
for file_path_name_audio, file_path_name_eeg in zip(file_path_name_audio_list, file_path_name_eeg_list):
    # groups test suits when at each test we take the next time series
    eval_list.append([file_path_name_audio, file_path_name_eeg])


from itertools import product
parameters = dict(
    loss_func=['margin', 'bce'],  # CrossEntropy
    norm=[False, True],
    lr_lst=[1e-3, 1e-4],
    data_augmen=[False],
    hidden_lst=[50]
)
param_val = [v for v in parameters.values()]

for test_set in eval_list:
    if not os.path.exists(file_path_save):
        os.makedirs(file_path_save)
    for loss_type, NORM, learning_rate, data_augmentation, hidden_size in product(*param_val):
        print(loss_type, NORM, learning_rate, data_augmentation, hidden_size)
        dct_params = {'num_context':             num_context,
                      'file_path_save':          file_path_save,
                      'file_path_name_get_data': file_path_name_get_data,
                      'file_path_name_net':      file_path_name_net,
                      'save_flag':               save_flag,
                      'num_epoch':               num_epoch,
                      'hidden_size':             hidden_size,
                      'output_size':             output_size,
                      'batch_size':              batch_size,
                      'learning_rate':           learning_rate,
                      'weight_decay':            weight_decay,
                      'loss_type':               loss_type,
                      'idx_keep_audioTime':      idx_keep_audioTime,
                      'random_seed_flag':        random_seed_flag,
                      'eval_flag':               Eval,
                      'eval_modulo':             eval_modulo,
                      'idx_keep_num':            idx_keep_num,
                      'channel_num':             channel_num,
                      'speaker_num':             speaker_num,
                      'norm_data':               NORM,
                      'hidden_channels':         hidden_channels,
                      'data_augmentation':       data_augmentation}

        import time
        ts_all = time.time()
        big_node(NN, test_set, dct_params)
        print(f'total time - {time.time() - ts_all}')
