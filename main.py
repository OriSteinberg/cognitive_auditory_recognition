import torch
import numpy as np
import os
import datetime
from glob import glob
import sys
from importlib import reload
import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42

np.random.seed(0)     # for reproducibility tests
torch.manual_seed(0)  # for reproducibility tests

Traning = True
Eval = True
save_flag = True

eval_modulo = 1
output_size = 1
hidden_size = 100

# batch_size = int(1024) # bce
batch_size = int(5)
num_epoch = 2400

learning_rate = 1e-3
weight_decay = 0

file_path_net           = 'C:/Py_ws/DL/thesis/'                 # XXX_path_to_net
file_path_name_get_data = 'C:/Py_ws/DL/thesis/get_binary_conv'  # XXX_path_and_name_to_get_data
num_context             = 1000
file_name_net           = 'binary_conv'
loss_type               = 'bce'

file_path_name_net = os.path.join(file_path_net, file_name_net)
sys.path.append(os.path.split(file_path_name_get_data)[0])
module = __import__(os.path.split(file_path_name_get_data)[1])
reload(module)
load_data = getattr(module, 'load_data')
get_data = getattr(module, 'get_data')

file_path_training = 'C:/Py_ws/DL/thesis/training'
sys.path.append(os.path.split(file_path_training)[0])
module = __import__(os.path.split(file_path_training)[1])
reload(module)
big_node = getattr(module, 'big_node')

# ******************** gathering the data into lists ********************
subj_dir = "C:/Users/User/Documents/MATLAB/EEG_data/"
subj_folder_list = [subj_dir + subj for subj in sorted(os.listdir(subj_dir))]
file_path_name_audio_list = []
file_path_name_eeg_list = []
for subj_folder in subj_folder_list[:1]:
    try:
        file_path_name_audio_list.append(sorted(glob(os.path.join(subj_folder, '*Envelope.*')))[-1])
        file_path_name_eeg_list.append(sorted(glob(os.path.join(subj_folder, '*EEG*.*')))[-1])
    except:
        print('-- missing --')
        print(subj_folder)
print(file_path_name_audio_list)
print(file_path_name_eeg_list)

idx_keep_audioTime = np.sort(np.random.permutation(num_context)[:250])
dct_params = {'idx_keep_audioTime': idx_keep_audioTime}

# ******************** Load data ********************
# this part is only for getting parameters from the data, we dont use the data itself
eval_list = []
for file_path_name_audio, file_path_name_eeg in zip(file_path_name_audio_list, file_path_name_eeg_list):
    audio, eeg, audio_unatt = load_data(file_path_name_audio, file_path_name_eeg)
    # eeg - [Trails, Channel, Time]   audio/audio_unatt - [Trails, Time]
    X, y = get_data(audio, eeg, audio_unatt, idx_sample=0, num_context=num_context, dct_params=dct_params)
    # x - [num_time(windows) * 2, channel + 1, num_context\np.size(idx_keep_audioTime)]
    # y - [num_time(windows) * 2, 1]

    full_set = audio.shape[0]  # num of trails
    # groups test suits when at each test we take the next time series
    eval_list.append([full_set, file_path_name_audio, file_path_name_eeg])

random_seed_flag = True
# random_seed_flag = False

for test_set in eval_list:
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    file_path_save = 'C:/Py_ws/DL/thesis/model/'  # path to save folder
    if not os.path.exists(file_path_save):
        os.makedirs(file_path_save)
    channel_num = eeg.shape[1]

    dct_params = {'idx_eeg':                 np.nan * np.ones(channel_num),
                  'num_context':             num_context,
                  'timestamp':               timestamp,
                  'file_path_save':          file_path_save,
                  'file_path_name_get_data': file_path_name_get_data,
                  'save_flag':               save_flag,
                  'num_epoch':               num_epoch,
                  'file_path_name_net':      file_path_name_net,
                  'hidden_size':             hidden_size,
                  'output_size':             output_size,
                  'batch_size':              batch_size,
                  'learning_rate':           learning_rate,
                  'weight_decay':            weight_decay,
                  'loss_type':               loss_type,
                  'idx_keep_audioTime':      idx_keep_audioTime,
                  'random_seed_flag':        random_seed_flag,
                  'eval_flag':               Eval,
                  'eval_modulo':             eval_modulo}

    trails_num = test_set[0]
    file_path_name_audio, file_path_name_eeg = test_set[1], test_set[2]
    import time
    ts = time.time()
    big_node(trails_num, file_path_name_audio, file_path_name_eeg, dct_params)
    te = time.time()
    print(te - ts)
