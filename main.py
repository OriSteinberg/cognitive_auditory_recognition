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
save_flag = False

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
    print(file_path_name_audio)
    audio, eeg, audio_unatt = load_data(file_path_name_audio, file_path_name_eeg)
    # eeg - [Trails, Channel, Time]   audio/audio_unatt - [Trails, Time]

    X, y = get_data(audio, eeg, audio_unatt, idx_sample=0, num_context=num_context, dct_params=dct_params)
    # x - [num_time(windows) * 2, channel + 1, num_context\np.size(idx_keep_audioTime)]
    # y - [num_time(windows) * 2, 1]

    full_set = audio.shape[0]             # num of trails
    data_prod_num = np.prod(X.shape[1:])  # num of product in each example  ((channel + 1) * num_context)
    # groups test suits when at each test we take the next time series
    for test in range(full_set):
        # If running less than a full set of splits and want to see different test partitions
        # for test in np.random.permutation(full_set).tolist():
        train = sorted(list(set(range(full_set)) - set([test])))
        eval_list.append([train, test, file_path_name_audio, file_path_name_eeg, data_prod_num])

# Optional: Test stability of training
# Can the identical network with identical inputs recover the same performance with/without different random seeds
# during initialization/training/optimization?
# DEBUG for Stability check, take eval list, first item, copy N times these should be identical runs of the network
stability = False
if stability:
    eval_list = [eval_list[0] for i in range(len(eval_list))]

# how many of the data splits to actually run
n_splits = len(eval_list)

random_seed_flag = True
# random_seed_flag = False

for idx_b in range(n_splits):
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    file_path_save = 'C:/Py_ws/DL/thesis/model/'  # path to save folder
    if not os.path.exists(file_path_save):
        os.makedirs(file_path_save)
    channel_num = eeg.shape[1]

    dct_params = {'idx_eeg':                 np.nan * np.ones(channel_num),
                  'num_context':             num_context,
                  'idx_split':               idx_b,
                  'timestamp':               timestamp,
                  'file_path_save':          file_path_save,
                  'file_path_name_get_data': file_path_name_get_data,
                  'save_flag':               save_flag,
                  'num_epoch':               num_epoch,
                  'file_path_name_net':      file_path_name_net,
                  #'data_prod_num':           data_prod_num,
                  'hidden_size':             hidden_size,
                  'output_size':             output_size,
                  'batch_size':              batch_size,
                  'learning_rate':           learning_rate,
                  'weight_decay':            weight_decay,
                  'loss_type':               loss_type,
                  'idx_keep_audioTime':      idx_keep_audioTime,
                  'random_seed_flag':        random_seed_flag,
                  'eval_modulo':             eval_modulo}

    train, test = eval_list[idx_b][0], eval_list[idx_b][1]
    file_path_name_audio, file_path_name_eeg = eval_list[idx_b][2], eval_list[idx_b][3]
    big_node(train, test, file_path_name_audio, file_path_name_eeg, dct_params)
