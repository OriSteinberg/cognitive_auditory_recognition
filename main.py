import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import hashlib
from glob import glob
# from IPython.core.debugger import set_trace
import sys
from importlib import reload
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
import matplotlib as mpl
# *************************************************************************************
# from get_binary_conv import load_data, get_data
from training import big_node
# *************************************************************************************

mpl.rcParams['pdf.fonttype'] = 42

np.random.seed(0)     # for reproducibility tests
torch.manual_seed(0)  # for reproducibility tests

collect = 'LL_HowTo_0DegreesSeparation'
modality = 'neuroscan'
# modality = 'dsi'

save_flag = True
num_predict = 1
hidden_size = 2
# hidden_size = 200
num_ch_output = 1
output_size = num_ch_output * num_predict

slow_opt_flag = False
# num_batch = int(1024) # bce
num_batch = int(5)
# num_epoch = 2400 #paper
num_epoch = 2

learning_rate = 1e-3
weight_decay = 0

file_path_net           = 'C:/Py_ws/DL/thesis/'            # XXX_path_to_net
file_path_name_get_data = 'C:/Py_ws/DL/thesis/get_binary_conv'  # XXX_path_and_name_to_get_data
num_context             = 1000
file_name_net           = 'binary_conv'
file_name_get_data      = 'get_binary_conv'
loss_type               = 'bce'

file_path_name_net = os.path.join(file_path_net, file_name_net)
sys.path.append(os.path.split(file_path_name_get_data)[0])
module = __import__(os.path.split(file_path_name_get_data)[1])
reload(module)
load_data = getattr(module, 'load_data')
get_data = getattr(module, 'get_data')

# ******************** gathering the data into lists ********************
subj_dir = "C:/Users/User/Documents/MATLAB/EEG_data/"
subj_folder_list = [subj_dir + subj for subj in sorted(os.listdir(subj_dir))]  # ['data/s1', 'data/s2']

file_path_name_audio_list = []
file_path_name_eeg_list = []
for subj_folder in subj_folder_list[:]:
    try:
        file_path_name_audio_list.append(sorted(glob(os.path.join(subj_folder, '*Envelope.*')))[-1])  # for real data
        file_path_name_eeg_list.append(sorted(glob(os.path.join(subj_folder, '*EEG*.*')))[-1])
    except:
        print('-- missing --')
        print(subj_folder)
print(file_path_name_audio_list)
print(file_path_name_eeg_list)

idx_keep_audioTime = np.sort(np.random.permutation(num_context)[:250])
dct_params = {'idx_keep_audioTime': idx_keep_audioTime}

################################################################################################################
stam = False
if stam:  # this is just foo foo
    audio, eeg, audio_unatt = load_data(file_path_name_audio_list[0], file_path_name_eeg_list[0]) # todo add all subjects

    print(audio.shape, eeg.shape, audio_unatt.shape)
    a = ~np.isnan(audio)
    fig, ax = plt.subplots()
    ax.stem(np.sum(a, axis=1))
    print(np.min(np.sum(a, axis=1)))

    # debug get data
    X, y, z_unatt = get_data(audio, eeg, audio_unatt, idx_sample=0, num_context=num_context, dct_params=dct_params)
    if X is not None:
        print(X.shape)
        print(y.shape)
        print(z_unatt)

        fig, ax = plt.subplots()
        ax.plot(X.data.numpy()[100].T)
        fig, ax = plt.subplots()
        ax.plot(y.data.numpy()[:100])

    # # Visualize differences
    # print(np.nanstd(X[:, 0, ].data.numpy(), axis=0))
    # fig, ax = plt.subplots(); ax.stem(np.nanmean(np.nanmean(eeg, axis=2), axis=0));
    # fig, ax = plt.subplots(); ax.stem(np.nanmean(np.nanstd(eeg, axis=2), axis=0));
    # fig, ax = plt.subplots(); ax.stem(np.nanstd(eeg, axis=2)[:, 26]);
    # fig, ax = plt.subplots(); ax.stem(np.nanstd(audio, axis=1));
    # fig, ax = plt.subplots(); ax.stem(np.nanstd(audio_unatt, axis=1));
    # fig, ax = plt.subplots(); ax.plot(audio[0][:500]); ax.plot(audio[-1][:500]);


    # Check availability of data after removing nan's
    eeg_1ch = np.squeeze(eeg[:, 0, :])

    num_dur = np.nansum(~np.isnan(eeg_1ch), axis=1)
    print(num_dur)
    print(np.where(num_dur < num_context))
    print(np.mean(num_dur[num_dur >= num_context] * 0.01))
    print(np.std(num_dur[num_dur >= num_context] * 0.01))

#####################################################################################################################

# # Required: Define all data splits
timestamp_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

# ******************** Load data ********************
eval_list = []
for file_path_name_audio, file_path_name_eeg in zip(file_path_name_audio_list, file_path_name_eeg_list):
    print(file_path_name_audio)
    audio, eeg, audio_unatt = load_data(file_path_name_audio, file_path_name_eeg)

    # exhaustive
    full_set = audio.shape[0]
    # full_set = 10 # debug, 4

    X, y, z_unatt = get_data(audio, eeg, audio_unatt, idx_sample=0, num_context=num_context, dct_params=dct_params)

    input_size = np.prod(X.shape[1:])
    print(input_size)
    for test in range(full_set):
        # for test in np.random.permutation(full_set).tolist(): #If running less than a full set of splits and want to see different test partitions
        train = sorted(list(set(range(full_set)) - set([test])))
        eval_list.append([train, [test], file_path_name_audio, file_path_name_eeg, input_size])

# Optional: Test stability of training
# Can the identical network with identical inputs recover the same performance with/without different random seeds
# during initialization/training/optimization?
# DEBUG for Stability check, take eval list, first item, copy N times these should be identical runs of the network
stability = True
if stability:
    eval_list = [eval_list[0] for i in range(len(eval_list))]  # todo why is this line

# # Required: Define how many of the data splits to actually run

n_splits = len(eval_list)
# n_splits = 5
# n_splits = 2
# n_splits = 3
n_splits = 1

random_seed_flag = True
# random_seed_flag = False
# ***************************************************************************
for idx_b in range(n_splits):
    timestamp = '%s_%s' % (timestamp_time, hashlib.md5((('').join(eval_list[idx_b][2] + eval_list[idx_b][3])).encode('utf')).hexdigest())
    file_path_save = 'C:/Py_ws/DL/thesis/model/'  # XXX_file_path_save_with_timestamp
    if not os.path.exists(file_path_save):
        os.makedirs(file_path_save)

    dct_params = {'idx_eeg':                 np.nan * np.ones(eeg.shape[1]),
                  'num_context':             num_context,
                  'num_predict':             num_predict,
                  'idx_split':               idx_b,
                  'timestamp':               timestamp,
                  'file_path_save':          file_path_save,
                  'file_path_name_get_data': file_path_name_get_data,
                  'save_flag':               save_flag,
                  'num_epoch':               num_epoch,
                  'file_path_name_net':      file_path_name_net,
                  'input_size':              eval_list[idx_b][4],
                  'hidden_size':             hidden_size,
                  'output_size':             output_size,
                  'num_batch':               num_batch,
                  'learning_rate':           learning_rate,
                  'weight_decay':            weight_decay,
                  'loss_type':               loss_type,
                  'num_ch_output':           num_ch_output,
                  'collect':                 collect,
                  'idx_keep_audioTime':      idx_keep_audioTime,
                  'random_seed_flag':        random_seed_flag,
                  'slow_opt_flag':           slow_opt_flag}

    train = eval_list[idx_b][0]  # train
    test = eval_list[idx_b][1]  # test
    file_path_name_audio = eval_list[idx_b][2]  # file_path_name_audio
    file_path_name_eeg = eval_list[idx_b][3]  # file_path_name_eeg
    big_node(train, test, file_path_name_audio, file_path_name_eeg, dct_params)
# ***************************************************************************


# # Create workflow
web = False
if web:
    wf = pe.Workflow(name="wf")
    for idx_b in range(n_splits):
        timestamp = '%s_%s' % (timestamp_time, hashlib.md5(
                                   (('').join(eval_list[idx_b][2] + eval_list[idx_b][3])).encode('utf')).hexdigest())

        file_path_save = 'C:/Py_ws/DL/thesis/model/'  # XXX_file_path_save_with_timestamp

        # Create the file_path_save here to avoid race conditions in the workflow
        if not os.path.exists(file_path_save):
            os.makedirs(file_path_save)

        # Remember, it is MUCH faster to submit lightweight arguments to a node than to submit the entire dataset.
        # That's why the dataset is loaded inside big_node.
        node_big = pe.Node(niu.Function(input_names=['train', 'test',
                                                     'file_path_name_audio',
                                                     'file_path_name_eeg',
                                                     'dct_params'],
                                        output_names=['outputs'],
                                        function=big_node),
                           name='big_node_%03d' % idx_b)

        dct_params = {'idx_eeg':                 np.nan * np.ones(eeg.shape[1]),
                      'num_context':             num_context,
                      'num_predict':             num_predict,
                      'idx_split':               idx_b,
                      'timestamp':               timestamp,
                      'file_path_save':          file_path_save,
                      'file_path_name_get_data': file_path_name_get_data,
                      'save_flag':               save_flag,
                      'num_epoch':               num_epoch,
                      'file_path_name_net':      file_path_name_net,
                      'input_size':              eval_list[idx_b][4],
                      'hidden_size':             hidden_size,
                      'output_size':             output_size,
                      'num_batch':               num_batch,
                      'learning_rate':           learning_rate,
                      'weight_decay':            weight_decay,
                      'loss_type':               loss_type,
                      'num_ch_output':           num_ch_output,
                      'collect':                 collect,
                      'idx_keep_audioTime':      idx_keep_audioTime,
                      'random_seed_flag':        random_seed_flag,
                      'slow_opt_flag':           slow_opt_flag}

        node_big.inputs.train = eval_list[idx_b][0]  # train
        node_big.inputs.test = eval_list[idx_b][1]  # test

        node_big.inputs.file_path_name_audio = eval_list[idx_b][2]  # file_path_name_audio
        node_big.inputs.file_path_name_eeg = eval_list[idx_b][3]  # file_path_name_eeg

        node_big.inputs.dct_params = dct_params
        wf.add_nodes([node_big])

    # # Optional: Test main processing function
    # ## Don't use nipype, just run the function
    # stats = big_node(train, [test], file_path_name_audio, file_path_name_eeg, dct_params)

    # # Required: Main Proc

    wf.config['execution']['crashdump_dir'] = 'C:/Py_ws/DL/thesis/crashdumpdir'  # XXX_path_to_crashdumpdir
    wf.base_dir = 'C:/Py_ws/DL/thesis'  # XXX_path_to_base_dir

    wf.config['execution']['parameterize_dirs'] = False
    wf.config['execution']['poll_sleep_duration'] = 10
    wf.config['execution']['job_finished_timeout'] = 30

    run_local_flag = True
    #run_local_flag = False

    if run_local_flag:
        eg = wf.run()
    else:
        # eg = wf.run('SLURM', plugin_args={'sbatch_args': '-p gpu --gres=gpu:tesla:2 --constraint=xeon-e5 --mem=15G'})
        eg = wf.run('SLURM', plugin_args={'sbatch_args': '--constraint=xeon-e5 --exclusive -O'})

print('Done successfully')

# # Optional: Look at network parameters from a saved output file¶
# ## Look at params
# module = import(os.path.split(file_path_name_net)[1]) reload(module) NN = getattr(module, 'NN')
#
# file_path_name_checkpoint = XXX_path_to_checkpoint
#
# model = NN(input_size, hidden_size, output_size) checkpoint = torch.load(file_path_name_checkpoint) model.load_state_dict(checkpoint['state_dict']) model.eval()
#
# ## a = list(model.parameters())
# ## [print(a[i]) for i in range(len(a))]
# p = nn.utils.parameters_to_vector(model.parameters())
#
# p[:100]
