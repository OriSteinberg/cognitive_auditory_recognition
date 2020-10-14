#####################################################
# # Required: Define the main processing function
#####################################################
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import datetime
import time
import os
import sys
from importlib import reload
import hashlib
import re
import nipype


def big_node(train, test, file_path_name_audio, file_path_name_eeg, dct_params):
    """Process data and make predictions.
    1. Unpack parameters, define model, define data
    2. Training loop
    3. Evaluation
    4. Save

    Arguments
    ---------
    train : list, Integer list of training parts
    test : list, Integer test part
    file_path_name_audio : string, Full path and name of the audio mat file
    file_path_name_eeg : string, Full path and name of the eeg mat file
    dct_params: dict, Collection of auxillary parameters
    """
    ################################################################
    #      Unpack parameters, define model, define data
    ################################################################
    # Setup the dnn, and create the monolithic block of data that will be used for training.

    def closs(x, y):
        xbar = torch.mean(x)
        ybar = torch.mean(y)
        num = 1. / x.numel() * torch.dot(x - xbar, y - ybar)
        denom = torch.std(x) * torch.std(y)
        return -num / denom

    num_context        = dct_params['num_context']
    num_predict        = dct_params['num_predict']
    num_epoch          = dct_params['num_epoch']
    idx_eeg            = dct_params['idx_eeg']
    save_flag          = dct_params['save_flag']
    file_path_save     = dct_params['file_path_save']
    file_path_name_net = dct_params['file_path_name_net']
    input_size         = dct_params['input_size']
    hidden_size        = dct_params['hidden_size']
    output_size        = dct_params['output_size']
    num_batch          = dct_params['num_batch']
    learning_rate      = dct_params['learning_rate']
    weight_decay       = dct_params['weight_decay']
    loss_type          = dct_params['loss_type']
    collect            = dct_params['collect']
    idx_split          = dct_params['idx_split']
    random_seed_flag   = dct_params['random_seed_flag']
    slow_opt_flag      = dct_params['slow_opt_flag']

    if random_seed_flag:
        np.random.seed(idx_split)
        torch.manual_seed(idx_split)
    else:
        np.random.seed(0)
        torch.manual_seed(0)

    torch.backends.cudnn.deterministic = True

    # Load and preprocess the data
    file_path_name_get_data = dct_params['file_path_name_get_data']
    sys.path.append(os.path.split(file_path_name_get_data)[0])
    module = __import__(os.path.split(file_path_name_get_data)[1])
    reload(module)
    get_data = getattr(module, 'get_data')
    load_data = getattr(module, 'load_data')

    # Comment out in order to have the same val set and therefore the same train set between runs
    # train = np.asarray(train)[np.random.permutation(len(train))].tolist()
    if 1:
        valset = train[-2:]
        print(valset)
        train = train[:-2]
        print(train)
    else:
        valset = []

    # path to folder containing the class.py module
    sys.path.append(os.path.split(file_path_name_net)[0])
    module = __import__(os.path.split(file_path_name_net)[1])
    reload(module)  # handle case of making changes to the module- forces reload
    NN = getattr(module, 'NN')

    model = NN(input_size, hidden_size, output_size)

    num_val = len(valset)
    num_tr = len(train)
    params = model.state_dict()

    if loss_type == 'mse':
        loss_fn = nn.MSELoss(size_average=True)  # True = MSE vs False = sum squared
    elif loss_type == 'corr':
        loss_fn = closs
    elif loss_type == 'bce':
        loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if False:  # torch.cuda.is_available():
        cuda_flag = True
        model.cuda()
        print('Using CUDA')
    else:
        print('No CUDA')
        cuda_flag = False

    loss_history = np.nan * np.zeros(num_epoch)
    loss_val_history = np.nan * np.zeros(num_epoch)
    model.train()  # Turn on dropout, batchnorm
    # model.eval()

    audio, eeg, audio_unatt = load_data(file_path_name_audio, file_path_name_eeg, train=train)

    X_all, y_all = torch.tensor([]), torch.tensor([])

    for idx_sample in train[:2]:
        X, y, z_unatt = get_data(audio, eeg, audio_unatt=audio_unatt,
                                 idx_sample=idx_sample, num_context=num_context, dct_params=dct_params)
        if X is not None:
            X_all = torch.cat((X_all, X), dim=0)
            y_all = torch.cat((y_all, y), dim=0)
    print(X_all.shape)

    # Outside the loop to only form conv matrix once
    Xval, yval, z_unatt = get_data(audio, eeg, audio_unatt=audio_unatt,
                                   idx_sample=valset[0], num_context=num_context, dct_params=dct_params)

    ################################################################
    #              Training loop
    ################################################################
    # Iterate over the dataset a fixed number of times or until an early stopping condition is reached.
    # Randomly select a new batch of training at each iteration

    example_val_y = np.nan # todo delete
    example_val_z_unatt = np.nan
    example_val_yhat = np.nan  # todo delete
    idx_sample_list = np.nan * np.ones(num_epoch)
    idx_sample = train[0]  # Initialize to the first training part
    idx_train = 0
    early_stop_flag = False
    early_stop_counter = 0
    start = time.perf_counter()
    t_start = datetime.datetime.now()
    print(t_start)
    while (idx_train < num_epoch) and (not early_stop_flag):  # todo for loop
        if np.mod(idx_train, num_epoch / 10) == 0:  # todo Y??
            print('epoch %d ' % idx_train)
            end = time.perf_counter()
            t_end = datetime.datetime.now()
            print('Time per epoch %2.5f ticks' % ((end - start) / (num_epoch / 10)))
            print((t_end - t_start) / (num_epoch / 10))
            start = time.perf_counter()
            t_start = datetime.datetime.now()
            print(t_start)

        idx_keep = np.random.permutation(X_all.data.size(0))[:num_batch]
        idx_keep = torch.from_numpy(idx_keep).type('torch.LongTensor')
        X_audio = X_all[idx_keep]
        y = y_all[idx_keep]
        # X_audio = X_audio + Variable(0. * torch.randn(X_audio.shape))    # Data augmentation via noise

        # print('-- got data--')
        if X_audio is not None:
            model.zero_grad()
            # print('-pre forward-')
            if cuda_flag:
                y = y.cuda()
                output = model.forward(X_audio.cuda())
            else:
                output = model.forward(X_audio)

            loss = loss_fn(output.view(-1), y.view(-1))

            optimizer.zero_grad()
            # print('opt zeroed')
            loss.backward()
            # print('loss.backward done')
            optimizer.step()
            loss_flag = 1

            if cuda_flag:
                loss = loss.cpu()
                output = output.cpu()
                y = y.cpu()
            loss_history[idx_train] = loss_flag * loss.data.numpy()

            if False:  # loss_history[idx_train] < 0.09:
                early_stop_flag = True
                print("early_stop!")

            # Check validation set performance
            if (len(valset) > 0) and (np.mod(idx_train, 1) == 0):  # 50   # todo remove mod
                # print('--- val check ---')
                model.eval()

                idx_keep = np.sort(np.random.permutation(Xval.data.size(0))[:num_batch])
                idx_keep = torch.from_numpy(idx_keep).type('torch.LongTensor')
                X = Xval[idx_keep]
                y = yval[idx_keep]

                if cuda_flag:
                    y_att = model.forward(X.cuda())
                else:
                    y_att = model.forward(X)

                if cuda_flag:
                    stat_1 = loss_fn(y_att.view(-1), y.cuda().view(-1))
                else:
                    stat_1 = loss_fn(y_att.view(-1), y.view(-1))
                    stat_1 = stat_1.data.numpy()
                loss_val_history[idx_train] = stat_1
                model.train()

                example_val_y = y.cpu().data.numpy()
                example_val_yhat = y_att.cpu().data.numpy()

            idx_train = idx_train + 1

    print('-- done training --')
    print(datetime.datetime.now())

    ################################################################
    #              Evaluation
    ################################################################
    # Test on the train set, then test on the test set.

    if True:
        example_tr_y = []
        example_tr_yhat = []
        example_tr_unatt = []
        for idx_tr in train[:1]:
            X, y, z_unatt = get_data(audio, eeg, audio_unatt=audio_unatt,
                                     idx_sample=idx_tr, num_context=num_context, dct_params=dct_params)
            if X is not None:
                model.eval()
                if cuda_flag:
                    y_att = model.forward(X.cuda())
                else:
                    y_att = model.forward(X)
                example_tr_y.append(y.cpu().data.numpy())
                example_tr_yhat.append(y_att.cpu().data.numpy())
            if z_unatt is None:
                example_tr_unatt.append(np.array(np.nan))
            else:
                example_tr_unatt.append(z_unatt.data.numpy())

    if True:
        X, y, z_unatt = get_data(audio, eeg, audio_unatt=audio_unatt,
                                 idx_sample=test[0], num_context=num_context, dct_params=dct_params)

        if X is not None:
            model.eval()
            if cuda_flag:
                y_att = model.forward(X.cuda())
            else:
                y_att = model.forward(X)
            example_te_y = y.cpu().data.numpy()[None, :]
            example_te_yhat = y_att.cpu().data.numpy()[None, :]
        else:
            example_te_y = np.nan
            example_te_yhat = np.nan
        if z_unatt is None:
            example_te_unatt = np.array([np.nan])
        else:
            example_te_unatt = z_unatt.data.numpy()[None, :]

    ################################################################
    #              Save
    ################################################################
    # Save network parameters and outputs

    ver_list = []
    for v in [torch, np, scipy, nipype]:
        ver_list.append(v.__name__ + "_" + v.__version__)
    ver_list.append('python_' + sys.version)

    if save_flag:
        dct_all = {**{'loss': loss_history, 'train': train, 'test': test,
                      'file_path_name_audio': file_path_name_audio,
                      'file_path_name_eeg': file_path_name_eeg,
                      'valset': valset,
                      'loss_val_history': loss_val_history,
                      'idx_sample_list': idx_sample_list,
                      'yValAtt': example_val_y,
                      'yValHat': example_val_yhat,
                      'yValUna': example_val_z_unatt,
                      'yTrainAtt': example_tr_y,
                      'yTrainHat': example_tr_yhat,
                      'yTrainUna': example_tr_unatt,
                      'yTestAtt': example_te_y,
                      'yTestHat': example_te_yhat,
                      'yTestUna': example_te_unatt,
                      'envTestAtt': example_te_y,  # output api compatible
                      'envHatAtt': example_te_yhat,  # output api compatible
                      'envTestUna': example_te_unatt,  # output api compatible
                      #'subjID': re.search('Subj_(\d+)_', file_path_name_audio).group(1),  # output api compatible
                      'subjID': file_path_name_audio.split('\\')[0][-2:],
                      'ver_list': ver_list
                      },

                   **dct_params}

        hashstr = ''
        for key, val in {**{'train': train}, **dct_params}.items():
            if type(val) is str:
                hashstr = hashstr + key + val
            elif type(val) in [float, int]:
                hashstr = hashstr + key + str(val)
            elif type(val) in [list]:
                if type(val[0]) is str:
                    hashstr = hashstr + key + ','.join(val)
                elif type(val[0]) in [float, int]:
                    hashstr = hashstr + key + ','.join([str(i) for i in val])
        hexstamp = hashlib.md5(hashstr.encode('utf')).hexdigest()

        now_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        file_path_name_checkpoint = os.path.join(file_path_save,
                                                 'checkpoint_eeg2env_%s_%s.pt'
                                                 % (hexstamp, now_str))
        torch.save({'state_dict': model.state_dict()}, file_path_name_checkpoint)

        print(file_path_name_checkpoint)
        # Replace all None elements of dict with NaN before saving to avoid save fail.
        for key, val in dct_all.items():
            if val is None:
                dct_all[key] = np.nan
        scipy.io.savemat(os.path.join(file_path_save,
                                      'checkpoint_eeg2env_%s_%s.mat'
                                      % (hexstamp, now_str)),
                         dct_all)

    model = None
    return model