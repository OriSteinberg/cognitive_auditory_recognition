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
    dct_params: dict, Collection of auxiliary parameters
    """
    ################################################################
    #      Unpack parameters, define model, define data
    ################################################################
    # Setup the dnn, and create the monolithic block of data that will be used for training.

    num_context        = dct_params['num_context']
    num_epoch          = dct_params['num_epoch']
    save_flag          = dct_params['save_flag']
    file_path_save     = dct_params['file_path_save']
    file_path_name_net = dct_params['file_path_name_net']
    batch_size         = dct_params['batch_size']
    loss_type          = dct_params['loss_type']
    idx_split          = dct_params['idx_split']
    random_seed_flag   = dct_params['random_seed_flag']
    eval_modulo        = dct_params['eval_modulo']
    #input_size         = dct_params['data_prod_num']
    hidden_size        = dct_params['hidden_size']
    output_size        = dct_params['output_size']
    lr                 = dct_params['learning_rate']
    weight_decay       = dct_params['weight_decay']
    trails_num         = dct_params['trails_num']

    if random_seed_flag:
        np.random.seed(idx_split)
        torch.manual_seed(idx_split)
    else:
        np.random.seed(0)
        torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    if False:  # torch.cuda.is_available():
        cuda_flag = True
        model.cuda()
        print('Using CUDA')
    else:
        print('No CUDA')
        cuda_flag = False

    # Load and preprocess the data
    file_path_name_get_data = dct_params['file_path_name_get_data']
    sys.path.append(os.path.split(file_path_name_get_data)[0])
    module = __import__(os.path.split(file_path_name_get_data)[1])
    reload(module)
    get_data = getattr(module, 'get_data')
    load_data = getattr(module, 'load_data')
    # path to folder containing the class.py module
    sys.path.append(os.path.split(file_path_name_net)[0])
    module = __import__(os.path.split(file_path_name_net)[1])
    reload(module)  # handle case of making changes to the module- forces reload
    NN = getattr(module, 'NN')

    model = NN(input_size, hidden_size, output_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # params = model.state_dict()

    if loss_type == 'mse':
        loss_fn = nn.MSELoss()
    elif loss_type == 'corr':
        loss_fn = closs
    elif loss_type == 'bce':
        loss_fn = nn.BCEWithLogitsLoss()
    loss_history = np.nan * np.zeros(num_epoch)
    loss_val_history = np.nan * np.zeros(num_epoch)
    model.train()  # Turn on dropout, batchnorm
    # model.eval()

    # Comment out in order to have the same val set and therefore the same train set between runs
    # train = np.asarray(train)[np.random.permutation(len(train))].tolist()
    if 1:
        valset = train[-2:]
        train = train[:-2]
    else:
        valset = []
    audio, eeg, audio_unatt = load_data(file_path_name_audio, file_path_name_eeg)
    x_all, y_all = torch.tensor([]), torch.tensor([])

    #####################################################################
    #####################################################################
    for idx_sample in train[:3]:  # todo
        x, y = get_data(audio, eeg, audio_unatt, idx_sample=idx_sample, num_context=num_context, dct_params=dct_params)
        if x is not None:
            x_all, y_all = torch.cat((x_all, x), dim=0), torch.cat((y_all, y), dim=0)
    set_size = x.size(0)
    for test in range(trails_num):
        x_train = torch.cat((x_all[:set_size * test], x_all[set_size * (test + 1):]))
        x_test = x_all[set_size * test: set_size * (test + 1)]

        # Outside the loop to only form conv matrix once
        xval, yval = get_data(audio, eeg, audio_unatt, idx_sample=valset[0], num_context=num_context, dct_params=dct_params)
        # xval, yval = torch.tensor([]), torch.tensor([])
        # for idx_sample in valset:
        #    x, y = get_data(audio, eeg, audio_unatt, idx_sample=idx_sample, num_context=num_context, dct_params=dct_params)
        #     if x is not None:
        #         xval, yval = torch.cat((xval, x), dim=0), torch.cat((yval, y), dim=0)


    ################################################################
    #              Training loop
    ################################################################
    # Iterate over the dataset a fixed number of times or until an early stopping condition is reached.
    # Randomly select a new batch of training at each iteration

    start = time.perf_counter()
    t_start = datetime.datetime.now()
    for idx_train in range(num_epoch):
        if np.mod(idx_train, num_epoch / 10) == 0:
            end = time.perf_counter()
            t_end = datetime.datetime.now()
            print('epoch %d, Time per epoch %2.5f ticks' % (idx_train, (end - start) / (num_epoch / 10)))
            print((t_end - t_start) / (num_epoch / 10))
            start = time.perf_counter()
            t_start = datetime.datetime.now()

        idx_keep = np.random.permutation(x_all.data.size(0))[:batch_size]
        idx_keep = torch.from_numpy(idx_keep).type('torch.LongTensor')
        x, y = x_all[idx_keep], y_all[idx_keep]
        # x = x + Variable(0. * torch.randn(x.shape))    # Data augmentation via noise

        if x is not None:
            model.zero_grad()
            if cuda_flag:
                y = y.cuda()
                output = model.forward(x.cuda())
            else:
                output = model.forward(x)

            loss = loss_fn(output.view(-1), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if cuda_flag:
                loss = loss.cpu()
            loss_history[idx_train] = loss.data.numpy()

            if loss_history[idx_train] < 0.09:
                print("early_stop!")
                break

            # Check validation set performance
            if (len(valset) > 0) and (np.mod(idx_train, eval_modulo) == 0):
                model.eval()

                idx_keep = np.sort(np.random.permutation(xval.data.size(0))[:batch_size])
                idx_keep = torch.from_numpy(idx_keep).type('torch.LongTensor')
                x = xval[idx_keep]
                y = yval[idx_keep]

                if cuda_flag:
                    y_att = model.forward(x.cuda())
                    stat_1 = loss_fn(y_att.view(-1), y.cuda().view(-1))
                else:
                    y_att = model.forward(x)
                    stat_1 = loss_fn(y_att.view(-1), y.view(-1))
                    stat_1 = stat_1.data.numpy()
                loss_val_history[idx_train // eval_modulo] = stat_1
                model.train()

                example_val_y = y.cpu().data.numpy()
                example_val_yhat = y_att.cpu().data.numpy()

    print('-- done training --')
    print(datetime.datetime.now())

    ################################################################
    #              Evaluation
    ################################################################
    # Test on the train set, then test on the test set.

    if True:
        example_tr_y = []
        example_tr_yhat = []
        for idx_tr in train[:1]:
            x, y = get_data(audio, eeg, audio_unatt, idx_sample=idx_tr, num_context=num_context, dct_params=dct_params)
            if x is not None:
                model.eval()
                if cuda_flag:
                    y_att = model.forward(x.cuda())
                else:
                    y_att = model.forward(x)
                example_tr_y.append(y.cpu().data.numpy())
                example_tr_yhat.append(y_att.cpu().data.numpy())

    if True:
        x, y = get_data(audio, eeg, audio_unatt, idx_sample=test[0], num_context=num_context, dct_params=dct_params)

        if x is not None:
            model.eval()
            if cuda_flag:
                y_att = model.forward(x.cuda())
            else:
                y_att = model.forward(x)
            example_te_y = y.cpu().data.numpy()[None, :]
            example_te_yhat = y_att.cpu().data.numpy()[None, :]
        else:
            example_te_y = np.nan
            example_te_yhat = np.nan

    train_correct = sum(np.sign(example_tr_y[0]-0.5) == np.sign(example_tr_yhat[0]))[0]
    test_correct = sum(np.sign(example_te_y.squeeze()-0.5) == np.sign(example_te_yhat.squeeze()))
    print(f'train accuracy - {train_correct}/{len(example_tr_y[0])} = {train_correct/len(example_tr_y[0])}')
    print(f'test accuracy - {test_correct}/{len(example_te_y.squeeze())} = {test_correct/len(example_te_y.squeeze())}')

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
                      'yValAtt': example_val_y,
                      'yValHat': example_val_yhat,
                      'yTrainAtt': example_tr_y,
                      'yTrainHat': example_tr_yhat,
                      'yTestAtt': example_te_y,
                      'yTestHat': example_te_yhat,
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
        file_path_name_checkpoint = os.path.join(file_path_save, 'checkpoint_eeg2env_%s_%s.pt' % (hexstamp, now_str))
        torch.save({'state_dict': model.state_dict()}, file_path_name_checkpoint)

        print(file_path_name_checkpoint)
        # Replace all None elements of dict with NaN before saving to avoid save fail.
        for key, val in dct_all.items():
            if val is None:
                dct_all[key] = np.nan
        scipy.io.savemat(os.path.join(file_path_save, 'checkpoint_eeg2env_%s_%s.mat' % (hexstamp, now_str)), dct_all)
    return None


def closs(x, y):
    xbar = torch.mean(x)
    ybar = torch.mean(y)
    num = 1. / x.numel() * torch.dot(x - xbar, y - ybar)
    denom = torch.std(x) * torch.std(y)
    return -num / denom
