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
import nipype


def big_node(test_set, dct_params, SHORT=False):
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
    trails_num, set_size, file_path_name_audio, file_path_name_eeg = test_set
    num_context        = dct_params['num_context']
    num_epoch          = dct_params['num_epoch']
    save_flag          = dct_params['save_flag']
    file_path_save     = dct_params['file_path_save']
    file_path_name_net = dct_params['file_path_name_net']
    batch_size         = dct_params['batch_size']
    loss_type          = dct_params['loss_type']
    eval_modulo        = dct_params['eval_modulo']
    hidden_size        = dct_params['hidden_size']
    output_size        = dct_params['output_size']
    lr                 = dct_params['learning_rate']
    weight_decay       = dct_params['weight_decay']
    eval_flag          = dct_params['eval_flag']
    idx_keep_num       = dct_params['idx_keep_num']
    channel_num        = dct_params['channel_num']
    audio_num          = dct_params['audio_num']

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

    model = NN(idx_keep_num, hidden_size, output_size)
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
    ts1 = time.time()
    if SHORT:
        print('slim data mode')
        audio, eeg, audio_unatt = load_data(file_path_name_audio, file_path_name_eeg)
        x_all, y_all = get_data(audio, eeg, audio_unatt, idx_sample=0, num_context=num_context, dct_params=dct_params)
    elif not os.path.exists(f'data/x_all_{num_context}_10.tensor'):
        print(f'data/x_all_{num_context}.tensor not exist, load & process data')
        audio, eeg, audio_unatt = load_data(file_path_name_audio, file_path_name_eeg)
        #x_all, y_all = torch.tensor([]), torch.tensor([])
        x_all, y_all = torch.zeros((set_size * 30, channel_num + audio_num, idx_keep_num)), torch.zeros((set_size * 30, 1))
        x_tmp, y_tmp = torch.tensor([]), torch.tensor([])
        for idx_sample in range(trails_num):  # todo
            if idx_sample > 0 and idx_sample % 10 == 0:
                torch.save(x_tmp, f'data/x_all_{num_context}_{idx_sample}.tensor')
                torch.save(y_tmp, f'data/y_all_{num_context}_{idx_sample}.tensor')
                x_all[set_size * (idx_sample - 10): set_size * idx_sample] = x_tmp
                y_all[set_size * (idx_sample - 10): set_size * idx_sample] = y_tmp
                x_tmp, y_tmp = torch.tensor([]), torch.tensor([])

            x, y = get_data(audio, eeg, audio_unatt, idx_sample=idx_sample, num_context=num_context, dct_params=dct_params)
            if x is not None:
                x_tmp, y_tmp = torch.cat((x_tmp, x), dim=0), torch.cat((y_tmp, y), dim=0)

        torch.save(x_tmp, f'data/x_all_{num_context}_{idx_sample + 1}.tensor')
        torch.save(y_tmp, f'data/y_all_{num_context}_{idx_sample + 1}.tensor')

        x_all[set_size * (idx_sample - 10): set_size * (idx_sample + 1)] = x_tmp
        y_all[set_size * (idx_sample - 10): set_size * (idx_sample + 1)] = y_tmp
    else:
        print(f'data/loading x_all_{num_context}.tensor')
        x_all, y_all = torch.zeros((set_size * 30, channel_num + audio_num, idx_keep_num)), torch.zeros((set_size * 30, 1))
        for i, banch in enumerate(['10', '20', '30']):
            print('load ' + banch)
            x = torch.load(f'data/x_all_{num_context}_{banch}.tensor')
            y = torch.load(f'data/y_all_{num_context}_{banch}.tensor')
            x_all[set_size * 10 * i: set_size * 10 * (i + 1)] = x
            y_all[set_size * 10 * i: set_size * 10 * (i + 1)] = y
    print(f'\n***** data process time - {time.time() - ts1} ******\n')

    acc_lst = []
    i = np.random.randint(0, (len(x_all) // set_size) - 8)
    acc_lst.append(accuracy(model, set_size, x_all[i * set_size: (i + 4) * set_size], y_all[i * set_size: (i + 4) * set_size],
             x_all[(i + 4) * set_size: (i + 8) * set_size], y_all[(i + 4) * set_size: (i + 8) * set_size], cuda_flag))

    early_stop = False
    for test in range(trails_num - 1):
        if early_stop:
            break
        print('\n******************* test %d' % test)
        ts_epoch = time.time()
        if SHORT:
            x_train, y_train = x_all, y_all
        else:
            x_train = torch.cat((x_all[:set_size * test], x_all[set_size * (test + 4):set_size * (trails_num-1)]))
            y_train = torch.cat((y_all[:set_size * test], y_all[set_size * (test + 4):set_size * (trails_num-1)]))
        x_test  = x_all[set_size * test: set_size * (test + 4)]
        y_test  = y_all[set_size * test: set_size * (test + 4)]
        xval    = x_all[-set_size * 2:]
        yval    = y_all[-set_size * 2:]
        print('training load data {}'.format(time.time() - ts_epoch))
        ts_epoch = time.time()
        ################################################################
        #              Training loop
        ################################################################
        # Iterate over the dataset a fixed number of times or until an early stopping condition is reached.
        # Randomly select a new batch of training at each iteration
        
        for idx_train in range(num_epoch):
            idx_keep = np.random.permutation(x_train.data.size(0))[:batch_size]
            idx_keep = torch.from_numpy(idx_keep).type('torch.LongTensor')
            x, y = x_train[idx_keep], y_train[idx_keep]
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
                    print("\n\nearly_stop!\n\n")
                    early_stop = True
                    break

                # Check validation set performance
                if (len(xval) > 0) and (np.mod(idx_train, eval_modulo) == 0):
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

        print(f'training run time - {time.time() - ts_epoch}')
        ts_epoch = time.time()

        ################################################################
        #              Evaluation
        ################################################################
        # Test on the train set, then test on the test set.

        if eval_flag:
            i = np.random.randint(0, (len(x_train) // set_size) - 4)
            acc_lst.append(accuracy(model, set_size, x_train[i * set_size: (i + 4) * set_size], y_train[i * set_size: (i + 4) * set_size],
                     x_test, y_test, cuda_flag))
        print(f'evaluation run time - {time.time() - ts_epoch}')

    ################################################################
    #              Save
    ################################################################
    # Save network parameters and outputs
    with open('accu_%s.txt' % (file_path_name_eeg.split('\\')[0][-2:]), 'w',) as f:
        f.write('test\tour(train, test)\tthier(train, test)\n\n')
        for i in range(len(acc_lst)):
            accu_str = f'{i} - {acc_lst[i][0]}, {acc_lst[i][1]}, {acc_lst[i][2]}, {acc_lst[i][3]}\n'
            f.write(accu_str)

    ver_list = []
    for v in [torch, np, scipy, nipype]:
        ver_list.append(v.__name__ + "_" + v.__version__)
    ver_list.append('python_' + sys.version)

    if save_flag:
        dct_all = {**{'loss': loss_history, 'test': test,
                      'file_path_name_audio': file_path_name_audio,
                      'file_path_name_eeg': file_path_name_eeg,
                      'loss_val_history': loss_val_history,
                      'acc_lst': acc_lst,
                      'subjID': file_path_name_audio.split('\\')[0][-2:],
                      'ver_list': ver_list
                      },
                   **dct_params}

        hashstr = ''
        for key, val in {**dct_params}.items():
            if type(val) is str:
                hashstr = hashstr + key + val
            elif type(val) in [float, int]:
                hashstr = hashstr + key + str(val)
            elif type(val) in [list]:
                if type(val[0]) is str:
                    hashstr = hashstr + key + ','.join(val)
                elif type(val[0]) in [float, int]:
                    hashstr = hashstr + key + ','.join([str(i) for i in val])

        now_str = datetime.datetime.now().strftime('%Y%m%d%H%M')
        file_path_name_checkpoint = os.path.join(file_path_save, 'checkpoint_eeg2env_%s.pt' % now_str)
        torch.save({'state_dict': model.state_dict()}, file_path_name_checkpoint)
        torch.save(model.state_dict(), file_path_name_checkpoint)

        print(file_path_name_checkpoint)
        # Replace all None elements of dict with NaN before saving to avoid save fail.
        for key, val in dct_all.items():
            if val is None:
                dct_all[key] = np.nan
        scipy.io.savemat(os.path.join(file_path_save, 'checkpoint_eeg2env_%s.mat' % now_str), dct_all)
    return None


def closs(x, y):
    xbar = torch.mean(x)
    ybar = torch.mean(y)
    num = 1. / x.numel() * torch.dot(x - xbar, y - ybar)
    denom = torch.std(x) * torch.std(y)
    return -num / denom


def accuracy(model, set_size, x_train, y_train, x_test, y_test, cuda_flag=False):
    train_correct_our, test_correct_our = 0., 0.
    train_correct_thier, test_correct_thier = 0., 0.
    for i in range(4):  # len(x_train) // set_size):
        x, y = x_train[set_size * i:set_size * (i + 1)], y_train[set_size * i:set_size * (i + 1)]
        if x is not None:
            model.eval()
            if cuda_flag:
                y_att = model.forward(x.cuda())
            else:
                y_att = model.forward(x)
            example_tr_y = y.cpu().data.numpy()
            example_tr_yhat = y_att.cpu().data.numpy()

        train_correct_our += sum(np.sign(example_tr_y - 0.5) == np.sign(example_tr_yhat))[0]
        for j in range((set_size // 2) - 1):
            if example_tr_yhat[j] < example_tr_yhat[j + (set_size // 2)]:
                train_correct_thier += 1.

    for i in range(4):  # len(x_test) // set_size):
        x, y = x_test[set_size * i:set_size * (i + 1)], y_test[set_size * i:set_size * (i + 1)]
        if x is not None:
            model.eval()
            if cuda_flag:
                y_att = model.forward(x.cuda())
            else:
                y_att = model.forward(x)
            example_te_y = y.cpu().data.numpy().squeeze()
            example_te_yhat = y_att.cpu().data.numpy().squeeze()
        else:
            example_te_y = np.nan
            example_te_yhat = np.nan

        test_correct_our += sum(np.sign(example_te_y - 0.5) == np.sign(example_te_yhat))
        for j in range((set_size // 2) - 1):
            if example_te_yhat[j] < example_te_yhat[j + (set_size // 2)]:
                test_correct_thier += 1.

    print('our: train accuracy - %.3f    test accuracy - %.3f' % (
        train_correct_our / (set_size * 4), test_correct_our / (set_size * 4)))
    print('their: train accuracy - %.3f    test accuracy - %.3f' % (
        train_correct_thier / (set_size * 2),
        test_correct_thier / (set_size * 2)))

    return train_correct_our, test_correct_our, train_correct_thier, test_correct_thier
