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


def big_node(trails_num, file_path_name_audio, file_path_name_eeg, dct_params):
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
    random_seed_flag   = dct_params['random_seed_flag']
    eval_modulo        = dct_params['eval_modulo']
    hidden_size        = dct_params['hidden_size']
    output_size        = dct_params['output_size']
    lr                 = dct_params['learning_rate']
    weight_decay       = dct_params['weight_decay']
    eval_flag          = dct_params['eval_flag']
    idx_keep_audioTime = dct_params['idx_keep_audioTime']

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

    model = NN(0, hidden_size, output_size)
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

    audio, eeg, audio_unatt = load_data(file_path_name_audio, file_path_name_eeg)
    x_all, y_all = torch.tensor([]), torch.tensor([])
    import time
    ts1 = time.time()
    for idx_sample in range(trails_num):  # todo
        x, y = get_data(audio, eeg, audio_unatt, idx_sample=idx_sample, num_context=num_context, dct_params=dct_params)
        if x is not None:
            x_all, y_all = torch.cat((x_all, x), dim=0), torch.cat((y_all, y), dim=0)
    te1 = time.time()
    print(f'\n***** get data - {te1 - ts1} ******\n')
    set_size = x.size(0)
    our_accu_lst, thier_accu_lst = [], []
    for test in range(trails_num - 1):
        print('\n\n******************* test %d' % test)
        x_train = torch.cat((x_all[:set_size * test], x_all[set_size * (test + 1):set_size * (trails_num-1)]))
        y_train = torch.cat((y_all[:set_size * test], y_all[set_size * (test + 1):set_size * (trails_num-1)]))
        x_test  = x_all[set_size * test: set_size * (test + 1)]
        y_test  = y_all[set_size * test: set_size * (test + 1)]
        xval    = x_all[-set_size:]
        yval    = y_all[-set_size:]


        ################################################################
        #              Training loop
        ################################################################
        # Iterate over the dataset a fixed number of times or until an early stopping condition is reached.
        # Randomly select a new batch of training at each iteration

        start = time.perf_counter()
        for idx_train in range(num_epoch):
            if np.mod(idx_train, num_epoch // 5) == 0:
                end = time.perf_counter()
                print('epoch %d, Time per epoch %2.5f ticks' % (idx_train, (end - start) / (num_epoch / 10)))
                start = time.perf_counter()

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
                    print("early_stop!")
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

        print('-- done training --')
        print(datetime.datetime.now())

        ################################################################
        #              Evaluation
        ################################################################
        # Test on the train set, then test on the test set.

        if eval_flag:
            x, y = x_train[:set_size], y_train[:set_size]
            if x is not None:
                model.eval()
                if cuda_flag:
                    y_att = model.forward(x.cuda())
                else:
                    y_att = model.forward(x)
                example_tr_y = y.cpu().data.numpy()
                example_tr_yhat = y_att.cpu().data.numpy()

            x, y = x_test, y_test
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

        example_te_y, example_te_yhat = example_te_y.squeeze(), example_te_yhat.squeeze()
        train_correct_our = sum(np.sign(example_tr_y - 0.5) == np.sign(example_tr_yhat))[0]
        test_correct_our = sum(np.sign(example_te_y - 0.5) == np.sign(example_te_yhat))
        print(f'train accuracy - {train_correct_our / len(example_tr_y)}    test accuracy - {test_correct_our / len(example_te_y)}')

        train_correct_thier, test_correct_thier = 0., 0.
        for i in range(len(example_tr_yhat) // 2):
            if example_tr_yhat[i] < example_tr_yhat[i + len(example_tr_yhat) // 2]:
                train_correct_thier += 1.
            if example_te_yhat[i] < example_te_yhat[i + len(example_te_yhat) // 2]:
                test_correct_thier += 1.
        print(f'train accuracy - {train_correct_thier / (len(example_te_y.squeeze()) / 2)}    test accuracy - {test_correct_thier / (len(example_te_y.squeeze()) / 2)}')
        our_accu_lst.append([train_correct_our / len(example_tr_y), test_correct_our / len(example_te_y)])
        thier_accu_lst.append([train_correct_thier / (len(example_te_y.squeeze()) / 2), test_correct_thier / (len(example_te_y.squeeze()) / 2)])

    ################################################################
    #              Save
    ################################################################
    # Save network parameters and outputs
    with open('accu.txt', 'w',) as f:
        f.write('test\tour(train, test)\tthier(train, test)\n\n')
        for i in range(len(our_accu_lst)):
            accu_str = f'{i} - {our_accu_lst[i][0]}, {our_accu_lst[i][1]}, {thier_accu_lst[i][0]}, {thier_accu_lst[i][1]}\n'
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
