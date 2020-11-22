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
LOAD_NUM = 5
TEST_LEN = 1


def big_node(model, test_set, dct_params):
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
    batch_size         = dct_params['batch_size']
    loss_type          = dct_params['loss_type']
    eval_modulo        = dct_params['eval_modulo']
    lr                 = dct_params['learning_rate']
    weight_decay       = dct_params['weight_decay']
    eval_flag          = dct_params['eval_flag']

    np.random.seed(0)
    torch.manual_seed(0)

    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        cuda_flag = True
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

    if cuda_flag:
        model.cuda()
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
    audio, eeg, audio_unatt = load_data(file_path_name_audio)
    data_set = audio, eeg, audio_unatt

    acc_lst = []
    early_stop = False
    for test in range((trails_num // LOAD_NUM)):
        if early_stop:
            acc_lst.append(
                accuracy(model, data_set, dct_params, get_data, num_context, train_idx=dev_idxes, test_idx=test_idxes))
            break
        print('\n******************* test %d' % test)

        ################################################################
        #              Data loading
        ################################################################
        ts = time.time()
        train_idxes = np.arange(LOAD_NUM) + (test * LOAD_NUM)
        dev_idxes = np.random.permutation(train_idxes)[:TEST_LEN]
        test_idxes = np.arange(TEST_LEN)
        if ((test + 1) * LOAD_NUM) + TEST_LEN < trails_num:
            test_idxes += ((test + 1) * LOAD_NUM)

        x_all, y_all = torch.tensor([]), torch.tensor([])
        for idx_sample in train_idxes:  #
            x, y = get_data(audio, eeg, audio_unatt, idx_sample=idx_sample, num_context=num_context,
                            dct_params=dct_params)
            if x is not None:
                x_all, y_all = torch.cat((x_all, x), dim=0), torch.cat((y_all, y), dim=0)

        if cuda_flag:
            x_all = x_all.cuda()
            y_all = y_all.cuda()
        div_idx = int(x_all.size(0) * 0.9)
        x_train, y_train = x_all[:div_idx], y_all[:div_idx]
        xval, yval = x_all[div_idx:], y_all[div_idx:]
        print('loading time - {}, example num - {}'.format(time.time() - ts, x_all.size(0)))
        ################################################################
        #              Training loop
        ################################################################
        # Iterate over the dataset a fixed number of times or until an early stopping condition is reached.
        # Randomly select a new batch of training at each iteration
        for i in range(4):
            ts_epoch = time.time()
            acc_lst.append(
                accuracy(model, data_set, dct_params, get_data, num_context, train_idx=dev_idxes, test_idx=test_idxes))
            print('accuracy time - {}'.format(time.time() - ts))
            ts_epoch = time.time()
            for idx_train in range(num_epoch):
                idx_keep = np.random.permutation(x_train.size(0))[:batch_size]
                idx_keep = torch.from_numpy(idx_keep).type('torch.LongTensor')
                x, y = x_train[idx_keep], y_train[idx_keep]
                # x = x + Variable(0. * torch.randn(x.shape))    # Data augmentation via noise

                model.zero_grad()
                output = model.forward(x)

                loss = loss_fn(output.view(-1), y.view(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_history[idx_train] = loss.cpu().data.numpy()

                if False: #loss_history[idx_train] < 0.09:
                    print("\n\nearly_stop!\n\n")
                    early_stop = True
                    break

                # Check validation set performance
                if (len(xval) > 0) and (np.mod(idx_train, eval_modulo) == 0):
                    model.eval()

                    idx_keep = np.sort(np.random.permutation(xval.size(0))[:batch_size])
                    idx_keep = torch.from_numpy(idx_keep).type('torch.LongTensor')
                    x, y = xval[idx_keep], yval[idx_keep]

                    y_att = model.forward(x)
                    stat_1 = loss_fn(y_att.view(-1), y.view(-1))
                    stat_1 = stat_1.cpu().data.numpy()
                    loss_val_history[idx_train // eval_modulo] = stat_1
                    model.train()

            print(f'training run time - {time.time() - ts_epoch}')

    acc_lst.append(
        accuracy(model, data_set, dct_params, get_data, num_context, train_idx=dev_idxes, test_idx=test_idxes))
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
    for v in [torch, np, scipy]:
        ver_list.append(v.__name__ + "_" + v.__version__)
    ver_list.append('python_' + sys.version)
    subject = file_path_name_audio.split('\\')[-1][:2]
    if save_flag:
        dct_all = {**{'loss': loss_history, 'test': test,
                      'file_path_name_audio': file_path_name_audio,
                      'file_path_name_eeg': file_path_name_eeg,
                      'loss_val_history': loss_val_history,
                      'acc_lst': acc_lst,
                      'subjID': subject,
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
        file_path_name_checkpoint = os.path.join(file_path_save, subject + '_eeg2env_%s.pt' % now_str)
        torch.save(model.state_dict(), file_path_name_checkpoint)

        print(file_path_name_checkpoint)
        # Replace all None elements of dict with NaN before saving to avoid save fail.
        for key, val in dct_all.items():
            if val is None:
                dct_all[key] = np.nan
        scipy.io.savemat(os.path.join(file_path_save, subject + '_eeg2env_%s.mat' % now_str), dct_all)
    return None


def closs(x, y):
    xbar = torch.mean(x)
    ybar = torch.mean(y)
    num = 1. / x.numel() * torch.dot(x - xbar, y - ybar)
    denom = torch.std(x) * torch.std(y)
    return -num / denom


def accuracy(model, data_set, dct_params, get_data, num_context, train_idx, test_idx):
    audio, eeg, audio_unatt = data_set
    train_correct_our, test_correct_our = 0., 0.
    train_correct_thier, test_correct_thier = 0., 0.
    all_train, all_test = 0., 0.
    for idx_sample in train_idx:
        x, y = get_data(audio, eeg, audio_unatt, idx_sample=idx_sample, num_context=num_context, dct_params=dct_params)
        set_size = x.size(0)
        if x is not None:
            model.eval()
            y_att = model.forward(x)
            example_tr_y = y.cpu().data.numpy()
            example_tr_yhat = y_att.cpu().data.numpy()
            all_train += len(y_att)

            train_correct_our += sum(np.sign(example_tr_y - 0.5) == np.sign(example_tr_yhat))[0]
            for j in range((set_size // 2) - 1):
                if example_tr_yhat[j] < example_tr_yhat[j + (set_size // 2)]:
                    train_correct_thier += 1.

    for idx_sample in test_idx:  # len(x_test) // set_size):
        x, y = get_data(audio, eeg, audio_unatt, idx_sample=idx_sample, num_context=num_context, dct_params=dct_params)
        set_size = x.size(0)
        if x is not None:
            model.eval()
            y_att = model.forward(x)
            example_te_y = y.cpu().data.numpy().squeeze()
            example_te_yhat = y_att.cpu().data.numpy().squeeze()
            all_test += len(y_att)

            test_correct_our += sum(np.sign(example_te_y - 0.5) == np.sign(example_te_yhat))
            for j in range((set_size // 2) - 1):
                if example_te_yhat[j] < example_te_yhat[j + (set_size // 2)]:
                    test_correct_thier += 1.
    tro, teo = train_correct_our / all_train, test_correct_our / all_test
    trt, tet = train_correct_thier / (all_train / 2), test_correct_thier / (all_test / 2)
    print('our: train accuracy - %.3f    test accuracy - %.3f' % (tro, teo))
    print('their: train accuracy - %.3f    test accuracy - %.3f' % (trt, tet))
    return tro, teo, trt, tet


def save_mat_as_tensore(dct_params, file_path_name_audio, file_path_name_eeg, trails_num, load_data, get_data):
    num_context = dct_params['num_context']
    ONE_MAT_FILE = dct_params['one_mat_flag']
    subject = file_path_name_audio.split('\\')[-1][:2]
    print(subject + f'/x_all_{num_context}_10.tensor not exist, load & process data')

    audio, eeg, audio_unatt = load_data(file_path_name_audio)
    x_tmp, y_tmp = torch.tensor([]), torch.tensor([])
    for idx_sample in range(trails_num):
        if idx_sample > 0 and idx_sample % 10 == 0:
            torch.save(x_tmp, subject + f'/x_all_{num_context}_{idx_sample}.tensor')
            torch.save(y_tmp, subject + f'/y_all_{num_context}_{idx_sample}.tensor')
            x_tmp, y_tmp = torch.tensor([]), torch.tensor([])

        x, y = get_data(audio, eeg, audio_unatt, idx_sample=idx_sample, num_context=num_context, dct_params=dct_params)
        if x is not None:
            x_tmp, y_tmp = torch.cat((x_tmp, x), dim=0), torch.cat((y_tmp, y), dim=0)

        torch.save(x_tmp, subject + f'/x_all_{num_context}_{idx_sample + 1}.tensor')
        torch.save(y_tmp, subject + f'/y_all_{num_context}_{idx_sample + 1}.tensor')


def load_tensor_mat(file_path_name_audio, num_context):
    print(f'loading - data/x_all_{num_context}.tensor')
    x_all, y_all = torch.tensor([]), torch.tensor([])
    #x_sets = sorted(os.listdir(subject, 'x_all*'))
    #y_sets = sorted(glob(os.path.join(subject, 'y_all*')))
    for i, banch in enumerate(['10', '20', '28']):
        print('load ' + banch)
        x = torch.load(subject + f'/x_all_{num_context}_{banch}.tensor')
        y = torch.load(subject + f'/y_all_{num_context}_{banch}.tensor')
        x_all = torch.cat((x_all, x), dim=0)
        y_all = torch.cat((y_all, y), dim=0)