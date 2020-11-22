import scipy.io
import numpy as np
import torch
from torch.autograd import Variable
import mat73


def load_data(file_path_name_audio, file_path_name_eeg=False):
    """Return attended audio, eeg, and unattended audio from *.mat file.
    """
    if not file_path_name_eeg:
        eeg, audio, audio_unatt = short_prep_mat(file_path_name_audio)
    else:
        loaded_data_audio = scipy.io.loadmat(file_path_name_audio)
        loaded_data_eeg = scipy.io.loadmat(file_path_name_eeg)

        audio = loaded_data_audio['data']
        eeg = loaded_data_eeg['data']
        eeg = eeg[:, :64, :]
        audio_unatt = loaded_data_audio['data_unatt']
        # eeg - [trails, 64, time]
        # audio, audio_unatt - [trails, time]
    return audio, eeg, audio_unatt


def get_data(audio, eeg, audio_unatt=None, idx_sample=None, num_context=1, dct_params=None):
    """
    Select a sequence of audio, audio_unattnd, & eeg data Reshape the selected data into batch_size frames for prediction

    Arguments
    ---------
    audio : (num_part, num_samples)
    eeg : (num_part, num_ch, num_samples)
    audio_unatt : (num_part, num_samples)
    idx_sample : row idx of audio and eeg data Defaults to a random sample if not specified
    num_context : scalar, Total number of samples of input used to predict an output sample.
        If one-to-one mapping with no delay, num_context=1
    dct_params : ['idx_keep_audioTime'] field index of samples into the time vector

    Returns
    -------
    X : Variable (batch_size, num_ch * num_context + num_context) eeg + audio
    y : Variable (batch_size, class)
    """

    if (dct_params is not None) and ('idx_keep_audioTime' in dct_params):
        idx_keep_audioTime = dct_params['idx_keep_audioTime']
    else:
        idx_keep_audioTime = None

    # Trim off NaNs
    a, e, au = audio[idx_sample], eeg[idx_sample], audio_unatt[idx_sample]
    # e - [Chanel, Time]   a/au - [1, Time]
    idx_a, idx_e = np.logical_not(np.isnan(a)), np.logical_not(np.isnan(e[1]))
    if np.abs(np.sum(idx_a) - np.sum(idx_e)) > 3:
        print('unequal samples')
    idx_keep = np.logical_and(idx_a, idx_e)
    a, e, au = a[idx_keep], e[:, idx_keep], au[idx_keep]

    if a.shape[0] >= num_context:
        # Make a conv matrix out of the eeg, attended audio, and unattended audio
        # Cat [X_eeg, X_audio], y = 1
        # Cat [X_eeg, X_audio_unatt], y = 0

        num_time = a.size - num_context + 1  # how many slide windows
        num_ch = e.shape[0]

        if idx_keep_audioTime is None:
            num_column_audio = num_context
            idx_keep_audioTime = np.arange(num_context)
        else:
            num_column_audio = np.size(idx_keep_audioTime)

        X_eeg         = np.nan * np.ones((num_time, num_ch, num_column_audio))
        X_audio       = np.nan * np.ones((num_time, num_column_audio))
        X_audio_unatt = np.nan * np.ones((num_time, num_column_audio))

        for idx in range(num_time):
            idx_keep = np.arange(num_context) + idx
            for idx_ch in range(num_ch):
                X_eeg[idx, idx_ch] = np.ravel(e[idx_ch, idx_keep])[idx_keep_audioTime]
            X_audio[idx] = np.ravel(a[idx_keep])[idx_keep_audioTime]
            X_audio_unatt[idx] = np.ravel(au[idx_keep])[idx_keep_audioTime]
        X_audio = X_audio[:, None, :]
        X_audio_unatt = X_audio_unatt[:, None, :]
        # X_eeg - [num_time(windows), channel, idx_keep_audioTime]
        # X_audio/ X_audio_unatt - [num_time(windows), idx_keep_audioTime]

        X1 = np.concatenate((X_eeg, X_audio), axis=1)
        X0 = np.concatenate((X_eeg, X_audio_unatt), axis=1)
        X = np.concatenate((X0, X1), axis=0)
        y = np.concatenate((np.zeros((num_time, 1)), np.ones((num_time, 1))), axis=0)
        # x - [num_time(windows) * 2, channel + 1, num_context\np.size(idx_keep_audioTime)]
        # y - [num_time(windows) * 2, 1]

        X = Variable(torch.from_numpy(X).type('torch.FloatTensor'))
        y = Variable(torch.from_numpy(np.array(y)).type('torch.FloatTensor'))
    else:
        print('-warning, too little data-')
        X, y = None, None
    return X, y


def prep_mat_file():
    dirs = ['s2']
    folder = 'C:/Py_ws/DL/thesis/data/two_audio_selective/'
    for sbj in dirs:
        loaded_data_audio = scipy.io.loadmat(folder + sbj + "/pilotVR05_Smat")
        loaded_data_eeg = scipy.io.loadmat(folder + sbj + "/pilotVR05_Rmat.mat")
        audio = loaded_data_audio['S'][0]
        eeg = loaded_data_eeg['R'][0]
        max_len = 99999
        for i in range(len(eeg)):
            if audio[i].shape[0] < max_len:
                max_len = audio[i].shape[0]
        data_a = np.zeros((audio.shape[0], max_len))
        data_au = np.zeros((audio.shape[0], max_len))
        for i, mat in enumerate(audio):
            data_a[i] = mat[:max_len, 0]
            data_au[i] = mat[:max_len, 1]

        data_e = np.zeros((audio.shape[0], 64, max_len))
        for i in range(len(eeg)):
            mat = eeg[i].transpose()
            data_e[i] = mat[:, :max_len]
        mat_a  = {'data': data_a, 'data_unatt': data_au}
        mat_e  = {'data': data_e}

        scipy.io.savemat('C:/Py_ws/DL/thesis/data/two_audio/' + sbj + '/Envelope.mat', mat_a)
        scipy.io.savemat('C:/Py_ws/DL/thesis/data/two_audio/' + sbj + '/EEG.mat', mat_e)


def prep_mat_file_v2():
    dirs = ['s2']
    folder = 'C:/Py_ws/DL/thesis/data/two_audio_selective/'
    for sbj in dirs:
        filepath = folder + sbj + "/s2_r_s.mat"
        data_dict = mat73.loadmat(filepath)
        audio = data_dict['S_ds']
        eeg = data_dict['R_ds']

        data_a = []
        data_au = []
        for i, mat in enumerate(audio):
            data_a.append(mat[:, 0])
            data_au.append(mat[:, 1])

        data_e = []
        for i in range(len(eeg)):
            mat = eeg[i].transpose()
            data_e.append(mat)
        mat_a  = {'data': data_a, 'data_unatt': data_au}
        mat_e  = {'data': [data_e]}

        scipy.io.savemat(folder + sbj + '/Envelope.mat', mat_a)
        scipy.io.savemat(folder + sbj + '/EEG.mat', mat_e)


def short_prep_mat(filename):
    data_dict = mat73.loadmat(filename)
    audio = data_dict['S_ds']
    eeg = data_dict['R_ds']
    data_a, data_au = [], []
    for i, mat in enumerate(audio):
        data_a.append(mat[:, 0])
        data_au.append(mat[:, 1])

    data_e = []
    for i in range(len(eeg)):
        mat = eeg[i].transpose()
        data_e.append(mat)
    return data_e, data_a, data_au
