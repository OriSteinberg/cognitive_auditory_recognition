import torch.nn as nn
import torch.nn.functional as F


class NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, in_channels=65, hidden_channels=64, audio_num=1):
        '''

        :param input_size: length of each example (number of time samples)
        :param hidden_size: fully connected hidden size
        :param output_size: whole net output size
        :param in_channels: conv layer input channel size
        :param hidden_channels: conv layer hidden channel size
        :param audio_num: number of speakers in each example
        '''
        super(NN, self).__init__()
        # weight matrix is (n_input, n_output) and a bias (n_output)
        dilation = 1
        kernel_size, stride = 3, 1
        self.conv_3 = nn.Conv1d(in_channels + audio_num, hidden_channels, kernel_size, stride=stride, dilation=dilation)
        self.m1 = nn.MaxPool1d(kernel_size, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False)

        dilation = 1
        kernel_size, stride = 1, 1
        out_channels = 2
        self.conv_1 = nn.Conv1d(hidden_channels, out_channels, kernel_size, stride=stride, dilation=dilation)
        self.m2 = nn.MaxPool1d(2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False)

        # i2o creates the output sample
        fc_in_size = (((input_size - 2) // 2) - 1) // 2
        self.i2o_l1 = nn.Linear(2 * fc_in_size , 2 * hidden_size, bias=True)
        self.i2o_l2 = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=True)
        self.i2o_l3 = nn.Linear(2 * hidden_size, 1 * hidden_size, bias=True)
        self.i2o_l4 = nn.Linear(1 * hidden_size, 1 * output_size, bias=True)

        self.bn_l0 = nn.BatchNorm1d(in_channels + audio_num)
        self.bn_l1 = nn.BatchNorm1d(2 * fc_in_size)
        self.bn_l2 = nn.BatchNorm1d(2 * hidden_size)
        self.bn_l3 = nn.BatchNorm1d(2 * hidden_size)

    def forward(self, x):
        x = self.bn_l0(x)
        x = F.elu(self.conv_3(x))
        x = self.m1(x)

        x = F.elu(self.conv_1(x))
        x = self.m2(x)
        x = x.view(x.shape[0], -1)

        x = self.i2o_l1(self.bn_l1(x))
        x = F.dropout(F.elu(x), p=0.25, training=self.training)

        x = self.i2o_l2(self.bn_l2(x))
        x = F.dropout(F.elu(x), p=0.25, training=self.training)

        x = self.i2o_l3(self.bn_l3(x))
        x = F.dropout(F.elu(x), p=0.25, training=self.training)

        x = self.i2o_l4(x)
        output = x
        return output
