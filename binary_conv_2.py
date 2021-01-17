import torch.nn
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
        self.in_channels = in_channels
        dilation = 1
        kernel_size_1, stride = 5, 1
        self.conv_1 = nn.Conv1d(in_channels, in_channels * 3 // 2, kernel_size_1, stride=stride, dilation=dilation, padding=0)
        self.conv_11 = nn.Conv1d(in_channels * 3 // 2, in_channels // 2, kernel_size_1, stride=stride, dilation=dilation, padding=0)

        kernel_size_2 = 5
        self.conv_2 = nn.Conv1d(in_channels // 2 + audio_num, hidden_channels, kernel_size_2, stride=stride, dilation=dilation)
        self.m2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False)

        dilation = 1
        kernel_size_3, stride = 1, 1
        out_channels = 2
        self.conv_3 = nn.Conv1d(hidden_channels, out_channels, kernel_size_3, stride=stride, dilation=dilation)
        self.m3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False)

        import numpy as np
        # i2o creates the output sample
        fc_in_size = int(((((input_size - (kernel_size_1-1)*2) - kernel_size_2+1) // 2) - kernel_size_3+1) // 2)
        self.i2o_l1 = nn.Linear(out_channels * fc_in_size, 2 * hidden_size, bias=True)
        self.i2o_l2 = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=True)
        self.i2o_l3 = nn.Linear(2 * hidden_size, 1 * hidden_size, bias=True)
        self.i2o_l4 = nn.Linear(1 * hidden_size, 1 * output_size, bias=True)

        self.bn_l0 = nn.BatchNorm1d(in_channels + audio_num)
        self.bn_l1 = nn.BatchNorm1d(2 * fc_in_size)
        self.bn_l2 = nn.BatchNorm1d(2 * hidden_size)
        self.bn_l3 = nn.BatchNorm1d(2 * hidden_size)
        self.bn_l11 = nn.BatchNorm1d(in_channels // 2)

    def forward(self, x):
        x = self.bn_l0(x)
        x_brain = x[:, :self.in_channels]
        x_audio = x[:, -1]
        x_brain = F.elu(self.conv_1(x_brain))
        x_brain = F.elu(self.conv_11(x_brain))
        x_brain = self.bn_l11(x_brain)
        x = torch.cat((x_brain, x_audio.unsqueeze(1)), dim=1)

        x = F.elu(self.conv_2(x))
        x = self.m2(x)

        x = F.elu(self.conv_3(x))
        x = self.m3(x)
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
