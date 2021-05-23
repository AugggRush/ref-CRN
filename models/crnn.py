import torch
import torch.nn as nn


class OnedirectionalLSTM(nn.Module):

    def __init__(self, batchSize, nIn, nHidden, n_layer, nOut):
        #    nIn = channels * features
        #    nHidden = 1024
        #    n_layer = 2
        #    nOut = 320(frame_size)
        super(OnedirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, num_layers=n_layer)
        self.embedding = nn.Linear(nHidden, nOut)
        self.activationFunc = nn.Tanh()


    def forward(self, inputs, hs, hc):
        #    inputs = [batch, sequence(num_frame), channels*features]
        rnn_in = inputs.permute(1, 0, 2)
        #    hiddenFlow = [n_layer(2), batchSize, nHidden]
        # hiddenFlow = (torch.randn([n_layer, batchSize, nHidden]).cuda(),\
        #         torch.randn([n_layer, batchSize, nHidden]).cuda())
        #    rnn_in = [sequence(1), batch, channels*features]
        recurrent, hiddenOut = self.rnn(rnn_in, (hs, hc))
        # self.hiddenFlow = hiddenOut
        #    recurrent = [sequence(1), batch, nHidden]
        T, B, H = recurrent.size()

        t_rec = recurrent.view(T * B, H)
        #    t_rec = [batch*sequence(num_frame), nHidden]
        fc_output = self.embedding(t_rec) 
        #    fc_output = [batch*sequence(num_frame), nOut] 
        act_output = self.activationFunc(fc_output)
        #    fc_output = [sequence(num_frame), batch, nOut] 
        act_output = act_output.view(T, B, -1)
        #    return [batch, sequence(1), nOut]
        return act_output.permute(1, 0, 2).contiguous(), hiddenOut[0], hiddenOut[1]

class cnnEncoder(nn.Module):

    def __init__(self, in_dim, in_features, kernel_channels, kernel_sizes):
        #    in_dim = input channels = 1
        #    in_features = frame_size
        super(cnnEncoder, self).__init__()
        #    out_features recored the number of output features of every conv layer
        out_features = []

        conv_block = nn.ModuleList()
        for i in range(len(kernel_channels)):
            if i == 0:
                start_layer = nn.Conv2d(in_dim, kernel_channels[i], kernel_size=kernel_sizes[i])
                # cnn.add_module('start_conv{}'.format(i), start_layer)
                out_features.append(in_features)
            else:
                block_item = nn.Sequential(
                    nn.Conv1d(kernel_channels[i-1], kernel_channels[i], kernel_size=kernel_sizes[i]),
                    nn.LayerNorm([kernel_channels[i], out_features[i-1] - kernel_sizes[i] + 1]),
                    nn.PReLU()
                )
                conv_block.append(block_item)
                out_features.append(out_features[i-1] - kernel_sizes[i] + 1)
        #    self.numFeatures = out_features
        self.startLayer = start_layer
        self.convBlock = conv_block
        self.features = out_features

    def forward(self, miniBatch_inputs):
        #    minibatch_input = [batch, sequence(num_frame), features(frame_size)]
    
        #    minibatch_input = [batch, channels(1), sequence(num_frame), features(frame_size)]
        miniBatch_inputs = miniBatch_inputs.unsqueeze(1)
        #    minibatch_input = [batch, channels(16), sequence(num_frame), features(frame_size)]
        start_output = self.startLayer(miniBatch_inputs)
        start_output = start_output.permute(0,2,1,3).contiguous()
        #    conv_block_input = [batch*sequence(num_frame), channels(16), features(frame_size)]
        conv_block_input = start_output.view(-1, start_output.size(2), start_output.size(3))
        
        for i_layer in range(len(self.convBlock)):
            #    layer_output = [batch*sequence(num_frame), kernel_channels[i+1], out_features[i+1]]
            if i_layer == 0:
                layer_output = self.convBlock[i_layer](conv_block_input)
                # layer_output.append(first_output)
            else:
                layer_output = self.convBlock[i_layer](layer_output)

        return layer_output



class CRNN(nn.Module):

    def __init__(self, in_channel, batch_size, frame_size, kernel_channels, kernel_sizes, num_hidden, num_lstmLayer):
        #    in_channel = 1
        #    batch_size = 160
        #    frame_size = 320
        super(CRNN, self).__init__()
        self.batch_size = batch_size
        self.kernel_channels = kernel_channels
        self.kernel_sizes = kernel_sizes
        self.num_hidden = num_hidden
        self.num_lstmLayer = num_lstmLayer
        self.encoder = cnnEncoder(in_channel, frame_size, kernel_channels, kernel_sizes)
        self.rnn = OnedirectionalLSTM(self.batch_size, self.encoder.features[-1]*kernel_channels[-1], \
                                num_hidden, num_lstmLayer, frame_size)


    def forward(self, minibatch_in):
        #    minibatch_in = [batchsize, num_frame, features]
        # oneBatch = minibatch_in.permute(1, 0, 2).continuous()
        #    oneBatch = [num_frame, batchsize, features]
        # conv features
        output = torch.tensor([]).cuda() 
        hsState = nn.Parameter(torch.zeros([self.num_lstmLayer, self.batch_size, self.num_hidden])).cuda()
        hcState = nn.Parameter(torch.zeros([self.num_lstmLayer, self.batch_size, self.num_hidden])).cuda()
        #   store output in every time step
        # for tStep in range(minibatch_in.size(1)):
        #     if tStep == 0:
        #         crn_in = minibatch_in[:,tStep,:]
        #     else:
        #         mask_flag = torch.rand(1)
        #         if mask_flag <= 0.3:
        #             crn_in = output[:,tStep-1,:]
        #         else:
        #             crn_in = minibatch_in[:,tStep,:]
        #     #    crn_in = [batch, sequence(1), frame_size]
        #     crn_in = crn_in.unsqueeze(1)
        #    encoderOutput = [batchsize*num_frame, channels, features]
        encoderOutput = self.encoder(minibatch_in) # [-1]
        #    lstmInput = [batchsize, num_frame(1), channels*features]
        lstmInput = encoderOutput.view(self.batch_size, -1, self.encoder.features[-1]*self.kernel_channels[-1])
        #    rnn_output = [batchsize, num_frame(1), frame_size]
        rnn_output, hsState, hcState = self.rnn(lstmInput, hsState, hcState)
        # output = torch.cat([output, rnn_output], dim=1)
            
        return rnn_output
