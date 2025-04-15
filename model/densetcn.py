import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from torch.cuda.amp import autocast

from .swish import Swish



class Chomp1d(nn.Module):
    def __init__(self, chomp_size, symm_chomp):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        self.symm_chomp = symm_chomp
        if self.symm_chomp:
            assert self.chomp_size % 2 == 0, "If symmetric chomp, chomp size needs to be even"
    def forward(self, x):
        if self.chomp_size == 0:
            return x
        if self.symm_chomp:
            return x[:, :, self.chomp_size//2:-self.chomp_size//2].contiguous()
        else:
            return x[:, :, :-self.chomp_size].contiguous()


class TemporalConvLayer(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, relu_type):
        super(TemporalConvLayer, self).__init__()
        self.net = nn.Sequential(
                nn.Conv1d( n_inputs, n_outputs, kernel_size,
                           stride=stride, padding=padding, dilation=dilation),
                nn.BatchNorm1d(n_outputs),
                Chomp1d(padding, True),
                nn.PReLU(num_parameters=n_outputs) if relu_type == 'prelu' else Swish() if relu_type == 'swish' else nn.ReLU(),)

    def forward(self, x):
        return self.net(x)


class _ConvBatchChompRelu(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size_set, stride, dilation, dropout, relu_type, se_module=False):
        super(_ConvBatchChompRelu, self).__init__()

        self.num_kernels = len( kernel_size_set )
        self.n_outputs_branch = n_outputs // self.num_kernels
        assert n_outputs % self.num_kernels == 0, "Number of output channels needs to be divisible by number of kernels"

        for k_idx,k in enumerate( kernel_size_set ):
            if se_module:
                from .se_module import SELayer
                setattr( self, 'cbcr0_se_{}'.format(k_idx), SELayer( n_inputs, reduction=16))
            cbcr = TemporalConvLayer( n_inputs, self.n_outputs_branch, k, stride, dilation, (k-1)*dilation, relu_type)
            setattr( self,'cbcr0_{}'.format(k_idx), cbcr )
        self.dropout0 = nn.Dropout(dropout)
        for k_idx,k in enumerate( kernel_size_set ):
            cbcr = TemporalConvLayer( n_outputs, self.n_outputs_branch, k, stride, dilation, (k-1)*dilation, relu_type)
            setattr( self,'cbcr1_{}'.format(k_idx), cbcr )
        self.dropout1 = nn.Dropout(dropout)

        self.se_module = se_module
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

        # final relu
        if relu_type == 'relu':
            self.relu_final = nn.ReLU()
        elif relu_type == 'prelu':
            self.relu_final = nn.PReLU(num_parameters=n_outputs)
        elif relu_type == 'swish':
            self.relu_final = Swish()

    def bn_function(self, inputs):
       
        x = torch.cat(inputs, 1)
        outputs = []
        for k_idx in range( self.num_kernels ):
            if self.se_module:
                branch_se = getattr(self,'cbcr0_se_{}'.format(k_idx))
            branch_convs = getattr(self,'cbcr0_{}'.format(k_idx))
            if self.se_module:
                outputs.append( branch_convs(branch_se(x)))
            else:
                outputs.append( branch_convs(x) )
        out0 = torch.cat(outputs, 1)
        out0 = self.dropout0( out0 )
        # second multi-branch set of convolutions
        outputs = []
        for k_idx in range( self.num_kernels ):
            branch_convs = getattr(self,'cbcr1_{}'.format(k_idx))
            outputs.append( branch_convs(out0) )
        out1 = torch.cat(outputs, 1)
        out1 = self.dropout1( out1 )
        # downsample?
        res = x if self.downsample is None else self.downsample(x)
        return self.relu_final(out1 + res)

    def forward(self, input):
        if isinstance(input, torch.Tensor):
            prev_features = [input]
        else:
            prev_features = input
        bottleneck_output = self.bn_function(prev_features)
        return bottleneck_output


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__( self, num_layers, num_input_features, growth_rate,
                  kernel_size_set, dilation_size_set,
                  dropout, relu_type, squeeze_excitation,
                  ):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            dilation_size = dilation_size_set[i%len(dilation_size_set)]
            layer = _ConvBatchChompRelu(
                n_inputs=num_input_features + i * growth_rate,
                n_outputs=growth_rate,
                kernel_size_set=kernel_size_set,
                stride=1,
                dilation=dilation_size,
                dropout=dropout,
                relu_type=relu_type,
                se_module=squeeze_excitation,
                )

            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, relu_type):
        super(_Transition, self).__init__()
        self.add_module('conv', nn.Conv1d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('norm', nn.BatchNorm1d(num_output_features))
        if relu_type == 'relu':
            self.add_module('relu', nn.ReLU())
        elif relu_type == 'prelu':
            self.add_module('prelu', nn.PReLU(num_parameters=num_output_features))
        elif relu_type == 'swish':
            self.add_module('swish', Swish())

# 原始code
class DenseTemporalConvNet(nn.Module):
    def __init__(self, block_config, growth_rate_set, input_size, reduced_size,
                 kernel_size_set, dilation_size_set,
                 dropout=0.2, relu_type='prelu',
                 squeeze_excitation=False,
                 ):
        super(DenseTemporalConvNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([]))

        trans = _Transition(num_input_features=input_size,
                            num_output_features=reduced_size,
                            relu_type='prelu')
        self.features.add_module('transition%d' % (0), trans)
        num_features = reduced_size

        for i, num_layers in enumerate(block_config):

            # print(i)
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                growth_rate=growth_rate_set[i],
                kernel_size_set=kernel_size_set,
                dilation_size_set=dilation_size_set,
                dropout=dropout,
                relu_type=relu_type,
                squeeze_excitation=squeeze_excitation,
                )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate_set[i]

            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=reduced_size,
                                    relu_type=relu_type)
                self.features.add_module('transition%d' % (i + 1), trans)
                # print(i+1)
                num_features = reduced_size

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm1d(num_features))


    def forward(self, x):
        # print(x.dtype)
        features = self.features(x)
        # print(features.dtype)
        return features

# 拼接 cat

# class DenseTemporalConvNet(nn.Module):
#     def __init__(self, block_config, growth_rate_set, input_size, reduced_size,
#                  kernel_size_set, dilation_size_set,
#                  dropout=0.2, relu_type='prelu',
#                  squeeze_excitation=False,
#                  ):
#         super(DenseTemporalConvNet, self).__init__()
#         self.features = nn.Sequential(OrderedDict([]))
#
#         self.initial_transition = _Transition(num_input_features=input_size,
#                                               num_output_features=reduced_size,
#                                               relu_type='prelu')
#         self.num_features = reduced_size
#         self.blocks = nn.ModuleList()
#         self.transitions = nn.ModuleList()
#
#         for i, num_layers in enumerate(block_config):
#             block = _DenseBlock(
#                 num_layers=num_layers,
#                 num_input_features=self.num_features,
#                 growth_rate=growth_rate_set[i],
#                 kernel_size_set=kernel_size_set,
#                 dilation_size_set=dilation_size_set,
#                 dropout=dropout,
#                 relu_type=relu_type,
#                 squeeze_excitation=squeeze_excitation,
#             )
#             self.blocks.append(block)
#             self.num_features += num_layers * growth_rate_set[i] + reduced_size
#
#             if i != len(block_config) - 1:
#                 trans = _Transition(num_input_features=self.num_features,
#                                     num_output_features=reduced_size,
#                                     relu_type=relu_type)
#                 self.transitions.append(trans)
#                 self.num_features = reduced_size
#
#         self.final_norm = nn.BatchNorm1d(self.num_features-reduced_size)
#
#     def forward(self, x):
#         x = self.initial_transition(x)
#         initial_features = x  # Save the initial features after the transition layer
#
#         for i, block in enumerate(self.blocks):
#             block_output = block(x)
#             if i < len(self.blocks) - 1:  # Avoid this on last block if no transition follows
#                 x = torch.cat([initial_features, block_output], dim=1)  # Concatenate along the feature dimension
#                 x = self.transitions[i](x)
#             else:
#                 x = block_output  # No concatenation on last block if no transition
#
#         x = self.final_norm(x)
#         return x


# 残差连接实现的逻辑在代码__init__()
# class DenseTemporalConvNet(nn.Module):
#     def __init__(self, block_config, growth_rate_set, input_size, reduced_size,
#                  kernel_size_set, dilation_size_set,
#                  dropout=0.2, relu_type='prelu',
#                  squeeze_excitation=False,
#                  ):
#         super(DenseTemporalConvNet, self).__init__()
#         self.features = nn.Sequential(OrderedDict([]))
#
#         # Transition 0
#         trans0 = _Transition(num_input_features=input_size,
#                             num_output_features=reduced_size,
#                             relu_type='prelu')
#         self.features.add_module('transition0', trans0)
#         self.transition0_output = reduced_size  # Save the output size of transition0
#         num_features = reduced_size
#
#         for i, num_layers in enumerate(block_config):
#
#             block = _DenseBlock(
#                 num_layers=num_layers,
#                 num_input_features=num_features,
#                 growth_rate=growth_rate_set[i],
#                 kernel_size_set=kernel_size_set,
#                 dilation_size_set=dilation_size_set,
#                 dropout=dropout,
#                 relu_type=relu_type,
#                 squeeze_excitation=squeeze_excitation,
#                 )
#             self.features.add_module('denseblock%d' % (i + 1), block)
#             num_features = num_features + num_layers * growth_rate_set[i]
#
#             if i != len(block_config) - 1:
#                 # Transition layers with residual connection from transition0
#                 trans = _Transition(num_input_features=num_features,
#                                     num_output_features=reduced_size,
#                                     relu_type=relu_type)
#                 self.features.add_module('transition%d' % (i + 1), trans)
#
#                 # Add residual connection after this transition layer
#                 num_features = reduced_size
#
#                 # Apply residual if dimensions match (ensure they have the same size) 将前面transition0的输出与当前层的输出相加，构成残差连接
#                 if self.transition0_output == reduced_size:
#                     self.features.add_module('residual%d' % (i + 1), nn.Identity())
#
#         # Final batch norm
#         self.features.add_module('norm5', nn.BatchNorm1d(num_features))
#
#     def forward(self, x):
#         # Forward pass with residual connection
#         # print(x.dtype)
#         features = self.features(x)
#         # print(features.dtype)
#         return features




# 改进版本，出现精度损失
# class DenseTemporalConvNet(nn.Module):
#     def __init__(self, block_config, growth_rate_set, input_size, reduced_size,
#                  kernel_size_set, dilation_size_set,
#                  dropout=0.2, relu_type='prelu',
#                  squeeze_excitation=False):
#         super(DenseTemporalConvNet, self).__init__()
#         self.block_config = block_config
#         self.growth_rate_set = growth_rate_set
#         self.input_size = input_size
#         self.reduced_size = reduced_size
#         self.kernel_size_set = kernel_size_set
#         self.dilation_size_set = dilation_size_set
#         self.dropout = dropout
#         self.relu_type = relu_type
#         self.squeeze_excitation = squeeze_excitation
#
#         # Final normalization layer after all blocks
#         self.final_norm = nn.BatchNorm1d(reduced_size)
#
#
#     def forward(self, x):
#         # print(x.size())  # 32 513 29
#         # residual = x
#         # Start with transition layer 0
#         # print(x.dtype) float
#         num_features = self.input_size
#         trans = _Transition(num_input_features=num_features,
#                             num_output_features=self.reduced_size,
#                             relu_type=self.relu_type)
#         print(x.dtype)
#
#         x = trans(x)
#         print(x.dtype)
#         num_features = self.reduced_size
#
#         # Sequentially apply each DenseBlock and Transition layers
#         for i, num_layers in enumerate(self.block_config):
#             block = _DenseBlock(
#                 num_layers=num_layers,
#                 num_input_features=num_features,
#                 growth_rate=self.growth_rate_set[i],
#                 kernel_size_set=self.kernel_size_set,
#                 dilation_size_set=self.dilation_size_set,
#                 dropout=self.dropout,
#                 relu_type=self.relu_type,
#                 squeeze_excitation=self.squeeze_excitation
#             )
#             x = block(x)
#             num_features += num_layers * self.growth_rate_set[i]
#
#             if i != len(self.block_config) - 1:
#                 trans = _Transition(num_input_features=num_features,
#                                     num_output_features=self.reduced_size,
#                                     relu_type=self.relu_type)
#                 x = trans(x)
#
#                 num_features = self.reduced_size
#
#         # Apply final normalization
#         print(x.dtype)
#         x = self.final_norm(x)
#
#         return x

# GPT修改版本 待跑版本 备注： 在transition layer0之后的数据传向后续中的每个transition layer x 中，组成跳跃连接。
# class DenseTemporalConvNet(nn.Module):
#     def __init__(self, block_config, growth_rate_set, input_size, reduced_size,
#                  kernel_size_set, dilation_size_set,
#                  dropout=0.2, relu_type='prelu',
#                  squeeze_excitation=False):
#         super(DenseTemporalConvNet, self).__init__()
#         self.features = nn.Sequential(OrderedDict([]))
#         self.reduced_size = reduced_size
#         self.input_size = input_size
#
#         # Initial transition
#         trans = _Transition(num_input_features=self.input_size,
#                             num_output_features=self.reduced_size,
#                             relu_type=relu_type)
#         self.features.add_module('transition%d' % (0), trans)
#         num_features = reduced_size
#
#         # Dense blocks and transitions
#         self.blocks = nn.ModuleList()
#         self.transitions = nn.ModuleList()
#         for i, num_layers in enumerate(block_config):
#             block = _DenseBlock(
#                 num_layers=num_layers,
#                 num_input_features=num_features,
#                 growth_rate=growth_rate_set[i],
#                 kernel_size_set=kernel_size_set,
#                 dilation_size_set=dilation_size_set,
#                 dropout=dropout,
#                 relu_type=relu_type,
#                 squeeze_excitation=squeeze_excitation,
#             )
#             self.blocks.append(block)
#             num_features = num_features + num_layers * growth_rate_set[i]
#
#             if i != len(block_config) - 1:
#                 trans = _Transition(num_input_features=num_features,
#                                     num_output_features=reduced_size,
#                                     relu_type=relu_type)
#                 self.transitions.append(trans)
#                 num_features = reduced_size
#
#         # Final batch norm
#         self.norm = nn.BatchNorm1d(num_features)
#
#         # Residual connection adjustment layer
#         self.adjust_residual = nn.Conv1d(input_size, reduced_size, kernel_size=1)
#
#     def forward(self, x):
#         # Save the initial input for residual connections
#         residual = x
#         # print(x.dtype)
#
#         # Initial transition layer
#         x = self.features.transition0(x)
#         # print(x.dtype)
#         # residual = x
#         # Adjust the initial input size for residual connections
#         residual = self.adjust_residual(residual)
#
#         for i, block in enumerate(self.blocks):
#             # print(i)
#             # Pass through the dense block
#             x = block(x)
#
#             # If there is a transition layer after this block, apply it
#             if i < len(self.transitions):
#                 # print(i)
#                 # Add the residual connection from the start to the output of the transition
#                 x = self.transitions[i](x)
#                 x = x + residual  # Residual connection from the start
#
#         # Final batch normalization
#         x = self.norm(x)
# #         print("-----------------------------")
# #         print(x.dtype)
# #
#         return x
