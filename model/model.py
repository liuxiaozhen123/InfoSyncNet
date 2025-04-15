from .video_cnn import VideoCNN
import torch
import torch.nn as nn
import random

import matplotlib.pyplot as plt
import seaborn as sns
from torch.cuda.amp import autocast, GradScaler
from .mobilenetv3 import MobileNetV3
from .densetcn import DenseTemporalConvNet
import torch.nn.functional as F



def initialize_weight(x):
    # nn.init.xavier_uniform_(x.weight)
    # if x.bias is not None:
    #    nn.init.constant_(x.bias, 0)
    if type(x) != list:
        for name, param in x.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data)
            if 'bias' in name:
                nn.init.constant_(param.data, 0)
    else:
        for xx in x:
            for name, param in xx.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param.data)
                if 'bias' in name:
                    nn.init.constant_(param.data, 0)


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, filter_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(filter_size, hidden_size)

        initialize_weight(self.layer1)
        initialize_weight(self.layer2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, dropout_rate, marker, layer_index ,head_size=8):
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_size
        self.marker = marker

        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5
        self.layer_index = layer_index

        if self.marker == 'Transformer':
            self.q_ = nn.Linear(hidden_size, head_size * att_size, bias=False)
            self.k_ = nn.Linear(hidden_size, head_size * att_size, bias=False)
            self.v_ = nn.Linear(hidden_size, head_size * att_size, bias=False)
        elif self.marker == 'Attention_DS_TCN':
            self.q_ = nn.Linear(hidden_size, head_size * att_size, bias=False)
            self.k_ = nn.Linear(hidden_size, head_size * att_size, bias=False)
            self.v_ = nn.Linear(hidden_size, head_size * att_size, bias=False)
        else:
            self.q_ = nn.GRU(hidden_size, head_size * att_size, 1, batch_first=True, bidirectional=False,
                             dropout=0.2).to(torch.float32)
            self.k_ = nn.GRU(hidden_size, head_size * att_size, 1, batch_first=True, bidirectional=False,
                             dropout=0.2).to(torch.float32)
            self.v_ = nn.GRU(hidden_size, head_size * att_size, 1, batch_first=True, bidirectional=False,
                             dropout=0.2).to(torch.float32)

        initialize_weight(self.q_)
        initialize_weight(self.k_)
        initialize_weight(self.v_)

        self.att_dropout = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(head_size * att_size, hidden_size,
                                      bias=False)
        initialize_weight(self.output_layer)

    def forward(self, q, k, v, mask=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        if self.marker == 'Transformer':
            qq = self.q_(q).view(batch_size, -1, self.head_size, d_k)
            kk = self.k_(k).view(batch_size, -1, self.head_size, d_k)
            vv = self.v_(v).view(batch_size, -1, self.head_size, d_v)
        elif self.marker == 'Attention_DS_TCN':
            qq = self.q_(q).view(batch_size, -1, self.head_size, d_k)
            kk = self.k_(k).view(batch_size, -1, self.head_size, d_k)
            vv = self.v_(v).view(batch_size, -1, self.head_size, d_v)
        else:
            print("1111111111111111111111111111111111111111")
            flatten_parameters = [param.flatten_parameters()
                                  for param in (self.q_, self.k_, self.v_)]
            qq, _ = self.q_(q)
            kk, _ = self.k_(k)
            vv, _ = self.v_(v)
            qq = qq.clone().view(batch_size, -1, self.head_size, d_k)
            kk = kk.clone().view(batch_size, -1, self.head_size, d_k)
            vv = vv.clone().view(batch_size, -1, self.head_size, d_v)

        qq = qq.transpose(1, 2)  # [b, h, q_len, d_k]
        vv = vv.transpose(1, 2)  # [b, h, v_len, d_v]
        kk = kk.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        qq.mul_(self.scale)
        x = torch.matmul(qq, kk)  # [b, h, q_len, k_len]
        if mask is not None:
            mask = mask.unsqueeze(1).expand(batch_size, mask.size(1), mask.size(1))
            x.masked_fill_(mask.unsqueeze(1), torch.tensor(-1e9, dtype=torch.float16))

        similarity_matrix = x.detach().cpu().numpy()
        # Select the batch and head you want to visualize

        attn_scores = torch.softmax(x, dim=3)




        x = self.att_dropout(attn_scores)
        x = x.matmul(vv)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.head_size * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x, attn_scores.detach()


class MultiHeadAttentionChannel(nn.Module):
    def __init__(self, hidden_size, dropout_rate, marker, layer_index ,head_size=8):
        super(MultiHeadAttentionChannel, self).__init__()

        self.head_size = head_size
        self.marker = marker

        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5
        self.layer_index = layer_index

        if self.marker == 'Transformer':
            self.q_ = nn.Linear(hidden_size, head_size * att_size, bias=False)
            self.k_ = nn.Linear(hidden_size, head_size * att_size, bias=False)
            self.v_ = nn.Linear(hidden_size, head_size * att_size, bias=False)
        elif self.marker == 'Attention_DS_TCN':
            self.q_ = nn.Linear(hidden_size, head_size * att_size, bias=False)
            self.k_ = nn.Linear(hidden_size, head_size * att_size, bias=False)
            self.v_ = nn.Linear(hidden_size, head_size * att_size, bias=False)
        else:
            self.q_ = nn.GRU(hidden_size, head_size * att_size, 1, batch_first=True, bidirectional=False,
                             dropout=0.2).to(torch.float32)
            self.k_ = nn.GRU(hidden_size, head_size * att_size, 1, batch_first=True, bidirectional=False,
                             dropout=0.2).to(torch.float32)
            self.v_ = nn.GRU(hidden_size, head_size * att_size, 1, batch_first=True, bidirectional=False,
                             dropout=0.2).to(torch.float32)

        initialize_weight(self.q_)
        initialize_weight(self.k_)
        initialize_weight(self.v_)

        self.att_dropout = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(head_size * att_size, hidden_size,
                                      bias=False)
        initialize_weight(self.output_layer)

    def forward(self, q, k, v, mask=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        if self.marker == 'Transformer':
            qq = self.q_(q).view(batch_size, -1, self.head_size, d_k)
            kk = self.k_(k).view(batch_size, -1, self.head_size, d_k)
            vv = self.v_(v).view(batch_size, -1, self.head_size, d_v)
        elif self.marker == 'Attention_DS_TCN':
            qq = self.q_(q).view(batch_size, -1, self.head_size, d_k)
            kk = self.k_(k).view(batch_size, -1, self.head_size, d_k)
            vv = self.v_(v).view(batch_size, -1, self.head_size, d_v)
        else:
            flatten_parameters = [param.flatten_parameters()
                                  for param in (self.q_, self.k_, self.v_)]
            qq, _ = self.q_(q)
            kk, _ = self.k_(k)
            vv, _ = self.v_(v)
            qq = qq.clone().view(batch_size, -1, self.head_size, d_k)
            kk = kk.clone().view(batch_size, -1, self.head_size, d_k)
            vv = vv.clone().view(batch_size, -1, self.head_size, d_v)

        qq = qq.transpose(1, 2).transpose(2, 3)  # [b, h, d_q, q_len]
        vv = vv.transpose(1, 2).transpose(2, 3)  # [b, h, d_v, v_len]
        kk = kk.transpose(1, 2)  # [b, h, k_len, d_k]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        qq.mul_(self.scale)
        x = torch.matmul(qq, kk)  # [b, h, d_q, d_v]
        # if mask is not None:
        #     mask = mask.unsqueeze(1).expand(batch_size, mask.size(1), mask.size(1))
        #     x.masked_fill_(mask.unsqueeze(1), torch.tensor(-1e9, dtype=torch.float16))

        similarity_matrix = x.detach().cpu().numpy()
        # Select the batch and head you want to visualize

        attn_scores = torch.softmax(x, dim=3)

        batch_index = 0  # specify the batch index
        num_heads = 8  # specify the number of heads
        # print(self.layer_index)
        if self.layer_index == 5:
            plt.figure(figsize=(36, 16))  # adjust the figure size as needed
            for head_index in range(num_heads):
                plt.subplot(2, 4, head_index + 1)  # arrange the heatmaps in a 2x4 grid
                sns.heatmap(similarity_matrix[batch_index][head_index], annot=False, cmap='viridis')
                plt.title(f'Head {head_index + 1}')
                plt.xlabel('Key Length')
                plt.ylabel('Query Length')

            plt.tight_layout()
            # Define the path using raw string or double backslashes to avoid escape character issues
            path = r'F:\code\pythoncode\lip\picture\heatmap_head_{}.png'.format(head_index + 1)
            # Save the figure to a file, with a unique name for each head
            plt.savefig(path)  # Modify path as needed
            plt.show()



        x = self.att_dropout(attn_scores)
        x = x.matmul(vv)  # [b, h, d_q, attn]

        x = x.transpose(1, 3).contiguous()  # [b, attn, h, d_q]
        x = x.view(batch_size, -1, self.head_size * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x, attn_scores.detach()


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate, marker, layer_index):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.self_attention = MultiHeadAttention(hidden_size, dropout_rate, marker, layer_index)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        # self.self_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        # self.self_attentionChannel = MultiHeadAttentionChannel(hidden_size, dropout_rate, marker, layer_index)
        # self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = FeedForwardNetwork(hidden_size, filter_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask):  # pylint: disable=arguments-differ
        y = self.self_attention_norm(x)
        y, attention_scores = self.self_attention(y, y, y, mask)
        y = self.self_attention_dropout(y)
        x = x + y

        # y = self.self_attention_norm(x)
        # y, attention_scores = self.self_attentionChannel(y, y, y, mask)
        # y = self.self_attention_dropout(y)
        # x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x , attention_scores


class Encoder(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate, n_layers, marker):
        super(Encoder, self).__init__()

        encoders = [EncoderLayer(hidden_size, filter_size, dropout_rate, marker, layer_index)
                    for layer_index in range(n_layers)]
        self.layers = nn.ModuleList(encoders)

        self.last_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, inputs, mask):
        encoder_output = inputs
        for i, enc_layer in enumerate(self.layers):
            encoder_output, atten_scores_layer = enc_layer(encoder_output, mask)
            batch_index = 0

            # 指定要绘制的头
            selected_heads = [0,1,2,3,4,5,6,7]
            # 从atten_scores_layer中提取所选头的注意力得分
            selected_attention_scores = atten_scores_layer[batch_index, selected_heads, :, :]
            # print(selected_attention_scores.size())  # 8 29 29

            # 对每个头的注意力得分应用softmax进行归一化  将其中的每个注意力分数张量进行归一化，然后将归一化后的张量堆叠到一起形成一个新的张量。
            normalized_attention_scores = torch.stack([
                (head_scores - head_scores.min()) / (head_scores.max() - head_scores.min())
                for head_scores in selected_attention_scores
            ])
            # print(normalized_attention_scores.size())  # 8 29 29
            max_attention = torch.max(normalized_attention_scores, dim=0)[0]  # 对8层上的每个帧对（i，j）计算注意力分数的最大值
            # print(max_attention.size())  29 29
            # 绘制热力图
            plt.figure(figsize=(10, 8))
            from matplotlib.colors import LinearSegmentedColormap
            ocean_blue = ["#FCF3CF", "#F39C12"]  # 浅蓝到深蓝
            cmap_ocean_blue = LinearSegmentedColormap.from_list("custom_ocean_blue", ocean_blue, N=256)

            sns.heatmap(max_attention.detach().cpu().numpy(), annot=False, cmap='coolwarm', cbar=True)
            plt.title(f'', fontsize=16)
            # plt.xlabel('Frame', fontsize=14)
            # plt.ylabel('Query', fontsize=14)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)

            # 保存为PDF
            path = r'F:\code\pythoncode\lip\picture-attention-dctcn\ABSOLUTELY03_max_encoder-layer_{}.pdf'.format(i)
            plt.savefig(path, format='pdf')  # 指定保存格式为PDF
            plt.clf()

        encoder_output = self.last_norm(encoder_output)
        return encoder_output

    
def _average_batch(x, lengths, B):  # 计算每个样本在指定长度内的特征平均值
    return torch.stack( [torch.mean( x[index][:, 0:i], 1) for index, i in enumerate(lengths)],0 )
    
class DenseTCN(nn.Module):
    def __init__( self, block_config, growth_rate_set, input_size, reduced_size, num_classes,
                  kernel_size_set, dilation_size_set,
                  dropout, relu_type,
                  squeeze_excitation=False,):
        super(DenseTCN, self).__init__()

        # 计算特征数：根据网络最后一个块的配置计算得出
        num_features = reduced_size + block_config[-1]*growth_rate_set[-1]
        # 建立主要的特征提取网络
        self.tcn_trunk = DenseTemporalConvNet(block_config, growth_rate_set, input_size, reduced_size,
                                          kernel_size_set, dilation_size_set,
                                          dropout=dropout, relu_type=relu_type,
                                          squeeze_excitation=squeeze_excitation,
                                          )
        # 使用线性层将提取的特征映射到目标类别上
        self.tcn_output = nn.Linear(num_features, num_classes)
        # 定义共识函数，作为处理不同时间步长数据的函数
        self.consensus_func = _average_batch

    def forward(self, x, lengths, B):
        x = self.tcn_trunk(x)
        # print("----------------------------------")
        # print(x.size())  # 32 1664 29
        # print(self.tcn_trunk(x))
        x = self.consensus_func(x, lengths, B )  # 根据 lengths 计算平均特征
        # print("++++++++++++++++++++++++++++++")  # 1 1664
        # print(x.size())  # 32 1664
        return self.tcn_output(x)

# print(DenseTCN)


class VideoModel(nn.Module):

    def __init__(self, args, marker='Attention_DS_TCN',
                 n_layers=6,
                 hidden_size=512,
                 filter_size=2048,
                 dropout_rate=0.1,
                 densetcn_block_config=[3, 3, 3 ,3],
                 densetcn_growth_rate_set=[384, 384, 384, 384],
                 densetcn_kernel_size_set=[3, 5, 7],
                 densetcn_dilation_size_set=[1, 2, 5],
                 densetcn_reduced_size=512,
                 relu_type = "swish",
                 dense_se = True,
                 has_inputs=True,
                 src_pad_idx=None,
                 trg_pad_idx=None):
        super(VideoModel, self).__init__()

        self.args = args
        self.video_cnn = VideoCNN(se=self.args.se)
        # self.video_cnn = MobileNetV3(model_mode="LARGE", num_classes=512, multiplier=1.0)
        self.marker = marker  # GRU, Transformer, GRUTransformer

        if self.marker == 'GRU':
            if (self.args.border):
                in_dim = 512 + 1
            else:
                in_dim = 512

            self.gru = nn.GRU(in_dim, 1024, 3, batch_first=True, bidirectional=True, dropout=0.2)
        
        elif self.marker == 'DS_TCN':

            if (self.args.border):
                in_dim = 512 + 1
            else:
                in_dim = 512
            self.denseTCN = DenseTCN(block_config=densetcn_block_config, growth_rate_set=densetcn_growth_rate_set, input_size=in_dim, reduced_size=densetcn_reduced_size, num_classes=500,
                  kernel_size_set=densetcn_kernel_size_set, dilation_size_set=densetcn_dilation_size_set,
                  dropout=0.2, relu_type=relu_type,
                  squeeze_excitation=dense_se)
            
        elif self.marker == 'Attention_DS_TCN':
            self.hidden_size = hidden_size
            self.emb_scale = hidden_size ** 0.5
            self.has_inputs = has_inputs
            self.src_pad_idx = src_pad_idx
            self.trg_pad_idx = trg_pad_idx
            self.t_emb_dropout = nn.Dropout(dropout_rate)

            if has_inputs:
                self.i_emb_dropout = nn.Dropout(dropout_rate)

                self.encoder = Encoder(hidden_size, filter_size, dropout_rate,
                                       n_layers, self.marker)
            if (self.args.border):
                in_dim = 512 + 1
            else:
                in_dim = 512
            self.denseTCN = DenseTCN(block_config=densetcn_block_config, growth_rate_set=densetcn_growth_rate_set, input_size=in_dim, reduced_size=densetcn_reduced_size, num_classes=500,
                  kernel_size_set=densetcn_kernel_size_set, dilation_size_set=densetcn_dilation_size_set,
                  dropout=0.2, relu_type=relu_type,
                  squeeze_excitation=dense_se)
            # print(self.denseTCN)
        
        elif self.marker == 'Transformer':
            self.hidden_size = hidden_size
            self.emb_scale = hidden_size ** 0.5
            self.has_inputs = has_inputs
            self.src_pad_idx = src_pad_idx
            self.trg_pad_idx = trg_pad_idx

            self.t_emb_dropout = nn.Dropout(dropout_rate)

            if has_inputs:
                self.i_emb_dropout = nn.Dropout(dropout_rate)

                self.encoder = Encoder(hidden_size, filter_size, dropout_rate,
                                       n_layers, self.marker)


        self.v_cls = nn.Linear(512, self.args.n_class)
        self.dropout = nn.Dropout(p=dropout_rate)

    # def forward(self, v, border=None, mask=None):
    def forward(self, v, border=None,mask=None):

        if (self.training):
            with autocast():
                # print(v.size())
                f_v = self.video_cnn(v)
                # print(f_v.size())
                f_v = self.dropout(f_v)
            f_v = f_v.float()
        else:
            # f_v,ans_label_a,ans_label_b = self.video_cnn(v, target, lam, mode)
            f_v = self.video_cnn(v)
            f_v = self.dropout(f_v)

        if self.marker == 'GRU':
            self.gru.flatten_parameters()
            if (self.args.border):
                border = border[:, :, None]
                h, _ = self.gru(torch.cat([f_v, border], -1))
            else:
                h, _ = self.gru(f_v)

        elif self.marker == 'DS_TCN':
            B, T = f_v.size()[:2]
            lengths = [f_v.shape[1]]*B
            if (self.args.border):
                border = border[:, :, None]
                # h, _ = self.gru(torch.cat([f_v, border], -1))
                f_v = torch.cat([f_v, border], dim=-1)

            else:
                f_v = f_v

            h = self.denseTCN(f_v.transpose(1, 2), lengths, B) # TC
            # print(h.size()) # 32 1664
        
        elif self.marker == 'Attention_DS_TCN':
            B, T = f_v.size()[:2]
            lengths = [f_v.shape[1]]*B
            if self.has_inputs:
                # print(f_v.size())  # 32 29 512
                h = self.encode(f_v, mask)

                # print(h.size())
            if (self.args.border):
                border = border[:, :, None]
                # print(border.size())  # 32 29 1
                h = torch.cat([h, border], dim=-1)
                h = self.denseTCN(h.transpose(1, 2), lengths, B)  # TC
            else:
                h = self.denseTCN(h.transpose(1, 2), lengths, B)
            # print(h)

        elif self.marker=='GRUTransformer':
            if self.has_inputs:
                # i_mask = None #utils.create_pad_mask(f_v, self.src_pad_idx)
                # print('mask', mask)
                h = self.encode(f_v, mask)

        elif self.marker=='Transformer':
            if self.has_inputs:
                # i_mask = None #utils.create_pad_mask(f_v, self.src_pad_idx)
                # print('mask', mask)
                h = self.encode(f_v, mask)
        
        else:  # encoder后面直接加gru，qkv不结合GRU
            if self.has_inputs:
                # i_mask = None #utils.create_pad_mask(f_v, self.src_pad_idx)
                # print('mask', mask)
                h = self.encode(f_v, mask)
            # print(h.size())  # 具体维度需要进一步验证
            self.gru.flatten_parameters()
            if (self.args.border):
                border = border[:, :, None]
                h, _ = self.gru(torch.cat([h, border], -1))
            else:
                h, _ = self.gru(h)

        # y_v = self.v_cls(self.dropout(h)).mean(1)
        if(self.marker != 'DS_TCN' and self.marker != 'Attention_DS_TCN'):
            y_v = self.v_cls(self.dropout(h)).mean(1)
        else:
            y_v= h

        return y_v

    # 对输入数据的前期处理
    def encode(self, inputs, i_mask=None):
        # Input embedding
        # inputs (B,T,C), i_mask (B,T)
        if i_mask is not None: # 检查是否提供了掩码
            inputs.masked_fill_(i_mask.squeeze(1).unsqueeze(-1), 0)  # mask fill
        inputs *= self.emb_scale  # 调整输入维度 input scale, remove position embeding 
        inputs = self.i_emb_dropout(inputs) # 对输入做dropout操作

        return self.encoder(inputs, i_mask)
