import logging

import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        #self.layer_norm = torch.nn.LayerNorm(dim1 + dim2)
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.1)
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        
    def forward(self, x1, x2):
        if x2 == None:
            x = x1
        else:
            x = torch.cat([x1, x2], dim=-1)
        #x = self.layer_norm(x)
        h = self.act(self.dropout(self.fc1(x)))
        return self.fc2(h)


class ScaledDotProductAttention(torch.nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn) # [n * b, l_q, l_k]
        attn = self.dropout(attn) # [n * b, l_v, d]
                
        output = torch.bmm(attn, v)
        
        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv
        if mask is not None:
            mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        # k = k * decay
        # v = v * decay

        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        #output = self.layer_norm(output)
        
        return output, attn
    

def expand_last_dim(x, num):
    view_size = list(x.size()) + [1]
    expand_size = list(x.size()) + [num]
    return x.view(view_size).expand(expand_size)


class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim, factor=5):
        super(TimeEncode, self).__init__()
        #init_len = np.array([1e8**(i/(time_dim-1)) for i in range(time_dim)])
        
        time_dim = expand_dim
        self.factor = factor
        
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())
        self.time_decay = torch.nn.Parameter(torch.ones(time_dim).float())
        
        
    def forward(self, ts, time_decay=False):
        # ts: [N, L]
        batch_size = ts.size(0)
        seq_len = ts.size(1)
                
        ts = ts.view(batch_size, seq_len, 1)# [N, L, 1]
        ##
        if time_decay:
            ts = ts * self.time_decay.view(1, 1, -1)
        ##
        map_ts = ts * self.basis_freq.view(1, 1, -1) # [N, L, time_dim]
        map_ts += self.phase.view(1, 1, -1)

        harmonic = torch.cos(map_ts)

        
        # if time_decay:
        #     # freq_decay = torch.cos(2 * math.pi * self.basis_freq * self.time_decay)
        #     freq_decay = self.time_decay
        #     harmonic = harmonic * freq_decay.view(1, 1, -1)

        return harmonic #self.dense(harmonic)
    
    
    
class PosEncode(torch.nn.Module):
    def __init__(self, expand_dim, seq_len):
        super().__init__()
        
        self.pos_embeddings = nn.Embedding(num_embeddings=seq_len, embedding_dim=expand_dim)
        
    def forward(self, ts, time_decay=False):
        # ts: [N, L]
        order = ts.argsort() # argsort(): 返回数组元素从小到大的索引值
        ts_emb = self.pos_embeddings(order)
        return ts_emb
    

class EmptyEncode(torch.nn.Module):
    def __init__(self, expand_dim):
        super().__init__()
        self.expand_dim = expand_dim
        
    def forward(self, ts, time_decay=False):
        out = torch.zeros_like(ts).float()
        out = torch.unsqueeze(out, dim=-1)
        out = out.expand(out.shape[0], out.shape[1], self.expand_dim)
        return out


class LSTMPool(torch.nn.Module):
    def __init__(self, feat_dim, edge_dim, time_dim):
        super(LSTMPool, self).__init__()
        self.feat_dim = feat_dim
        self.time_dim = time_dim
        self.edge_dim = edge_dim
        
        self.att_dim = feat_dim + edge_dim + time_dim
        
        self.act = torch.nn.ReLU()
        
        self.lstm = torch.nn.LSTM(input_size=self.att_dim, 
                                  hidden_size=self.feat_dim, 
                                  num_layers=1, 
                                  batch_first=True)
        self.merger = MergeLayer(feat_dim, feat_dim, feat_dim, feat_dim)

    def forward(self, src, src_t, seq, seq_t, seq_e, mask):
        # seq [B, N, D]
        # mask [B, N]
        seq_x = torch.cat([seq, seq_e, seq_t], dim=2)
            
        _, (hn, _) = self.lstm(seq_x)
        
        hn = hn[-1, :, :] #hn.squeeze(dim=0)

        out = self.merger.forward(hn, src)
        return out, None
    

class MeanPool(torch.nn.Module):
    def __init__(self, feat_dim, edge_dim):
        super(MeanPool, self).__init__()
        self.edge_dim = edge_dim
        self.feat_dim = feat_dim
        self.act = torch.nn.ReLU()
        self.merger = MergeLayer(edge_dim + feat_dim, feat_dim, feat_dim, feat_dim)
        
    def forward(self, src, src_t, src_s, seq, seq_t, seq_e, mask):
        # seq [B, N, D]
        # mask [B, N]
        src_x = src
        seq_x = torch.cat([seq, seq_e], dim=2) #[B, N, De + D]
        hn = seq_x.mean(dim=1) #[B, De + D]
        output = self.merger(hn, src_x)
        return output, None
    

class AttnModel(torch.nn.Module):
    """Attention based temporal layers
    """
    def __init__(self, feat_dim, edge_dim, time_dim, 
                 attn_mode='prod', n_head=2, drop_out=0.1):
        """
        args:
          feat_dim: dim for the node features
          edge_dim: dim for the temporal edge features
          time_dim: dim for the time encoding
          attn_mode: choose from 'prod' and 'map'
          n_head: number of heads in attention
          drop_out: probability of dropping a neural.
        """
        super(AttnModel, self).__init__()
        
        self.feat_dim = feat_dim
        self.time_dim = time_dim
        
        self.edge_in_dim = (feat_dim + edge_dim + time_dim)
        self.model_dim = self.edge_in_dim
        #self.edge_fc = torch.nn.Linear(self.edge_in_dim, self.feat_dim, bias=False)

        self.merger = MergeLayer(self.model_dim, feat_dim, feat_dim, feat_dim)

        #self.act = torch.nn.ReLU()
        
        assert(self.model_dim % n_head == 0)
        self.logger = logging.getLogger(__name__)
        self.attn_mode = attn_mode
        
        if attn_mode == 'prod':
            self.multi_head_target = MultiHeadAttention(n_head, 
                                             d_model=self.model_dim, 
                                             d_k=self.model_dim // n_head, 
                                             d_v=self.model_dim // n_head, 
                                             dropout=drop_out)
            self.logger.info('Using scaled prod attention')
            
        elif attn_mode == 'map':
            self.multi_head_target = MapBasedMultiHeadAttention(n_head, 
                                             d_model=self.model_dim, 
                                             d_k=self.model_dim // n_head, 
                                             d_v=self.model_dim // n_head, 
                                             dropout=drop_out)
            self.logger.info('Using map based attention')
        else:
            raise ValueError('attn_mode can only be prod or map')
        
        
    def forward(self, src, src_t, src_s, seq, seq_t, seq_e, mask):
        """"Attention based temporal attention forward pass
        args:
          src: float Tensor of shape [B, D]
          src_t: float Tensor of shape [B, Dt], Dt == D
          seq: float Tensor of shape [B, N, D]
          seq_t: float Tensor of shape [B, N, Dt]
          seq_e: float Tensor of shape [B, N, De], De == D
          mask: boolean Tensor of shape [B, N], where the true value indicate a null value in the sequence.

        returns:
          output, weight

          output: float Tensor of shape [B, D]
          weight: float Tensor of shape [B, N]
        """

        src_ext = torch.unsqueeze(src, dim=1) # src [B, 1, D]
        #src_e_ph = torch.zeros_like(src_ext) 
        src_e_ph = torch.unsqueeze(src_s, dim=1)
        # seq_e.min(dim=1, keepdim=True)
        q = torch.cat([src_ext, src_e_ph, src_t], dim=2) # [B, 1, D + De + Dt] -> [B, 1, D]
        k = torch.cat([seq, seq_e, seq_t], dim=2) # [B, N, D + De + Dt] -> [B, N, D]
        
        mask = torch.unsqueeze(mask, dim=2) # mask [B, N, 1]
        mask = mask.permute([0, 2, 1]) #mask [B, 1, N]

        #decay = decay.permute([0, 2, 1]) #decay [B, 1, N]

        # # target-attention
        output, attn = self.multi_head_target(q=q, k=k, v=k, mask=mask) # output: [B, 1, D + Dt], attn: [B, 1, N]
        output = output.squeeze()
        attn = attn.squeeze()

        output = self.merger(output, src)
        return output, attn


class ConvPool(nn.Module):
    def __init__(self, in_channels, out_channels, feat_dim, n_head, device):
        super(ConvPool, self).__init__()
        self.conv_1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        self.dim = feat_dim
        self.device = device
        self.layernorm = nn.LayerNorm(feat_dim)

    def forward(self, x):
        # x shape: [num_nodes, dim]
        residual = x
        x = x.unsqueeze(0) 
        x = x.transpose(1, 2)
        x = self.conv_1(x)
        x = x.transpose(1, 2)
        x = F.relu(x)
        output = self.layernorm(x + residual)

        return output.squeeze(0)

    
class global_attention(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super(global_attention, self).__init__()
        self.linear1 = nn.Linear(in_features, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)
    def forward(self, x):
        energy = self.linear2(x)
        alpha = F.softmax(energy, dim=1)
        attended_x = torch.sum(alpha * x, dim=1) 
        return attended_x

        
class TGAN(torch.nn.Module):
    def __init__(self, ngh_finder, node_num, edge_num, feat_dim, device,
                 attn_mode='prod', use_time='time', agg_method='attn',
                 num_layers=3, n_head=4, null_idx=0, num_heads=1, drop_out=0.1, decay=-1, seq_len=None):
        super(TGAN, self).__init__()
        #self.conv_pool = torch.nn.ModuleList([ConvPool(feat_dim, feat_dim, feat_dim, num_heads, device) for _ in range(num_layers)])
        self.node_num = node_num
        self.edge_num = edge_num
        self.feat_dim = feat_dim
        self.num_layers = num_layers 
        self.ngh_finder = ngh_finder
        self.null_idx = null_idx
        self.device = device
        self.logger = logging.getLogger(__name__)
       
        self.n_feat_dim = self.feat_dim
        self.e_feat_dim = self.feat_dim
        self.model_dim = self.feat_dim

        self.node_embed = nn.Sequential(
            nn.Embedding(self.node_num, self.n_feat_dim),
            nn.Dropout(drop_out)
        )

        self.edge_embed = nn.Sequential(
            torch.nn.Embedding(self.edge_num, self.e_feat_dim),
            nn.Dropout(drop_out)
        )

        self.use_time = use_time
        self.merge_layer_list = torch.nn.ModuleList([MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, self.feat_dim) for _ in range(num_layers)])
        self.att_affine_list = torch.nn.ModuleList([MergeLayer(self.feat_dim, 0, self.feat_dim // 2, 1) for _ in range(num_layers)])

        self.output = MergeLayer(self.feat_dim, 0, self.feat_dim // 2, 1)
        self.layer_weight = nn.Parameter(torch.ones(num_layers).float())
        

        nn.init.xavier_normal_(self.node_embed[0].weight)
        nn.init.xavier_normal_(self.edge_embed[0].weight)

        if agg_method == 'attn':
            self.logger.info('Aggregation uses attention model')
            self.attn_model_list = torch.nn.ModuleList([AttnModel(self.feat_dim, 
                                                               self.feat_dim, 
                                                               self.feat_dim,
                                                               attn_mode=attn_mode, 
                                                               n_head=n_head, 
                                                               drop_out=drop_out) for _ in range(num_layers)])
        elif agg_method == 'lstm':
            self.logger.info('Aggregation uses LSTM model')
            self.attn_model_list = torch.nn.ModuleList([LSTMPool(self.feat_dim,
                                                                 self.feat_dim,
                                                                 self.feat_dim) for _ in range(num_layers)])
        elif agg_method == 'mean':
            self.logger.info('Aggregation uses constant mean model')
            self.attn_model_list = torch.nn.ModuleList([MeanPool(self.feat_dim,
                                                                 self.feat_dim) for _ in range(num_layers)])
        else:
        
            raise ValueError('invalid agg_method value, use attn or lstm')
        
        
        if use_time == 'time':
            self.logger.info('Using time encoding')
            self.time_encoder = TimeEncode(expand_dim=self.feat_dim)
        elif use_time == 'pos':
            assert(seq_len is not None)
            self.logger.info('Using positional encoding')
            self.time_encoder = PosEncode(expand_dim=self.feat_dim, seq_len=seq_len)
        elif use_time == 'empty':
            self.logger.info('Using empty encoding')
            self.time_encoder = EmptyEncode(expand_dim=self.feat_dim)
        else:
            raise ValueError('invalid time option!')
        
        self.affinity_score = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, 1) 

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight'):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif "layer_norm" in pn or "embed" in pn:
                    # weights of blacklist modules will NOT be weight decayed
                    if fpn in decay:
                        decay.remove(fpn)
                    no_decay.add(fpn)
                else:
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.lr)
        return optimizer
    

    def forward(self, cas_l, src_idx_l, target_idx_l, cut_time_l, e_l, num_neighbors=20):
        src_list, target_list = [], []
        src_embed = self.tem_conv(src_list, cas_l, src_idx_l, cut_time_l, e_l, self.num_layers, num_neighbors)
        target_embed = self.tem_conv(target_list, cas_l, target_idx_l, cut_time_l, e_l, self.num_layers, num_neighbors)
        # graph pooling

        src_list = [src_list[0], src_list[2]]
        target_list = [target_list[0], target_list[2]]
        embed_list = []
        att = 0
        for i in range(len(src_list)):
            embed = self.merge_layer_list[i](src_list[i], target_list[i])
            #embed = self.conv_pool[i](embed)
            att_score = F.softmax(self.att_affine_list[i](embed, None), dim=0)
            if i == 1:
                att = att_score
            att_embed = torch.sum(embed * att_score, dim=0)
            embed_list.append(att_embed)
        embed = torch.stack(embed_list, dim=0)
        weighted_embed = torch.sum(embed * self.layer_weight.view(-1, 1), dim=0, keepdim=True)
        
        # embed = self.merge_layer_list[0](src_embed, target_embed)
        # att_score = F.softmax(self.att_affine_list[0](embed, None), dim=0)
        # weighted_embed = torch.sum(embed * att_score, dim=0)

        
        g_score = self.output(weighted_embed, None)

        g_score = torch.clamp(g_score, min=0)
        return g_score, att

    def tem_conv(self, feat_l, cas_l, src_idx_l, cut_time_l, e_l, curr_layers, num_neighbors=20):
        assert(curr_layers >= 0)
    
        device = self.device
    
        batch_size = len(src_idx_l)
        
        src_node_batch_th = torch.from_numpy(src_idx_l).long().to(device)
        cut_time_l_th = torch.from_numpy(cut_time_l).float().to(device)
        cut_time_l_th = torch.unsqueeze(cut_time_l_th, dim=1)
        # query node always has the start time -> time span == 0
        src_node_t_embed = self.time_encoder(torch.zeros_like(cut_time_l_th), time_decay=True)
        src_node_feat = self.node_embed(src_node_batch_th)
        #
        src_e_batch = torch.from_numpy(e_l).long().to(device)
        src_node_edge_feat = self.edge_embed(torch.zeros_like(src_e_batch))
        if curr_layers == 0:
            return src_node_feat
        else:
            src_node_conv_feat = self.tem_conv(feat_l,
                                               cas_l,
                                           src_idx_l, 
                                           cut_time_l,
                                           e_l,
                                           curr_layers=curr_layers - 1, 
                                           num_neighbors=num_neighbors)
            src_ngh_node_batch, src_ngh_eidx_batch, src_ngh_t_batch = self.ngh_finder.get_temporal_neighbor(
                                                                    cas_l,
                                                                    src_idx_l, 
                                                                    cut_time_l, 
                                                                    num_neighbors=num_neighbors)

            src_ngh_node_batch_th = torch.from_numpy(src_ngh_node_batch).long().to(device)
            

            
            src_ngh_t_batch_delta = cut_time_l[:, np.newaxis] - src_ngh_t_batch
            src_ngh_eidx_batch_delta = e_l[:, np.newaxis] - src_ngh_eidx_batch
            src_ngh_eidx_batch_delta[src_ngh_eidx_batch_delta < 0] = 0
            
            src_ngh_eidx_batch_th = torch.from_numpy(src_ngh_eidx_batch_delta).long().to(device)
            src_ngh_t_batch_th = torch.from_numpy(src_ngh_t_batch_delta).float().to(device)
            #src_ngh_t_batch_th = torch.as_tensor(src_ngh_t_batch_delta, dtype=torch.float32).to(device)

            # get previous layer's node features
            src_ngh_node_batch_flat = src_ngh_node_batch.flatten() #reshape(batch_size, -1)
            src_ngh_t_batch_flat = src_ngh_t_batch.flatten() #reshape(batch_size, -1)  
            src_ngh_e_batch_flat = src_ngh_eidx_batch.flatten()
            src_ngh_node_conv_feat = self.tem_conv(feat_l,
                                                   cas_l,
                                                   src_ngh_node_batch_flat, 
                                                   src_ngh_t_batch_flat,
                                                   src_ngh_e_batch_flat,
                                                   curr_layers=curr_layers - 1, 
                                                   num_neighbors=num_neighbors)
            src_ngh_feat = src_ngh_node_conv_feat.view(batch_size, num_neighbors, -1)
            
            # get edge time features and node features
            #normed_src_ngh_t_batch_th = (src_ngh_t_batch_th - torch.mean(src_ngh_t_batch_th)) / torch.std(src_ngh_t_batch_th)
            src_ngh_t_embed = self.time_encoder(src_ngh_t_batch_th, time_decay=True) # B, N, DIM


            #src_ngn_edge_feat = torch.zeros_like(src_ngh_feat)   
            # add scale features 
            src_ngn_edge_feat = self.edge_embed(src_ngh_eidx_batch_th)
            # attention aggregation
            mask = src_ngh_node_batch_th == 0
            attn_m = self.attn_model_list[curr_layers - 1]
                        
            local, weight = attn_m(src_node_conv_feat, 
                                   src_node_t_embed,
                                   src_node_edge_feat,
                                   src_ngh_feat,
                                   src_ngh_t_embed, 
                                   src_ngn_edge_feat, 
                                   mask)
            feat_l.append(local)
            return local
