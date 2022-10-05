import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
import dgl.function as fn
from dgl.nn import GATConv


def disable_grad(module):
    for param in module.parameters():
        param.requires_grad = False


def _init_input_modules(g, ntypes, hidden_dims, textset=None):
    # We initialize the linear projections of each input feature ``x`` as
    # follows:
    # * If ``x`` is a scalar integral feature, we assume that ``x`` is a categorical
    #   feature, and assume the range of ``x`` is 0..max(x).
    # * If ``x`` is a float one-dimensional feature, we assume that ``x`` is a
    #   numeric vector.
    # * If ``x`` is a field of a textset, we process it as bag of words.
    module_dicts = []
    for ntype in ntypes:
        module_dict = nn.ModuleDict()
    
        for column, data in g.nodes[ntype].data.items():
            if (column == dgl.NID) or (column == 'id') or (column == 'ID'):
                continue
            if data.dtype == torch.float32:
                assert data.ndim == 2
                m = nn.Linear(data.shape[1], hidden_dims)
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
                module_dict[column] = m
            elif data.dtype == torch.int64:
                assert data.ndim == 1
                m = nn.Embedding(
                    data.max() + 2, hidden_dims, padding_idx=-1)
                nn.init.xavier_uniform_(m.weight)
                module_dict[column] = m
    
        if textset[ntype] is not None:
            for column, field in textset[ntype].fields.items():
                if field.vocab.vectors:
                    module_dict[column] = BagOfWordsPretrained(field, hidden_dims)
                else:
                    module_dict[column] = BagOfWords(field, hidden_dims)
                    
        module_dicts.append(module_dict)            
    return module_dicts


class BagOfWordsPretrained(nn.Module):
    def __init__(self, field, hidden_dims):
        super().__init__()

        input_dims = field.vocab.vectors.shape[1]
        self.emb = nn.Embedding(
            len(field.vocab.itos), input_dims,
            padding_idx=field.vocab.stoi[field.pad_token])
        self.emb.weight[:] = field.vocab.vectors
        self.proj = nn.Linear(input_dims, hidden_dims)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0)

        disable_grad(self.emb)

    def forward(self, x, length):
        """
        x: (batch_size, max_length) LongTensor
        length: (batch_size,) LongTensor
        """
        x = self.emb(x).sum(1) / length.unsqueeze(1).float()
        return self.proj(x)


class BagOfWords(nn.Module):
    def __init__(self, field, hidden_dims):
        super().__init__()

        self.emb = nn.Embedding(
            len(field.vocab.itos), hidden_dims,
            padding_idx=field.vocab.stoi[field.pad_token])
        nn.init.xavier_uniform_(self.emb.weight)

    def forward(self, x, length):
        return self.emb(x).sum(1) / length.unsqueeze(1).float()


class LinearProjector(nn.Module):
    """
    Projects each input feature of the graph linearly and sums them up
    """
    def __init__(self, full_graph, item_ntype, user_ntype, hidden_dims, item_textset=None, user_textset=None):
        super().__init__()
        
        textset = dict()
        textset[item_ntype] = item_textset
        textset[user_ntype] = user_textset
            
        self.item_ntype = item_ntype
        self.user_ntype = user_ntype
        
        self.item_inputs, self.user_inputs = _init_input_modules(full_graph, [item_ntype, user_ntype], hidden_dims, textset)

    def forward(self, ndata):
        projections = []
        
        ntypes, inputs = [], []
        if len(ndata[self.item_ntype].data[dgl.NID]) > 0:
            ntypes.append(self.item_ntype)
            inputs.append(self.item_inputs)
        if len(ndata[self.user_ntype].data[dgl.NID]) > 0:
            ntypes.append(self.user_ntype)
            inputs.append(self.user_inputs)
         
        for ntype, inp in zip(ntypes, inputs):
            projection = []
            
            for feature in inp.keys():
                data = ndata[ntype].data[feature]
                if feature == dgl.NID or feature == 'id' or feature.endswith('__len'):
                    continue
                module = inp[feature]
                if isinstance(module, (BagOfWords, BagOfWordsPretrained)):
                    length = ndata[ntype].data[feature + '__len']
                    result = module(data, length)
                else:
                    result = module(data)
                projection.append(result)
                
            if len(torch.stack(projection, 1).sum(1))>0:
                projections.append(torch.stack(projection, 1).sum(1))
            
        return projections


class Layer(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, n_heads, 
                 item_ntype, user_ntype,
                 user_to_item_etype, item_to_user_etype,
                 agg_att='mean'):
        super(Layer, self).__init__()
        
        # 2 nets
        #self.conv = dglnn.HeteroGraphConv({
        #                user_to_item_etype: GATConv(input_dims, hidden_dims, num_heads=n_heads, feat_drop=0.3, attn_drop=0.3),
        #                item_to_user_etype: GATConv(input_dims, hidden_dims, num_heads=n_heads, feat_drop=0.3, attn_drop=0.3)},
        #                aggregate='sum')
        
        self.agg_att = agg_att
        
        att_conv = GATConv(input_dims, hidden_dims, num_heads=n_heads, feat_drop=0.3, attn_drop=0.3)
        
        self.conv = dglnn.HeteroGraphConv({
                        user_to_item_etype: att_conv,
                        item_to_user_etype: att_conv},
                        aggregate='sum')
        
        if agg_att == 'mean':
            self.W = nn.Linear(input_dims + hidden_dims, output_dims)
        elif agg_att == 'concat':
            self.W = nn.Linear(input_dims*n_heads + hidden_dims, output_dims)
        
        
        self.reset_parameters()
        
        self.dropout = nn.Dropout(0.5)
        
        self.item_ntype = item_ntype
        self.user_ntype = user_ntype
        
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W.weight, gain=gain)
        nn.init.constant_(self.W.bias, 0)
        
    
    def forward(self, block, h):
        h_src, h_dst = h
        with block.local_scope():

            new_h_dst = self.conv(block, h_src)
            
            z = dict()
            
            if self.user_ntype in h_dst.keys():
                if self.agg_att == 'mean':
                    new_h_dst[self.user_ntype] = torch.mean(new_h_dst[self.user_ntype], 1)
                elif self.agg_att == 'concat':
                    new_h_dst[self.user_ntype] = new_h_dst[self.user_ntype].reshape(new_h_dst[self.user_ntype].size()[0], -1)
                z_user = F.relu(self.W(self.dropout(torch.cat([new_h_dst[self.user_ntype], h_dst[self.user_ntype]], 1))))
                z_user_norm = z_user.norm(2, 1, keepdim=True)
                z_user_norm = torch.where(z_user_norm == 0, torch.tensor(1.).to(z_user_norm), z_user_norm)
                z_user = z_user / z_user_norm
                z[self.user_ntype] = z_user
                
            if self.item_ntype in h_dst.keys():
                if self.agg_att == 'mean':
                    new_h_dst[self.item_ntype] = torch.mean(new_h_dst[self.item_ntype], 1)
                elif self.agg_att == 'concat':
                    new_h_dst[self.item_ntype] = new_h_dst[self.item_ntype].reshape(new_h_dst[self.item_ntype].size()[0], -1)
                z_item = F.relu(self.W(self.dropout(torch.cat([new_h_dst[self.item_ntype], h_dst[self.item_ntype]], 1))))
                z_item_norm = z_item.norm(2, 1, keepdim=True)
                z_item_norm = torch.where(z_item_norm == 0, torch.tensor(1.).to(z_item_norm), z_item_norm)
                z_item = z_item / z_item_norm
                z[self.item_ntype] = z_item
            
            return z


class Net(nn.Module):
    def __init__(self, hidden_dims, num_heads, num_layers, agg_att,
                 item_ntype, user_ntype, user_to_item_etype, item_to_user_etype):
        super(Net, self).__init__()
        
        self.item_ntype = item_ntype
        self.user_ntype = user_ntype
        
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(Layer(hidden_dims, hidden_dims, hidden_dims, num_heads, item_ntype, user_ntype, user_to_item_etype, item_to_user_etype, agg_att))
            

    def forward(self, blocks, h_item, h_user):
        h_src = dict({self.user_ntype: h_user, self.item_ntype: h_item})
        for layer, block in zip(self.convs, blocks):

            N_DST_ITEMS = len(block.dstnodes[self.item_ntype].data[dgl.NID])
            N_DST_USERS = len(block.dstnodes[self.user_ntype].data[dgl.NID])
            if (N_DST_ITEMS > 0) & (N_DST_USERS > 0):
                h_dst = {self.user_ntype: h_src[self.user_ntype][:block.number_of_nodes('DST/' + self.user_ntype)],
                         self.item_ntype: h_src[self.item_ntype][:block.number_of_nodes('DST/' + self.item_ntype)]}
                h_src = layer(block, (h_src, h_dst))
            elif N_DST_ITEMS == 0:
                h_dst = {self.user_ntype: h_src[self.user_ntype][:block.number_of_nodes('DST/' + self.user_ntype)]}
                h_src = layer(block, (h_src, h_dst))
            elif N_DST_USERS == 0:
                h_dst = {self.item_ntype: h_src[self.item_ntype][:block.number_of_nodes('DST/' + self.item_ntype)]}
                h_src = layer(block, (h_src, h_dst))
        return h_src
            


class UserToItemScorer(nn.Module):
    def __init__(self, full_graph, item_ntype, user_ntype):
        super().__init__()

        self.item_ntype = item_ntype
        self.user_ntype = user_ntype
        item_n_nodes = full_graph.number_of_nodes(item_ntype)
        user_n_nodes = full_graph.number_of_nodes(user_ntype)
        self.bias = nn.Parameter(torch.zeros(item_n_nodes+user_n_nodes, 1))

    def _add_bias(self, edges):
        bias_src = self.bias[edges.src[dgl.NID]]
        bias_dst = self.bias[edges.dst[dgl.NID]]
        return {'s': edges.data['s'] + bias_src + bias_dst}

    def forward(self, user_item_graph, h_item, h_user):
        user_to_item_etype = list(user_item_graph.metagraph()[self.user_ntype][self.item_ntype])[0]
        with user_item_graph.local_scope():
            #user_item_graph.ndata['h'] = {self.item_ntype: F.normalize(h_item), self.user_ntype: F.normalize(h_user)}
            user_item_graph.ndata['h'] = {self.item_ntype: h_item, self.user_ntype: h_user}
            user_item_graph.apply_edges(fn.u_dot_v('h', 'h', 's'), etype=user_to_item_etype)
            user_item_graph.apply_edges(self._add_bias, etype=user_to_item_etype)
            pair_score = list(user_item_graph.edata['s'].values())[0]
        return pair_score