import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchtext.legacy import data
import dgl
import tqdm

from layers import *
from sampler import *
import evaluation

def num_tries_gt_zero(scores, batch_size, max_trials, max_num, device):
    '''
    scores: [batch_size x N] float scores
    returns: [batch_size x 1] the lowest indice per row where scores were first greater than 0. plus 1
    '''
    tmp = scores.gt(0).nonzero().t()
    # We offset these values by 1 to look for unset values (zeros) later
    values = tmp[1] + 1
    # TODO just allocate normal zero-tensor and fill it?
    # Sparse tensors can't be moved with .to() or .cuda() if you want to send in cuda variables first
    if device.type == 'cuda':
        t = torch.cuda.sparse.LongTensor(tmp, values, torch.Size((batch_size, max_trials+1))).to_dense()
    else:
        t = torch.sparse.LongTensor(tmp, values, torch.Size((batch_size, max_trials+1))).to_dense()
    t[(t == 0)] += max_num # set all unused indices to be max possible number so its not picked by min() call

    tries = torch.min(t, dim=1)[0]
    return tries


def warp_loss(positive_predictions, negative_predictions, num_labels, device):
    '''
    positive_predictions: [batch_size x 1] floats between -1 to 1
    negative_predictions: [batch_size x N] floats between -1 to 1
    num_labels: int total number of labels in dataset (not just the subset you're using for the batch)
    device: pytorch.device
    '''
    batch_size, max_trials = negative_predictions.size(0), negative_predictions.size(1)

    offsets, ones, max_num = (torch.arange(0, batch_size, 1).long().to(device) * (max_trials + 1),
                              torch.ones(batch_size, 1).float().to(device),
                              batch_size * (max_trials + 1) )

    sample_scores = (1 + negative_predictions - positive_predictions)
    # Add column of ones so we know when we used all our attempts, This is used for indexing and computing should_count_loss if no real value is above 0
    sample_scores, negative_predictions = (torch.cat([sample_scores, ones], dim=1),
                                           torch.cat([negative_predictions, ones], dim=1))

    tries = num_tries_gt_zero(sample_scores, batch_size, max_trials, max_num, device)
    attempts, trial_offset = tries.float(), (tries - 1) + offsets
    loss_weights, should_count_loss = ( torch.log(torch.floor((num_labels - 1) / attempts)),
                                        (attempts <= max_trials).float()) #Don't count loss if we used max number of attempts

    losses = loss_weights * ((1 - positive_predictions.view(-1)) + negative_predictions.view(-1)[trial_offset]) * should_count_loss

    return losses.sum()


class PinSAGEModel(nn.Module):
    def __init__(self, full_graph, 
                 item_ntype, user_ntype, 
                 user_to_item_etype, item_to_user_etype,
                 item_textset, user_textset,
                 hidden_dims, n_layers, num_heads=2, agg_att='mean'):
        super().__init__()
        
        self.item_ntype = item_ntype
        self.user_ntype = user_ntype
        
        self.proj = LinearProjector(full_graph, item_ntype, user_ntype, 
                                    hidden_dims, item_textset, user_textset=None)
        self.net = Net(hidden_dims, num_heads, n_layers, agg_att, item_ntype, user_ntype, user_to_item_etype, item_to_user_etype)
        self.scorer = UserToItemScorer(full_graph, item_ntype, user_ntype)

    def forward(self, pos_graph, neg_graph, blocks):
        h_item, h_user = self.get_repr(blocks)
        pos_score = self.scorer(pos_graph, h_item, h_user)
        neg_score = self.scorer(neg_graph, h_item, h_user)
        return pos_score, neg_score

    def get_repr(self, blocks):
        h_item, h_user = self.proj(blocks[0].srcnodes)
        
        N_DST_ITEMS = len(blocks[-1].dstnodes[self.item_ntype].data[dgl.NID])
        N_DST_USERS = len(blocks[-1].dstnodes[self.user_ntype].data[dgl.NID])
        
        if N_DST_ITEMS*N_DST_USERS>0:
            h_item_dst, h_user_dst = self.proj(blocks[-1].dstnodes)
            new_h = self.net(blocks, h_item, h_user)
            return h_item_dst + new_h[self.item_ntype], h_user_dst + new_h[self.user_ntype]
        elif N_DST_ITEMS == 0:
            h_dst = self.proj(blocks[-1].dstnodes)
            new_h = self.net(blocks, h_item, h_user)
            return h_dst[0] + new_h[self.user_ntype]
        elif N_DST_USERS == 0:
            h_dst = self.proj(blocks[-1].dstnodes)
            new_h = self.net(blocks, h_item, h_user)
            return h_dst[0] + new_h[self.item_ntype]
            

def train(dataset, args):
    g = dataset['train-graph']
    val_matrix = dataset['val-matrix'].tocsr()
    test_matrix = dataset['test-matrix'].tocsr()
    item_texts = dataset['item-texts']
    user_ntype = dataset['user-type']
    item_ntype = dataset['item-type']
    user_to_item_etype = dataset['user-to-item-type']
    item_to_user_etype = dataset['item-to-user-type']
    timestamp = dataset['timestamp-edge-column']

    device = torch.device(args.device)

    g.nodes[user_ntype].data['id'] = torch.arange(g.number_of_nodes(user_ntype))
    g.nodes[item_ntype].data['id'] = torch.arange(g.number_of_nodes(item_ntype))

    if item_texts is not None:
        fields = {}
        examples = []
        for key, texts in item_texts.items():
            fields[key] = data.Field(include_lengths=True, lower=True, batch_first=True)
        for i in range(g.number_of_nodes(item_ntype)):
            example = data.Example.fromlist(
                [item_texts[key][i] for key in item_texts.keys()],
                [(key, fields[key]) for key in item_texts.keys()])
            examples.append(example)
        textset = data.Dataset(examples, fields)
        for key, field in fields.items():
            field.build_vocab(getattr(textset, key))
    else:
        textset = None

    batch_sampler =  UserToItemBatchSampler(g, user_ntype, item_ntype, args.batch_size)
    neighbor_sampler = NeighborSampler(g, user_ntype, item_ntype, args.num_neighbors, args.num_layers)
    collator = PinSAGECollator(neighbor_sampler, g, item_ntype, user_ntype, textset)
    dataloader = DataLoader(
        batch_sampler,
        collate_fn=collator.collate_train,
        num_workers=args.num_workers)

    dataloader_test_items = DataLoader(
        torch.arange(g.number_of_nodes(item_ntype)),
        batch_size=args.batch_size,
        collate_fn=collator.collate_items,
        num_workers=args.num_workers)

    dataloader_test_users = DataLoader(
        torch.arange(g.number_of_nodes(user_ntype)),
        batch_size=args.batch_size,
        collate_fn=collator.collate_users,
        num_workers=args.num_workers)

    dataloader_it = iter(dataloader)


    # Model
    model = PinSAGEModel(g, item_ntype, user_ntype, 
                         user_to_item_etype, item_to_user_etype,
                         textset, None, 
                         args.hidden_dims, args.num_layers, args.num_heads, args.agg_att).to(device)
    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    base_ndcg = 0
    for epoch_id in range(args.num_epochs):
        model.train()
        for batch_id in tqdm.trange(args.batches_per_epoch):
            pos_graph, neg_graph, blocks = next(dataloader_it)
            # Copy to GPU
            for i in range(len(blocks)):
                blocks[i] = blocks[i].to(device)
            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)
            pos_res, neg_res = model(pos_graph, neg_graph, blocks)
            loss = warp_loss(pos_res, neg_res, num_labels=g.number_of_nodes(item_ntype), device=torch.device('cuda'))
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        model.eval()
        with torch.no_grad():
            h_item_batches = []

            for blocks in dataloader_test_items:
                for i in range(len(blocks)):
                    blocks[i] = blocks[i].to(device)
                h_item_batches.append(model.get_repr(blocks))
            h_item = torch.cat(h_item_batches, 0)
        
            h_user_batches = []

            for blocks in dataloader_test_users:
                for i in range(len(blocks)):
                    blocks[i] = blocks[i].to(device)
                h_user_batches.append(model.get_repr(blocks))
            h_user = torch.cat(h_user_batches, 0)
            
            metrics = evaluation.evaluate_nn(dataset, h_item, h_user, args.k, args.batch_size)
            
            if metrics[0][2] - base_ndcg > -0.01:
                old_metrics = metrics
                base_ndcg = metrics[0][2]
                torch.save(model.state_dict(), args.model_output_path)
            
                print('Rec by user emb', metrics[0],
                      '\n',
                      'Rec by latest item', metrics[1])
            else:
                print('Early stopping')
                break
                
    print('RESULT')
    for k in [1, 5, 10, 15, 20]:
        metrics = evaluation.evaluate_nn(dataset, h_item, h_user, k, args.batch_size)
        
        print(f'Epoch {epoch_id}, Rec by user emb: PR@{k}: {metrics[0][0]}|REC@{k}: {metrics[0][1]}|NDCG@{k}: {metrics[0][2]}')
        print(f'Epoch {epoch_id}, Rec by latest item: PR@{k}: {metrics[1][0]}|REC@{k}: {metrics[1][1]}|NDCG@{k}: {metrics[1][2]}')
        
            

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('model_output_path', type=str)
    parser.add_argument('--random-walk-length', type=int, default=2)
    parser.add_argument('--random-walk-restart-prob', type=float, default=0.5)
    parser.add_argument('--num-random-walks', type=int, default=10)
    parser.add_argument('--num-neighbors', type=int, default=3)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--num-heads', type=int, default=2)
    parser.add_argument('--hidden-dims', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cpu')        # can also be "cuda:0"
    parser.add_argument('--num-epochs', type=int, default=1)
    parser.add_argument('--batches-per-epoch', type=int, default=20000)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--agg_att', type=str, default='mean')
    args = parser.parse_args()
    # Load dataset
    with open(args.dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    train(dataset, args)
