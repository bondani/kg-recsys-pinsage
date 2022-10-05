import numpy as np
import dgl
import torch
from torch.utils.data import IterableDataset, DataLoader

def compact_and_copy(frontier, seeds):
    block = dgl.to_block(frontier, seeds)
    for col, data in frontier.edata.items():
        if col == dgl.EID:
            continue
        block.edata[col] = data[block.edata[dgl.EID]]
    return block

class UserToItemBatchSampler(IterableDataset):
    def __init__(self, g, user_type, item_type, batch_size):
        self.g = g
        self.user_type = user_type
        self.item_type = item_type
        self.user_to_item_etype = list(g.metagraph()[user_type][item_type])[0]
        self.item_to_user_etype = list(g.metagraph()[item_type][user_type])[0]
        self.batch_size = batch_size

    def __iter__(self):
        # return: batch(user, pos_item, neg_item)
        while True:
            heads = torch.randint(0, self.g.number_of_nodes(self.user_type), (self.batch_size,))
            tails = dgl.sampling.random_walk(
                self.g,
                heads,
                metapath=[self.user_to_item_etype])[0][:, 1]
            neg_tails = torch.randint(0, self.g.number_of_nodes(self.item_type), (self.batch_size,))

            mask = (tails != -1)
            yield heads[mask], tails[mask], neg_tails[mask]

class NeighborSampler(object):
    def __init__(self, g, user_type, item_type, num_neighbors, num_layers):
        self.g = g
        self.user_type = user_type
        self.item_type = item_type
        self.user_to_item_etype = list(g.metagraph()[user_type][item_type])[0]
        self.item_to_user_etype = list(g.metagraph()[item_type][user_type])[0]
        self.num_neighbors = num_neighbors
        self.num_layers = num_layers
        
    def sample_blocks(self, seeds, heads=None, tails=None, neg_tails=None):
        blocks = []
        cur = seeds
        for layer in range(self.num_layers):
            frontier = dgl.sampling.sample_neighbors(self.g, cur, fanout=self.num_neighbors)
            if heads is not None:
                eids_uv = frontier.edge_ids(torch.cat([heads, heads]), torch.cat([tails, neg_tails]), 
                                         etype=self.user_to_item_etype, return_uv=True)[2]
                eids_vu = frontier.edge_ids(torch.cat([tails, neg_tails]), torch.cat([heads, heads]), 
                                         etype=self.item_to_user_etype, return_uv=True)[2]
                if len(eids_uv) > 0:
                    old_frontier = frontier
                    frontier = dgl.remove_edges(old_frontier, eids_uv, self.user_to_item_etype)
                if len(eids_vu) > 0:
                    old_frontier = frontier
                    frontier = dgl.remove_edges(old_frontier, eids_vu, self.item_to_user_etype)
                    
                
            block = dgl.to_block(frontier, cur) 
                
            cur = {}
            for ntype in block.srctypes: 
                cur[ntype] = block.srcnodes[ntype].data[dgl.NID]
            blocks.insert(0, block)
        return blocks         
        
    def sample_from_item_pairs(self, heads, tails, neg_tails):
        # Create a graph with positive connections only and another graph with negative
        # connections only.
        pos_graph = dgl.heterograph(
            {(self.user_type, self.user_to_item_etype, self.item_type): (heads, tails),
             (self.item_type, self.item_to_user_etype, self.user_type): (tails, heads)},
            num_nodes_dict={ntype: self.g.number_of_nodes(ntype) for ntype in self.g.ntypes})
        neg_graph = dgl.heterograph(
            {(self.user_type, self.user_to_item_etype, self.item_type): (heads, neg_tails),
             (self.item_type, self.item_to_user_etype, self.user_type): (neg_tails, heads)},
            num_nodes_dict={ntype: self.g.number_of_nodes(ntype) for ntype in self.g.ntypes})
        pos_graph, neg_graph = dgl.compact_graphs([pos_graph, neg_graph])
        seeds = pos_graph.ndata[dgl.NID]

        blocks = self.sample_blocks(seeds, heads, tails, neg_tails)
        return pos_graph, neg_graph, blocks

def assign_simple_node_features(ndata, g, ntype, assign_id=False):
    """
    Copies data to the given block from the corresponding nodes in the original graph.
    """
    for col in g.nodes[ntype].data.keys():
        if not assign_id and col == dgl.NID:
            continue
        induced_nodes = ndata[ntype].data[dgl.NID]
        ndata[ntype].data[col] = g.nodes[ntype].data[col][induced_nodes]


def assign_textual_node_features(ndata, textset, ntype, empty=False):
    """
    Assigns numericalized tokens from a torchtext dataset to given block.

    The numericalized tokens would be stored in the block as node features
    with the same name as ``field_name``.

    The length would be stored as another node feature with name
    ``field_name + '__len'``.

    block : DGLHeteroGraph
        First element of the compacted blocks, with "dgl.NID" as the
        corresponding node ID in the original graph, hence the index to the
        text dataset.

        The numericalized tokens (and lengths if available) would be stored
        onto the blocks as new node features.
    textset : torchtext.data.Dataset
        A torchtext dataset whose number of examples is the same as that
        of nodes in the original graph.
    """
    node_ids = ndata[ntype].data[dgl.NID].numpy()
    if empty:
        for field_name, field in textset.fields.items():
            ndata[ntype].data[field_name] = torch.Tensor()
            ndata[ntype].data[field_name + '__len'] = torch.Tensor()
    else:
        for field_name, field in textset.fields.items():
            examples = [getattr(textset[i], field_name) for i in node_ids]
            tokens, lengths = field.process(examples)
    
            if not field.batch_first:
                tokens = tokens.t()
            
            ndata[ntype].data[field_name] = tokens
            ndata[ntype].data[field_name + '__len'] = lengths
        
def assign_features_to_blocks(blocks, g, item_ntype, user_ntype, item_textset=None, empty_dst_textset=False):
    # For the first block (which is closest to the input), copy the features from
    # the original graph as well as the texts.
    assign_simple_node_features(blocks[0].srcnodes, g, item_ntype)
    assign_simple_node_features(blocks[0].srcnodes, g, user_ntype)
    assign_simple_node_features(blocks[-1].dstnodes, g, item_ntype)
    assign_simple_node_features(blocks[-1].dstnodes, g, user_ntype)
    if item_textset is not None:
        assign_textual_node_features(blocks[0].srcnodes, item_textset, item_ntype)
        assign_textual_node_features(blocks[-1].dstnodes, item_textset, item_ntype, empty_dst_textset)

class PinSAGECollator(object):
    def __init__(self, sampler, g, item_ntype, user_ntype, textset):
        self.sampler = sampler
        self.item_ntype = item_ntype
        self.user_ntype = user_ntype
        self.g = g
        self.textset = textset

    def collate_train(self, batches):
        heads, tails, neg_tails = batches[0]
        # Construct multilayer neighborhood via PinSAGE...
        pos_graph, neg_graph, blocks = self.sampler.sample_from_item_pairs(heads, tails, neg_tails)
        assign_features_to_blocks(blocks, self.g, self.item_ntype, self.user_ntype, self.textset)

        return pos_graph, neg_graph, blocks

    def collate_items(self, samples):
        batch = torch.LongTensor(samples)
        blocks = self.sampler.sample_blocks({self.item_ntype: batch})
        assign_features_to_blocks(blocks, self.g, self.item_ntype, self.user_ntype, self.textset)

        return blocks
    
    def collate_users(self, samples):
        batch = torch.LongTensor(samples)
        blocks = self.sampler.sample_blocks({self.user_ntype: batch})
        assign_features_to_blocks(blocks, self.g, self.item_ntype, self.user_ntype, self.textset, empty_dst_textset=True)
        return blocks