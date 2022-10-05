import numpy as np
import torch
import torch.nn.functional as F
import pickle
import dgl
import argparse

def calc_precision(hits, k):
    return hits[:, :k].mean(axis=1)

def calc_recall(hits, interactions, k):
    return hits[:, :k].sum(axis=1) / interactions.sum(axis=1)

def calc_dcg(hits, k):
    return np.sum((2 ** hits[:, :k] - 1) / np.log2(np.arange(2, k + 2)), axis=1)

def calc_ndcg(hits, k):
    dcg = calc_dcg(hits, k)
    sorted_hits = np.flip(np.sort(hits))
    idcg = calc_dcg(sorted_hits, k)
    idcg[idcg == 0] = np.inf
    return dcg / idcg


def k_metrics(recommendations, interactions, k):
    
    #_, sorted_items = torch.sort(recommendations, descending=True)
    #sorted_items = sorted_items.numpy()
    
    hits = np.zeros_like(recommendations)
    mask = []
    for idx, items in enumerate(recommendations):
        if interactions[idx].sum() > 0:
            hits[idx, :] = interactions[idx, items]
            mask.append(True)
        else:
            mask.append(False)
        
    pr = sum(calc_precision(hits[mask], k))/interactions[mask].shape[0]
    rec = sum(calc_recall(hits[mask], interactions[mask], k))/interactions[mask].shape[0]
    ndcg = sum(calc_ndcg(hits[mask], k))/interactions[mask].shape[0]

    return pr, rec, ndcg


class NNRecommender(object):
    def __init__(self, user_ntype, item_ntype, timestamp, batch_size):
        self.user_ntype = user_ntype
        self.item_ntype = item_ntype
        self.batch_size = batch_size
        self.timestamp = timestamp

    def recommend(self, full_graph, h_user, h_item, K):
        """
        Return a (n_user, n_items) matrix of recommended items for each user
        """
        user_to_item_etype = list(full_graph.metagraph()[self.user_ntype][self.item_ntype])[0]
        n_users = full_graph.number_of_nodes(self.user_ntype)
        recommended_batches = []
        user_batches = torch.arange(n_users).split(self.batch_size)
        for user_batch in user_batches:
            h_user_batch = h_user[user_batch]
            #dist = F.normalize(h_user_batch) @ F.normalize(h_item).t()
            dist = h_user_batch @ h_item.t()
            for i, u in enumerate(user_batch.tolist()):
                interacted_items = full_graph.successors(u, etype=user_to_item_etype)
                dist[i, interacted_items] = -float('inf')
            recommended_batches.append(dist.topk(K, 1)[1])
            #recommended_batches.append(dist.cpu().numpy())
        recommendations = torch.cat(recommended_batches, 0)
        #recommendations = np.concatenate(recommended_batches, axis=0)
        return recommendations
    

class LatestNNRecommender(object):
    def __init__(self, user_ntype, item_ntype, user_to_item_etype, timestamp, batch_size):
        self.user_ntype = user_ntype
        self.item_ntype = item_ntype
        self.user_to_item_etype = user_to_item_etype
        self.batch_size = batch_size
        self.timestamp = timestamp

    def recommend(self, full_graph, h_item, K):
        """
        Return a (n_user, n_items) matrix of recommended items for each user
        """
        graph_slice = full_graph.edge_type_subgraph([self.user_to_item_etype])
        n_users = full_graph.number_of_nodes(self.user_ntype)
        latest_interactions = dgl.sampling.select_topk(graph_slice, 1, self.timestamp, edge_dir='out')
        user, latest_items = latest_interactions.all_edges(form='uv', order='srcdst')
        # each user should have at least one "latest" interaction
        assert torch.equal(user, torch.arange(n_users))

        recommended_batches = []
        user_batches = torch.arange(n_users).split(self.batch_size)
        for user_batch in user_batches:
            latest_item_batch = latest_items[user_batch].to(device=h_item.device)
            dist = h_item[latest_item_batch] @ h_item.t()
            # exclude items that are already interacted
            for i, u in enumerate(user_batch.tolist()):
                interacted_items = full_graph.successors(u, etype=self.user_to_item_etype)
                dist[i, interacted_items] = -np.inf
            recommended_batches.append(dist.topk(K, 1)[1])

        recommendations = torch.cat(recommended_batches, 0)
        return recommendations


def evaluate_nn(dataset, h_item, h_user, k, batch_size):
    g = dataset['train-graph']
    val_matrix = dataset['val-matrix'].tocsr()
    test_matrix = dataset['test-matrix'].tocsr()
    item_texts = dataset['item-texts']
    user_ntype = dataset['user-type']
    item_ntype = dataset['item-type']
    user_to_item_etype = dataset['user-to-item-type']
    timestamp = dataset['timestamp-edge-column']

    rec_engine1 = NNRecommender(user_ntype, item_ntype, timestamp, batch_size)
    recommendations1 = rec_engine1.recommend(g, h_user, h_item, k).cpu().numpy()
    
    rec_engine2 = LatestNNRecommender(user_ntype, item_ntype, user_to_item_etype, timestamp, batch_size)
    recommendations2 = rec_engine2.recommend(g, h_item, k).cpu().numpy()
    
    
    return [k_metrics(recommendations1, val_matrix.toarray(), k),  k_metrics(recommendations2, val_matrix.toarray(), k)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('item_embedding_path', type=str)
    parser.add_argument('user_embedding_path', type=str)
    parser.add_argument('-k', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    with open(args.dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    with open(args.item_embedding_path, 'rb') as f:
        item_emb = torch.FloatTensor(pickle.load(f))
    with open(args.user_embedding_path, 'rb') as f:
        user_emb = torch.FloatTensor(pickle.load(f))
    print(evaluate_nn(dataset, item_emb, user_emb, args.k, args.batch_size))
