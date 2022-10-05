import os
import re
import argparse
import pickle
import pandas as pd
import numpy as np
import scipy.sparse as ssp
import dgl
import torch
import torchtext
from builder import PandasGraphBuilder
from data_utils import *
import networkx as nx
from fastnode2vec import Graph, Node2Vec
import string
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel
from tqdm import tqdm

def prepr(x):
    x = x.replace("[", "")
    x = x.replace("]", "")
    x = x.replace("'", "")
    x = x.split(',')
    return x

def bert_preprocessing(df, text_col, tokenizer, max_length):
    new_df = df.copy()
    new_df[text_col] = new_df[text_col].apply(lambda x: x.lower())
    new_df[text_col] = new_df[text_col].apply(lambda x: re.sub('\d+', '', x))
    punc_table = str.maketrans('', '', string.punctuation)
    new_df[text_col] = new_df[text_col].apply(lambda x: x.translate(punc_table))
    new_df[text_col] = new_df[text_col].apply(lambda x: x.encode('ascii', 'ignore').decode())
    new_df[text_col] = new_df[text_col].apply(lambda x: " ".join(x.split()))
    
    preprocess_text = []
    for x in new_df[text_col].values:
        preprocess_text.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x))[:max_length])
        
    result_texts = []
    cls_is, pad_id, sep_id = tokenizer.cls_token_id, tokenizer.pad_token_id, tokenizer.sep_token_id
    for t in preprocess_text:
        text = [[cls_is] + t + [sep_id] + [pad_id] * (max_length - len(t))]
        result_texts.append(text)
    result_texts = np.array(result_texts).reshape(len(preprocess_text), -1)
    attention_mask = np.where(result_texts != pad_id, 1, 0)
    return {'text': torch.from_numpy(result_texts), 
            'attention_mask': attention_mask}


class Data4Bert(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data['text'])
    
    def __getitem__(self, index):        
        return {'text': self.data['text'][index],
               'attention_mask': self.data['attention_mask'][index]}

def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def bert_emb(model, loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for batch in loader:
        text, att_mask = batch['text'].to(device), batch['attention_mask'].to(device)
        with torch.no_grad():
            emb = model(input_ids=text, attention_mask=att_mask)[0]
            emb = mean_pooling(emb, att_mask)
        yield emb.detach().cpu()

def to_bert(model, loader):
    embs = [None] * len(loader)
    for i, emb in tqdm(enumerate(bert_emb(model, loader))):
        embs[i] = emb
    return torch.vstack(embs)


def get_item_text_embs(df, text_cols, id_col):
    embs = []
    for col in text_cols:
        print(f'Start {col}')
        df[col] = df[col].astype(str)
        #var_len = df[col].apply(lambda x: len(x.split(' '))).var()
        var_len = 10
        median_len = df[col].apply(lambda x: len(x.split(' '))).median()
        max_length = int(median_len + var_len)
        
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)
        texts = pd.DataFrame({'texts': list(df[col].unique())})
        n_texts = len(texts['texts'])
        print(f'Start tokenize, n_texts: {n_texts}')
        train_dict = bert_preprocessing(texts, 'texts', bert_tokenizer, max_length)
        print('End tokenize')
        
        train_data = Data4Bert(train_dict)
        train_loader = DataLoader(train_data, batch_size=128, shuffle=False)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = BertModel.from_pretrained('bert-base-uncased', output_attentions=False, output_hidden_states=False).to(device)
        
        train_embs = to_bert(model, train_loader)
        print(f'End {col}')
        embs.append({col: train_embs})
        
    return embs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory', type=str)
    parser.add_argument('meta_directory', type=str)
    parser.add_argument('output_path', type=str)
    args = parser.parse_args()
    data_directory = args.data_directory
    meta_directory = args.meta_directory
    output_path = args.output_path
    
    events = pd.read_csv(data_directory)
    meta = pd.read_csv(meta_directory)
    
    meta['also_buy'] = meta['also_buy'].apply(lambda x: prepr(x))
    meta['also_view'] = meta['also_view'].apply(lambda x: prepr(x))
    meta['category'] = meta['category'].apply(lambda x: prepr(x))
    meta['category'] = meta['category'].apply(lambda x: ' '.join(x))
    
    meta = meta.replace('', np.NaN)
    
    meta['brand'] = meta['brand'].astype('category')
    meta['brand_cat'] = meta['brand'].cat.codes.astype('int64')
    meta['brand_cat'] = meta['brand_cat'] + 1
    
    text_cols = ['category', 'title']
    id_col = 'asin'

    meta_embs = get_item_text_embs(meta, text_cols, id_col)
    
    dict_category_embs = dict(zip(meta['category'].unique(), meta_embs[0]['category']))
    dict_ids_cat = dict(zip(meta['asin'], meta['category']))
    dict_ids_embs_cat = dict()
    for k, v in dict_ids_cat.items():
        dict_ids_embs_cat[k] = dict_category_embs[v]
        
    del dict_category_embs, dict_ids_cat
    
    dict_title_embs = dict(zip(meta['title'].unique(), meta_embs[1]['title']))
    dict_ids_title = dict(zip(meta['asin'], meta['title']))
    dict_ids_embs_title = dict()
    for k, v in dict_ids_title.items():
        dict_ids_embs_title[k] = dict_title_embs[v]
        
    del dict_title_embs, dict_ids_title
    
    meta['also_buy'] = meta['also_buy'].apply(lambda x: '' if x==list() else x)
    meta['also_view'] = meta['also_view'].apply(lambda x: '' if x==list() else x)
    
    node_embs = dict()
    
    for col in ['also_buy', 'also_view']:
        edges = list()
        for asin, row in list(zip(meta['asin'], meta[col])):
            if row != '':
                for el in row:
                    edges.append([asin, el])
                    
        G = Graph(edges, directed=True, weighted=False)
        n2v = Node2Vec(G, dim=32, walk_length=100, context=10, p=2.0, q=0.5, workers=2)
        n2v.train(epochs=100)
        
        col_embs = dict()
        for asin, row in list(zip(meta['asin'], meta[col])):
            if row!='':
                col_embs[asin] = n2v.wv[asin]
        node_embs[col] = col_embs
        
    item_embs_also_buy = meta['asin'].apply(lambda x: node_embs['also_buy'][x] if x in node_embs['also_buy'].keys() else np.zeros(32))
    item_embs_also_view = meta['asin'].apply(lambda x: node_embs['also_view'][x] if x in node_embs['also_view'].keys() else np.zeros(32))
    
    item_embs_title = meta['asin'].apply(lambda x: dict_ids_embs_title[x] if x in dict_ids_embs_title.keys() else torch.zeros(768))
    item_embs_cat = meta['asin'].apply(lambda x: dict_ids_embs_cat[x] if x in dict_ids_embs_cat.keys() else torch.zeros(768))
    
    user_feats = events[['reviewerID', 'asin']].merge(meta[['asin', 'brand_cat']], on='asin')
    user_feats.drop(['asin'], axis=1, inplace=True)
    user_feats = user_feats[['reviewerID', 'brand_cat']].groupby('reviewerID').agg(pd.Series.mode).reset_index()
    user_feats['brand_cat'] = user_feats['brand_cat'].apply(lambda x: x[0] if isinstance(x, np.ndarray) else x)
    
    graph_builder = PandasGraphBuilder()
    graph_builder.add_entities(user_feats, 'reviewerID', 'customer')
    graph_builder.add_entities(meta, 'asin', 'product')
    graph_builder.add_binary_relations(events, 'reviewerID', 'asin', 'buy')
    graph_builder.add_binary_relations(events, 'asin', 'reviewerID', 'buy-by')
    g = graph_builder.build()
    
    g.nodes['product'].data['emb_also_buy'] = torch.FloatTensor(item_embs_also_buy)
    g.nodes['product'].data['emb_also_view'] = torch.FloatTensor(item_embs_also_view)
    g.nodes['product'].data['emb_title'] = torch.vstack(list(item_embs_title))
    g.nodes['product'].data['emb_cat'] = torch.vstack(list(item_embs_cat))
    g.nodes['product'].data['brand_cat'] = torch.LongTensor(meta['brand_cat'])
    g.nodes['customer'].data['fav_brand_cat'] = torch.LongTensor(user_feats['brand_cat'])
    
    g.edges['buy'].data['unixReviewTime'] = torch.LongTensor(events['unixReviewTime'].values)
    g.edges['buy-by'].data['unixReviewTime'] = torch.LongTensor(events['unixReviewTime'].values)
    
    g.edges['buy'].data['overall'] = torch.LongTensor(events['overall'].values)
    g.edges['buy-by'].data['overall'] = torch.LongTensor(events['overall'].values)
    
    n_edges = g.number_of_edges('buy')
    train_indices, val_indices, test_indices = train_test_split_by_time(events, 'unixReviewTime', 'reviewerID')
    train_g = build_train_graph(g, train_indices, 'customer', 'product', 'buy', 'buy-by')
    assert train_g.out_degrees(etype='buy').min() > 0
    val_matrix, test_matrix = build_val_test_matrix(g, val_indices, test_indices, 'customer', 'product', 'buy')
    
    dataset = {
          'train-graph': train_g,
          'val-matrix': val_matrix,
          'test-matrix': test_matrix,
          'item-texts': None,
          'item-images': None,
          'user-type': 'customer',
          'item-type': 'product',
          'user-to-item-type': 'buy',
          'item-to-user-type': 'buy-by',
          'timestamp-edge-column': 'unixReviewTime'}
    
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)