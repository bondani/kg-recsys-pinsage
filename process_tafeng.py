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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str)
    parser.add_argument('output_path', type=str)
    args = parser.parse_args()
    directory = args.directory
    output_path = args.output_path

    ## Build heterogeneous graph

    # Load data
    df = pd.read_csv(directory, delimiter=',')

    df['TRANSACTION_DT'] = pd.to_datetime(df['TRANSACTION_DT'], format='%m/%d/%Y').astype('category')
    df['TRANSACTION_DT_cat'] = df['TRANSACTION_DT'].cat.codes
    df['AGE_GROUP'] = df['AGE_GROUP'].astype('category')
    df['AGE_GROUP_cat'] = df['AGE_GROUP'].cat.codes + 1
    df['PIN_CODE'] = df['PIN_CODE'].astype('category')
    df['PIN_CODE_cat'] = df['PIN_CODE'].cat.codes
    df['PRODUCT_SUBCLASS'] = df['PRODUCT_SUBCLASS'].astype('category')
    df['PRODUCT_SUBCLASS_cat'] = df['PRODUCT_SUBCLASS'].cat.codes
    df['PRICE'] = df['SALES_PRICE']/df['AMOUNT']

    # Item meta
    prod_class = dict(zip(df['PRODUCT_ID'], df['PRODUCT_SUBCLASS']))
    price = df[['PRODUCT_ID', 'PRICE']].groupby('PRODUCT_ID').mean().reset_index()
    items = pd.DataFrame({'PRODUCT_ID': prod_class.keys(), 'PRODUCT_SUBCLASS_cat': prod_class.values()})
    items = items.merge(price, how='left', on='PRODUCT_ID')

    # User meta
    users = df[['CUSTOMER_ID', 'AGE_GROUP_cat', 'PIN_CODE_cat']].drop_duplicates()

    # Interactions
    events = df[['CUSTOMER_ID', 'PRODUCT_ID', 'TRANSACTION_DT_cat']].drop_duplicates()

    # Build graph
    graph_builder = PandasGraphBuilder()
    graph_builder.add_entities(users, 'CUSTOMER_ID', 'customer')
    graph_builder.add_entities(items, 'PRODUCT_ID', 'product')
    graph_builder.add_binary_relations(events, 'CUSTOMER_ID', 'PRODUCT_ID', 'buy')
    graph_builder.add_binary_relations(events, 'PRODUCT_ID', 'CUSTOMER_ID', 'buy-by')
    g = graph_builder.build()

    # Assign features.
    g.nodes['customer'].data['AGE_GROUP'] = torch.LongTensor(users['AGE_GROUP_cat'].values)
    g.nodes['customer'].data['PIN_CODE'] = torch.LongTensor(users['PIN_CODE_cat'].values)
    g.nodes['product'].data['PRODUCT_SUBCLASS'] = torch.LongTensor(items['PRODUCT_SUBCLASS_cat'].values)
    g.nodes['product'].data['PRICE'] = torch.LongTensor(items['PRICE'].values)

    g.edges['buy'].data['TRANSACTION_DT'] = torch.LongTensor(events['TRANSACTION_DT_cat'].values)
    g.edges['buy-by'].data['TRANSACTION_DT'] = torch.LongTensor(events['TRANSACTION_DT_cat'].values)

    n_edges = g.number_of_edges('buy')

    # Train-validation-test split
    train_indices, val_indices, test_indices = train_test_split_by_time(events, 'TRANSACTION_DT_cat', 'CUSTOMER_ID')

    # Build the graph with training interactions only.
    train_g = build_train_graph(g, train_indices, 'customer', 'product', 'buy', 'buy-by')
    assert train_g.out_degrees(etype='buy').min() > 0

    # Build the user-item sparse matrix for validation and test set.
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
        'timestamp-edge-column': 'TRANSACTION_DT'}

    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
