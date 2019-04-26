import numpy as np
import scipy.sparse as sp
import torch
import json
import os
import networkx as nx


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    if dataset == "cora":
        idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                            dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        labels = encode_onehot(idx_features_labels[:, -1])

        # build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                        dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        features = normalize(features)
        adj = normalize(adj + sp.eye(adj.shape[0]))

        idx_train = range(140)
        idx_val = range(200, 500)
        idx_test = range(500, 1500)

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(labels)[1])
        adj = sparse_mx_to_torch_sparse_tensor(adj)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        return adj, features, labels, idx_train, idx_val, idx_test
    elif dataset == "adni":
        """
        Temporal_mapping.json and AD_network and AD_parc has to be in the path directory
        """
        sub_info = json.load(open(os.path.join(path, "temporal_mapping.json")))
        base_net_ids = []
        labels = {}
        for key, data in sub_info.items():
            base_net_ids.append(data[0]["network_id"])
            labels[data[0]["network_id"]] = data[0]["dx_data"]

        network = {}
        excluded_id = []
        for nid in base_net_ids:
            filepath = path + "/AD_network/" + nid + "_fdt_network_matrix"

            try:
                network[nid] = np.genfromtxt(filepath, dtype=np.float32)
            except OSError:
                print("Skipping {}".format(nid))
                excluded_id.append(nid)
                del labels[nid]

        feature = {}
        for nid in base_net_ids:
            if nid not in excluded_id:
                filepath = path + "/AD_parc/" + nid + "_parcellationTable.json"
                try:
                    with open(filepath) as f:
                        table = json.load(f)
                    feat = np.zeros((len(table), 2))
                    for i, node in enumerate(table):
                        feat[i][0] = node["ThickAvg"]
                        feat[i][1] = node["SurfArea"]

                    feature[nid] = feat
                except FileNotFoundError:
                    print("Skipping parc {}".format(nid))
                    del network[nid]
                    del labels[nid]

        return process_adni_data(feature, network, labels)


def process_adni_data(feature, network, labels):
    F = []
    A = []
    L = []

    for l in labels.keys():
        F.append(feature[l])
        A.append(network[l])
        L.append(labels[l])

    return F, A, L


def generate_grid_graph(n_nodes):
    G = nx.grid_2d_graph(n_nodes, n_nodes)
    return nx.to_numpy_matrix(G)


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


if __name__ == '__main__':
    a, b, c = load_data('/home/turja', 'adni')
    G = nx.grid_2d_graph(4, 4)


