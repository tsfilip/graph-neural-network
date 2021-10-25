import torch
import torch.nn as nn


class FFN(nn.Module):
    def __init__(self, out_features, n_features, dropout_rate, activation="gelu"):
        super(FFN, self).__init__()
        if activation == "gelu":
            activation = nn.GELU()
        elif activation == "relu":
            activation = nn.ReLU()
        else:
            raise ValueError("Invalid argument: Activation function name must be relu or gelu.")

        layers = nn.ModuleList()
        in_features = [n_features] + out_features[:-1]

        for in_shape, out_shape in zip(in_features, out_features):
            layers.append(nn.BatchNorm1d(in_shape))
            layers.append(nn.Dropout(p=dropout_rate))
            layers.append(nn.Linear(in_shape, out_shape))
            layers.append(activation)
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class GraphConvLayer(nn.Module):
    def __init__(self, ffn_units,
                 n_features,
                 dropout_rate,
                 normalize=False,
                 aggregation_type="mean",
                 combination_type="concat"):
        super(GraphConvLayer, self).__init__()

        self.preprocess_ffn = FFN(ffn_units, n_features, dropout_rate)
        n_out_features = 2 * n_features if combination_type == "concat" else n_features
        self.update_ffn = FFN(ffn_units, n_out_features, dropout_rate)
        self.aggregation_type = aggregation_type
        self.combination_type = combination_type
        self.normalize = normalize

    def preprocess_nodes(self, node_features, edge_weights=None):
        y = self.preprocess_ffn(node_features)

        if edge_weights is not None:
            y = y * torch.unsqueeze(edge_weights, -1)

        return y

    def aggregate(self, indices, neighbour_msg, n_nodes):
        n_features = neighbour_msg.shape[-1]
        aggregated_msg = torch.zeros(n_nodes, n_features, dtype=neighbour_msg.dtype).to(neighbour_msg.device)

        # for each node aggregate features of his neighbours
        if self.aggregation_type == "sum":
            indices = torch.unsqueeze(indices, -1)
            indices = indices.expand([-1, n_features])  # reshape to [n_edges, n_features]
            aggregated_msg = aggregated_msg.scatter_add(0, indices, neighbour_msg)
        elif self.aggregation_type == "mean":
            indices = torch.unsqueeze(indices, -1)
            indices = indices.expand([-1, n_features])
            aggregated_msg = aggregated_msg.scatter_add(0, indices, neighbour_msg)

            unique, count = torch.unique(indices, return_counts=True)
            counter = torch.zeros(n_nodes, dtype=aggregated_msg.dtype).to(aggregated_msg.device)
            counter[unique] = count.type(counter.dtype)  # number of neighbors for each node
            counter = torch.unsqueeze(counter, -1)  # reshape to [n_nodes, 1]

            aggregated_msg = aggregated_msg / counter
            aggregated_msg[aggregated_msg != aggregated_msg] = 0  # Replace nan with zero because division by zero
        else:
            raise ValueError("Invalid argument: aggregation type must be mean or sum")

        return aggregated_msg

    def update(self, node_representation, aggregated_msg):
        if self.combination_type == "concat":
            embedding = torch.cat((node_representation, aggregated_msg), -1)
        elif self.combination_type == "sum":
            embedding = node_representation + aggregated_msg
        else:
            raise ValueError("Invalid argument: Combination type must be concat or sum")

        embedding = self.update_ffn(embedding)

        if self.normalize:
            embedding = nn.functional.normalize(embedding, dim=-1, p=2)

        return embedding

    def forward(self, node_representations, edges, edge_weights):
        node_indices, neighbour_indices = edges[0], edges[1]

        neighbour_features = node_representations[neighbour_indices]
        neighbour_features = self.preprocess_nodes(neighbour_features, edge_weights)
        aggregated_msg = self.aggregate(node_indices, neighbour_features, node_representations.shape[0])

        embedding = self.update(node_representations, aggregated_msg)
        return embedding


class GNNClassifier(nn.Module):
    def __init__(self,
                 graph_info,
                 n_class,
                 out_features,
                 aggregation_type="sum",
                 combination_type="concat",
                 dropout_rate=0.2,
                 normalization=True,
                 *args,
                 **kwargs):

        super(GNNClassifier, self).__init__(*args, **kwargs)

        node_features, edges, edge_weights = graph_info
        n_features = node_features.shape[-1]
        in_shape = out_features[-1]

        self.node_features = node_features.float()
        self.edges = edges
        if edge_weights is None:
            edge_weights = torch.ones(edges.shape[1])
        else:
            edge_weights = edge_weights

        # Rescale weights so that their sum is equal to 1
        edge_weights = edge_weights / torch.sum(edge_weights)

        self.edge_weights = edge_weights

        self.preprocess = FFN(out_features, n_features, dropout_rate)
        self.gcn1 = GraphConvLayer(out_features,
                                   in_shape,
                                   dropout_rate,
                                   normalization,
                                   aggregation_type,
                                   combination_type)
        self.gcn2 = GraphConvLayer(out_features,
                                   in_shape,
                                   dropout_rate,
                                   normalization,
                                   aggregation_type,
                                   combination_type)
        self.postprocess = FFN(out_features, in_shape, dropout_rate)
        self.logits = nn.Linear(in_shape, n_class)

    def forward(self, input_node_indices):
        # Preprocess features of all nodes
        x = self.preprocess(self.node_features)

        y = self.gcn1(x, self.edges, self.edge_weights)
        # Skip connection
        x = x + y

        y = self.gcn2(x, self.edges, self.edge_weights)
        # Skip connection
        x = x + y

        x = self.postprocess(x)

        embedding = x[input_node_indices]
        # Process embedding with linear layer
        embedding = self.logits(embedding)
        return embedding



