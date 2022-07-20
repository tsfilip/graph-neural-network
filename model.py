"""Graph neural network classifier implementation."""
import torch
import torch.nn as nn


# ======================================================================================================================
#                                                Graph Convolution Network
# ======================================================================================================================


class FFN(nn.Module):
    def __init__(self, out_features, n_features, dropout_rate=0.2, activation="gelu"):
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
                 dropout_rate=0.2,
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
        """Pre-process node features with FFN"""
        y = self.preprocess_ffn(node_features)

        if edge_weights is not None:
            y = y * torch.unsqueeze(edge_weights, -1)

        return y

    def aggregate(self, indices, neighbour_msg, n_nodes):
        """Aggregate representations from neighbours."""
        n_features = neighbour_msg.shape[-1]
        aggregated_msg = torch.zeros(n_nodes, n_features, dtype=neighbour_msg.dtype).to(neighbour_msg.device)

        # Aggregate representation of each node neighbour.
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
        """Combine node representation together with aggregated neighbours messages."""
        if self.combination_type == "concat":
            embedding = torch.cat((node_representation, aggregated_msg), -1)
        elif self.combination_type == "sum":
            embedding = node_representation + aggregated_msg
        else:
            raise ValueError("Invalid argument: Combination type must be concat or sum")

        # post-process combined representation with FFN
        embedding = self.update_ffn(embedding)

        # Apply L2 normalization
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


class GCNClassifier(nn.Module):
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

        super(GCNClassifier, self).__init__(*args, **kwargs)

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

        # Post-process node representations
        x = self.postprocess(x)

        # Select nodes from mini-batch.
        embedding = x[input_node_indices]
        # Process embedding with linear layer
        embedding = self.logits(embedding)
        return embedding


# ======================================================================================================================
#                                                Graph Attention Network
# ======================================================================================================================

class GraphAttentionLayer(nn.Module):
    """Graph attention layer for processing graph structured data.
    Args:
        n_heads: number of attention heads,
        in_features: input feature shape,
        out_features: number of features for each attention head
    """
    def __init__(self, in_features, out_features, n_heads, concat=True):
        super(GraphAttentionLayer, self).__init__()

        if concat:
            assert out_features % n_heads == 0
            out_features = out_features // n_heads

        self.concat = concat
        self.n_heads = n_heads
        self.preprocess = nn.Linear(in_features, n_heads * out_features, bias=False)
        self.attention_layer = nn.Linear(2 * out_features, 1, bias=False)
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, node_representations, adjacency_matrix):
        n_nodes = node_representations.shape[0]

        x = self.preprocess(node_representations)
        x = x.view(n_nodes, self.n_heads, -1)  # shape (n_nodes, n_heads, out_features)

        x_source = x.repeat(n_nodes, 1, 1)  # shape (n_nodes * n_nodes, n_heads, out_features)
        x_target = x.repeat_interleave(n_nodes, 0)

        x_cat = torch.cat([x_source, x_target], -1)
        x_cat = x_cat.view(n_nodes, n_nodes, self.n_heads, -1)  # shape (n_nodes, n_nodes, n_heads, 2 * out_features)
        x_cat = self.activation(self.attention_layer(x_cat))  # shape (n_nodes, n_nodes, n_heads, 1)
        x_cat = x_cat.squeeze(-1)

        x_cat = x_cat.masked_fill(~adjacency_matrix, float('-inf'))  # shape (n_nodes, n_nodes, n_heads)

        a = self.softmax(x_cat)
        h = torch.einsum("ijh,jhk->ihk", a, x)  # shape(n_nodes, n_head, out_features)

        if self.concat:
            h = h.reshape(n_nodes, -1)  # shape (n_nodes, out_features * n_heads)
        else:
            h = torch.mean(h, axis=1)  # shape (n_Nodes, out_features)

        return h


class GANClassfier(nn.Module):
    def __init__(self,
                 graph_info,
                 n_class,
                 out_features,
                 dropout_rate=0.2,
                 concat=True,
                 n_heads=4,
                 *args,
                 **kwargs):
        super(GANClassfier, self).__init__(*args, **kwargs)

        node_features, edges, _ = graph_info
        n_features = node_features.shape[-1]
        in_shape = out_features[-1]

        # Create adjacency matrix of shape (n_nodes, n_nodes)
        node_indices, neighbour_indices = edges[0], edges[1]
        adjacency_matrix = torch.eye(node_features.shape[0], dtype=torch.bool).to(node_features.device)
        adjacency_matrix[node_indices, neighbour_indices] = True
        adjacency_matrix[neighbour_indices, node_indices] = True
        adjacency_matrix = torch.unsqueeze(adjacency_matrix, -1)
        self.adjacency_matrix = adjacency_matrix
        self.node_features = node_features.float()

        self.preprocess = FFN(out_features, n_features, dropout_rate)
        self.gan1 = GraphAttentionLayer(in_shape, in_shape, n_heads, concat)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm([in_shape])
        self.postprocess = FFN(out_features, in_shape, dropout_rate)
        self.logits = nn.Linear(in_shape, n_class)

    def forward(self, input_node_indices):
        # Preprocess features of all nodes
        x = self.preprocess(self.node_features)

        y = self.gan1(x, self.adjacency_matrix)
        # Skip connection
        x = x + self.dropout(y)
        x = self.norm(x)

        # Post-process node representations
        x = self.postprocess(x)

        # Select nodes from mini-batch.
        embedding = x[input_node_indices]
        # Process embedding with linear layer
        embedding = self.logits(embedding)
        return embedding
