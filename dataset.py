import os
import re
import numpy as np
import pandas as pd


def read_dataset(file_path):
    """Method for reading and processing CORA dataset.
    Args:
        file_path: path to dataset location. Location contain cites, content and paper csv files.
    """
    cites = pd.read_csv(os.path.join(file_path, "cites.csv"))
    content = pd.read_csv(os.path.join(file_path, "content.csv"))
    papers = pd.read_csv(os.path.join(file_path, "paper.csv"))
    # Find unique node class names
    class_names = papers["class_label"].unique()

    # Remove unnecessary characters
    content["word_cited_id"] = content["word_cited_id"].apply(lambda x: re.sub(r"\D", "", x))
    content["word_cited_id"] = pd.factorize(content["word_cited_id"].astype("int32"), sort=True)[0]

    # Replace string class names with class index
    papers["class_label"] = papers["class_label"].apply(lambda x: np.where(class_names == x)[0][0])

    # Re-factorize paper ids
    paper_key = dict(zip(papers["paper_id"].tolist(), range(len(papers["paper_id"]))))
    papers["paper_id"] = papers["paper_id"].apply(lambda x: paper_key[x])
    content["paper_id"] = content["paper_id"].apply(lambda x: paper_key[x])
    cites["cited_paper_id"] = cites["cited_paper_id"].apply(lambda x: paper_key[x])
    cites["citing_paper_id"] = cites["citing_paper_id"].apply(lambda x: paper_key[x])

    n_word = content["word_cited_id"].max() + 1  # number of different words in dictionary
    n_papers = papers.shape[0]
    n_edges = cites.shape[0]

    # Node representations with ones for words contained in paper.
    node_features = np.zeros([n_papers, n_word])
    node_features[content["paper_id"], content["word_cited_id"]] = 1
    del content

    edge_weights = np.ones(n_edges, dtype=np.float32)
    features_col = [f"word_{i}" for i in range(n_word)]
    node_features = pd.DataFrame(node_features, columns=features_col)
    papers = pd.concat([papers, node_features], axis=1)

    # Split dataset to train and test sets
    train_samples = []
    test_samples = []
    for _, group in papers.groupby("class_label"):
        mask = np.random.uniform(size=group.shape[0]) <= 0.5
        train_samples.append(group[mask])
        test_samples.append(group[~mask])

    train_samples = pd.concat(train_samples, axis=0)
    test_samples = pd.concat(test_samples, axis=0)

    x_train = train_samples["paper_id"].to_numpy()
    x_test = test_samples["paper_id"].to_numpy()
    y_train = train_samples["class_label"].to_numpy()
    y_test = test_samples["class_label"].to_numpy()

    return (x_train, y_train), (x_test, y_test), \
           (node_features.to_numpy(), cites.to_numpy().T, edge_weights), \
            class_names