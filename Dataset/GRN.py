'''scGPT'''

import copy
import json
import os
import sys
import warnings

import torch
import numpy as np
import pandas as pd

# sys.path.insert(0, "../")
from scgpt.tasks import GeneEmbedding
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.model import TransformerModel
from scgpt.preprocess import Preprocessor
from scgpt.utils import set_seed
from joblib import Parallel, delayed


def select_cells(adata, condition='ctrl', num_cells=200):
    ctrl_cells = adata[adata.obs['condition'] == condition]

    if ctrl_cells.shape[0] >= num_cells:
        selected_cells = ctrl_cells[np.random.choice(ctrl_cells.shape[0], num_cells, replace=False)]
    else:
        raise ValueError(f"Control group does not have enough cells. Available: {ctrl_cells.shape[0]}")

    return selected_cells


def co_expression_GRN(adata,data_name, threshold=0.45, num_cells=500):
    adata = select_cells(adata, condition='ctrl', num_cells=num_cells)
    if data_name=="sciplex3":
        gene_names = adata.var.index.tolist()
    else:
        gene_names = adata.var['gene_name'].values
    expression_matrix = pd.DataFrame(adata.X.toarray(), columns=gene_names)
    co_expression_matrix = expression_matrix.corr(method='pearson')
    co_expression_matrix = co_expression_matrix.values
    co_expression_edges = [
        (gene_names[i], gene_names[j],
         1.0 if i == j and np.isnan(co_expression_matrix[i, j]) else co_expression_matrix[i, j])
        for i in range(len(gene_names))
        for j in range(len(gene_names))
        # if gene1 != gene2 and abs(co_expression_matrix.loc[gene1, gene2]) > threshold
        if abs(co_expression_matrix[i, j]) > threshold or i==j
    ]
    return co_expression_edges

def get_GRN(adata,data_name,threshold=0.25,path="",threshold_co=0.45):
    print("Constructing GRN...")
    os.environ["KMP_WARNINGS"] = "off"
    warnings.filterwarnings('ignore')

    set_seed(42)
    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]
    n_hvg = 1200
    n_bins = 51
    mask_value = -1
    pad_value = -2
    n_input_bins = n_bins

    # Specify model path; here we load the pre-trained scGPT blood model
    model_config_file = path+"/scGPT/args.json"
    model_file = path+"/scGPT/best_model.pt"
    vocab_file = path+"/scGPT/vocab.json"

    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    # Retrieve model parameters from config files
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    print(
        f"Resume model from {model_file}, the model args will override the "
        f"config {model_config_file}."
    )
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]

    gene2idx = vocab.get_stoi()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ntokens = len(vocab)  # size of vocabulary
    model = TransformerModel(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        vocab=vocab,
        pad_value=pad_value,
        n_input_bins=n_input_bins,
    )

    try:
        model.load_state_dict(torch.load(model_file))
        print(f"Loading all model params from {model_file}")
    except:
        # only load params that are in the model and match the size
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file)
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        for k, v in pretrained_dict.items():
            print(f"Loading params {k} with shape {v.shape}")
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

    model.to(device)


    # Retrieve the data-independent gene embeddings from scGPT
    gene_ids = np.array([id for id in gene2idx.values()])
    gene_embeddings = model.encoder(torch.tensor(gene_ids, dtype=torch.long).to(device))
    gene_embeddings = gene_embeddings.detach().cpu().numpy()

    # Filter on the intersection between the Immune Human HVGs found in step 1.2 and scGPT's 30+K foundation model vocab
    if data_name=="sciplex3":
        gene_embeddings = {gene: gene_embeddings[i] for i, gene in enumerate(gene2idx.keys()) if
                           gene in adata.var.index.tolist()}
    else:
        gene_embeddings = {gene: gene_embeddings[i] for i, gene in enumerate(gene2idx.keys()) if
                       gene in adata.var['gene_name'].tolist()}
    print('Retrieved gene embeddings for {} genes.'.format(len(gene_embeddings)))

    # Construct gene embedding network
    embed = GeneEmbedding(gene_embeddings)
    GRN = embed.generate_network(threshold=threshold)

    co_expr_GRN = co_expression_GRN(adata,data_name=data_name,threshold=threshold_co)
    GRN.add_weighted_edges_from(co_expr_GRN)
    print("Completed GRN.")
    return GRN