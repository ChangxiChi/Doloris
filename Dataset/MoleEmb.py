import numpy as np
import torch
from unimol_tools import UniMolRepr
# smiles unimol representations

def extract_mole_embed(smiles_list=['c1ccc(cc1)C2=NCC(=O)Nc3c2cc(cc3)[N+](=O)[O]']):
    print("Extracting molecule embedding...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    clf = UniMolRepr(data_type='molecule', remove_hs=False)
    unimol_repr = clf.get_repr(smiles_list, return_atomic_reprs=True)
    # CLS token repr
    print(np.array(unimol_repr['cls_repr']).shape)
    print("Completed !")
    return torch.tensor(np.array(unimol_repr['cls_repr'])).to(device)


from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from tqdm import tqdm
import pandas as pd


def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    return None


def compute_morgan_count_embedding(smiles, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetHashedMorganFingerprint(mol, radius=2, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def get_rdkit_embeddings_tensor(
    smiles_list=['c1ccc(cc1)C2=NCC(=O)Nc3c2cc(cc3)[N+](=O)[O]','CC(=O)Oc1ccccc1C(=O)O', 'COC(=O)CCCCCC(=O)O'],
    n_bits=2048,
    threshold=0.01,
    skip_variance_filter=False,
    return_smiles=False
):
    canon_smiles_set = set()
    canon_smiles = []
    embeddings = []

    for smile in tqdm(smiles_list, desc="Processing SMILES"):
        canon = canonicalize_smiles(smile)
        if canon is None or canon in canon_smiles_set:
            continue
        emb = compute_morgan_count_embedding(canon, n_bits)
        if emb is not None:
            canon_smiles.append(canon)
            canon_smiles_set.add(canon)
            embeddings.append(emb)

    df = pd.DataFrame(embeddings, index=canon_smiles, columns=[f"latent_{i}" for i in range(n_bits)])
    df.drop(columns=["latent_0"], inplace=True)

    if not skip_variance_filter:
        low_std_cols = df.columns[df.std() <= threshold].tolist()
        df.drop(columns=low_std_cols, inplace=True)

    df = (df - df.mean()) / df.std()

    if return_smiles:
        return df.values, canon_smiles
    return df.values

if __name__ == '__main__':
    get_rdkit_embeddings_tensor()