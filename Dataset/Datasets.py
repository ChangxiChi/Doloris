from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import scipy.sparse as sp

class AnnDataBatchDataset(torch.utils.data.Dataset):
    def __init__(self, adata, batch_size=32):
        self.adata = adata
        self.batch_size = batch_size
        self.num_cells = adata.shape[0]

        self.cell_expression = torch.tensor(adata.X.toarray()).to('cuda')
        self.condition = adata.obs['condition']

    def __len__(self):
        return self.num_cells

    def __getitem__(self, idx):
        cell_expression = self.cell_expression[idx, :]
        cell_condition = self.condition.iloc[idx]  # obtain cell's condition

        return cell_expression, cell_condition



class SourceModelDataset(Dataset):
    def __init__(self, adata, cell_list):
        self.cell_expression_all = torch.tensor(adata.X.toarray()).to('cuda')
        self.cell_list = cell_list
        self.cell_type_all = adata.obs['cell_type'].values.tolist()

        self.ctrl_base_list={}
        for cell_type in self.cell_list:
            current_type_adata=adata[adata.obs['cell_type'] == cell_type]
            current_type_adata=current_type_adata[current_type_adata.obs['condition']=='ctrl']
            mean_gene_expr=torch.mean(torch.tensor(current_type_adata.X.toarray()),dim=0).to('cuda')
            self.ctrl_base_list[cell_type]=mean_gene_expr

    def __len__(self):
        return self.cell_expression_all.shape[0]

    def __getitem__(self, idx):
        cell_expression = self.cell_expression_all[idx, :]
        if isinstance(idx, list):
            cell_type = [self.cell_type_all[i] for i in idx]
        else:
            cell_type = [self.cell_type_all[idx]]

        indices=[]
        ctrl_base=[]
        for ctype in cell_type:
            for i,label in enumerate(self.cell_list):
                if ctype==label:
                    indices.append(i)
            ctrl_base.append(self.ctrl_base_list[ctype])
        # indices = [ [i for i, label in enumerate(self.cell_list) if label == ctype] for ctype in cell_type ]
        ctrl_base=torch.stack(ctrl_base).to('cuda').squeeze()
        indices_tensor = torch.tensor(indices).to('cuda').squeeze()
        return {
            'feature':cell_expression,
            'cell_type':indices_tensor,
            # 'ctrl_base':ctrl_base,
        }

# obtain
class TargetModelDataset_Molecular(Dataset):
    def __init__(self,adata,adata_ctrl,cell_list,mole_embed,mole_list):
        self.adata=adata
        self.adata.obs.loc[self.adata.obs['condition'] == 'ctrl', 'dose_val'] = 0

        self.cell_expression_all = torch.tensor(adata.X.toarray()).to('cuda')
        self.cell_list = cell_list
        self.cell_type_all = adata.obs['cell_type'].values.tolist()
        self.SMILES_all = adata.obs['SMILES'].values.tolist()
        self.mole_embed=torch.tensor(mole_embed).to('cuda')
        self.dosage=torch.tensor(adata.obs['dose_val'].values.tolist())
        self.mole_list=mole_list # store a list of molecular

        self.ctrl_mean_list={}
        self.ctrl_var_list={}
        for cell_type in self.cell_list:
            current_type_adata=adata_ctrl[adata_ctrl.obs['cell_type'] == cell_type]
            current_type_adata=current_type_adata[current_type_adata.obs['condition']=='ctrl']
            mean_gene_expr=torch.mean(torch.tensor(current_type_adata.X.toarray()),dim=0).to('cuda')
            var_gene_expr= torch.var(torch.tensor(current_type_adata.X.toarray()),dim=0).to('cuda')
            self.ctrl_mean_list[cell_type]=mean_gene_expr
            self.ctrl_var_list[cell_type]=var_gene_expr


    def __len__(self):
        return self.adata.shape[0]

    # def __getitem__(self, idx):
    #     cell_expression = self.cell_expression_all[idx, :]
    #     if isinstance(idx, list):
    #         cell_type = [self.cell_type_all[i] for i in idx]
    #         molecular = [self.SMILES_all[i] for i in idx]
    #         dosage = self.dosage[idx]
    #     else:
    #         cell_type = [self.cell_type_all[idx]]
    #         molecular = [self.SMILES_all[idx]]
    #         dosage = self.dosage[idx].unsqueeze(0)
    #
    #     molecular = [s.split("|")[0] for s in molecular if s != '']
    #
    #     indices=[]
    #     ctrl_base=[]
    #     for ctype in cell_type:
    #         for i,label in enumerate(self.cell_list):
    #             if ctype==label:
    #                 indices.append(i)
    #         ctrl_base.append(self.ctrl_base_list[ctype])
    #
    #     indices_tensor = torch.tensor(indices).to('cuda')
    #
    #     indices_mole=[]
    #
    #     for mole in molecular:
    #         for i,smile in enumerate(self.mole_list):
    #             if smile==mole:
    #                 indices_mole.append(i)
    #
    #     mole_embed=self.mole_embed[indices_mole]
    #     ctrl_base=torch.stack(ctrl_base).to('cuda').squeeze()
    #     return {
    #         'cell_type':indices_tensor.to(torch.long) ,
    #         'feature':cell_expression,
    #         'mole':mole_embed.squeeze(),
    #         'dosage':dosage,
    #         'ctrl_base':ctrl_base,
    #     }

    def __getitem__(self, idx):
        cell_expression = self.cell_expression_all[idx, :]

        if isinstance(idx, list):
            cell_type = [self.cell_type_all[i] for i in idx]
            molecular = [self.SMILES_all[i] for i in idx]
            dosage = self.dosage[idx]
        else:
            cell_type = [self.cell_type_all[idx]]
            molecular = [self.SMILES_all[idx]]
            dosage = self.dosage[idx].unsqueeze(0)

        # 处理 molecular：控制组是空字符串或 None
        molecular_cleaned = []
        indices_mole = []
        for s in molecular:
            if s == 'CS(C)=O':
                molecular_cleaned.append(None)
                indices_mole.append(-1)
            else:
                mol = s.split("|")[0]
                molecular_cleaned.append(mol)
                for i, smile in enumerate(self.mole_list):
                    if mol == smile:
                        indices_mole.append(i)

        indices = []
        ctrl_mean = []
        ctrl_var = []
        for ctype in cell_type:
            for i, label in enumerate(self.cell_list):
                if ctype == label:
                    indices.append(i)
            ctrl_mean.append(self.ctrl_mean_list[ctype])
            ctrl_var.append(self.ctrl_var_list[ctype])

        indices_tensor = torch.tensor(indices).to('cuda')
        ctrl_mean = torch.stack(ctrl_mean).to('cuda').squeeze()
        ctrl_var = torch.stack(ctrl_var).to('cuda').squeeze()

        mole_embed = []
        for i in indices_mole:
            if i == -1:
                mole_embed.append(torch.zeros_like(self.mole_embed[0]))  # 全零嵌入
            else:
                mole_embed.append(self.mole_embed[i])
        mole_embed = torch.stack(mole_embed).to('cuda')


        return {
            'cell_type': indices_tensor.to(torch.long),
            'feature': cell_expression,
            'mole': mole_embed.squeeze(),
            'dosage': dosage,
            'ctrl_mean': ctrl_mean,
            'ctrl_var': ctrl_var,
            'x0':cell_expression,
        }


class TargetModelDataset_Gene(torch.utils.data.Dataset):
    def __init__(self, adata,adata_ctrl,cell_list,gene_list):
        self.gene_list=gene_list
        self.adata=adata
        if sp.issparse(adata.X):
            self.cell_expression_all = torch.tensor(adata.X.toarray()).to('cuda')
        else:
            self.cell_expression_all = torch.tensor(adata.X).to('cuda')

        self.cell_list = cell_list
        self.cell_type_all = adata.obs['cell_type'].values.tolist()
        self.knockout_all = adata.obs['condition'].values.tolist()

        self.ctrl_mean_list={}
        self.ctrl_var_list={}
        for cell_type in self.cell_list:
            current_type_adata=adata_ctrl[adata_ctrl.obs['cell_type'] == cell_type]
            current_type_adata=current_type_adata[current_type_adata.obs['condition']=='ctrl']
            mean_gene_expr=torch.mean(torch.tensor(current_type_adata.X.toarray()),dim=0).to('cuda')
            var_gene_expr= torch.var(torch.tensor(current_type_adata.X.toarray()),dim=0).to('cuda')
            self.ctrl_mean_list[cell_type]=mean_gene_expr
            self.ctrl_var_list[cell_type]=var_gene_expr



    def __len__(self):
        return self.adata.shape[0]

    def __getitem__(self, idx):
        cell_expression = self.cell_expression_all[idx, :]
        if isinstance(idx, list):
            cell_type = [self.cell_type_all[i] for i in idx]
            knockout = [self.knockout_all[i] for i in idx]

        else:
            cell_type = [self.cell_type_all[idx]]
            knockout = [self.knockout_all[idx]]

        indices = []
        ctrl_mean=[]
        ctrl_var=[]
        for ctype in cell_type:
            for i, label in enumerate(self.cell_list):
                if ctype == label:
                    indices.append(i)
            ctrl_mean.append(self.ctrl_mean_list[ctype])
            ctrl_var.append(self.ctrl_var_list[ctype])

        indices_tensor = torch.tensor(indices).to('cuda')

        knockout_indices=[]
        for k in knockout:
            if k.split("+")[0]!='ctrl':
                knock_gene=k.split("+")[0]
            else:
                if len(k.split("+"))!=1:
                    knock_gene = k.split("+")[1]
                else:
                    knock_gene = 'ctrl'

            if knock_gene=='ctrl':
                knockout_indices.append(-1)
            else:
                knockout_indices.append(self.gene_list.index(knock_gene))

        knockout_indices = torch.tensor(knockout_indices).to('cuda')
        ctrl_mean=torch.stack(ctrl_mean).to('cuda').squeeze()
        ctrl_var=torch.stack(ctrl_var).to('cuda').squeeze()

        return {
            'cell_type': indices_tensor,
            'feature': cell_expression,
            'knockout': knockout_indices,
            'ctrl_mean': ctrl_mean,
            'ctrl_var': ctrl_var,
            'x0': cell_expression,
        }

def return_dataloader(adata,cell_type,adata_ctrl=None,mole_embed=None,mole_list=None,gene_name=None,source_model=True,pert_type="molecular",batch_size=32):
    if source_model:
        return DataLoader(SourceModelDataset(adata=adata,cell_list=cell_type),batch_size=batch_size,shuffle=True)
    else:
        if pert_type=="molecular":
            return DataLoader(TargetModelDataset_Molecular(adata,adata_ctrl=adata_ctrl,cell_list=cell_type,mole_embed=mole_embed,mole_list=mole_list),batch_size=batch_size,shuffle=True)
        else:
            return DataLoader(TargetModelDataset_Gene(adata,adata_ctrl=adata_ctrl,cell_list=cell_type,gene_list=gene_name),batch_size=batch_size,shuffle=True)


import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

class MaskModelDataset_Gene(Dataset):
    def __init__(self, adata, adata_ctrl, cell_list, gene_list,data_name="adamson"):
        """
        Dataset now returns the Marginal Probability (Soft Label) per condition.
        Length = Number of unique (Cell Type, Condition) pairs.
        """
        self.gene_list = gene_list
        self.cell_list = cell_list
        
        self.ctrl_stats = {} 
        print("Pre-computing Control Statistics...")
        
        for cell_type in cell_list:
            subset = adata_ctrl[(adata_ctrl.obs['cell_type'] == cell_type) & 
                                (adata_ctrl.obs['condition'] == 'ctrl')]
            
            if subset.shape[0] > 0:
                if sp.issparse(subset.X):
                    data = subset.X.toarray()
                else:
                    data = subset.X
                
                mean_expr = np.mean(data, axis=0)
                var_expr = np.var(data, axis=0)
            else:
                mean_expr = np.zeros(len(gene_list))
                var_expr = np.zeros(len(gene_list))

            self.ctrl_stats[cell_type] = {
                'mean': torch.tensor(mean_expr, dtype=torch.float32),
                'var': torch.tensor(var_expr, dtype=torch.float32)
            }

        print("Aggregating Conditions and Calculating Marginal Probabilities...")
        self.data_samples = []
        
        if data_name=="norman":
            grouped = adata.obs.groupby(['cell_type', 'knockout'])
        elif data_name=="adamson": 
            grouped = adata.obs.groupby(['cell_type', 'condition'])

        for (ctype, cond), group_indices in tqdm(grouped):
            if ctype not in self.cell_list:
                continue

            if sp.issparse(adata.X):
                X_subset = adata.X[adata.obs.index.get_indexer(group_indices.index)]
                X_subset = X_subset.toarray()
            else:
                X_subset = adata.X[adata.obs.index.get_indexer(group_indices.index)]

            marginal_prob = np.mean((X_subset > 0).astype(np.float32), axis=0)

            ko_idx = self._parse_knockout(cond)
            ctype_idx = self.cell_list.index(ctype)

            self.data_samples.append({
                'cell_type_idx': ctype_idx,
                'cell_type_name': ctype,
                'knockout_idx': ko_idx,
                'condition_name': cond,
                'marginal_prob': torch.tensor(marginal_prob, dtype=torch.float32)
            })

    def _parse_knockout(self, k):
        parts = k.split("+")
        if parts[0] != 'ctrl':
            knock_gene = parts[0]
        else:
            if len(parts) > 1:
                knock_gene = parts[1]
            else:
                knock_gene = 'ctrl'

        if knock_gene == 'ctrl':
            return -1
        else:
            try:
                return self.gene_list.index(knock_gene)
            except ValueError:
            
                return -1 

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        sample = self.data_samples[idx]
        
        ctype_name = sample['cell_type_name']
        
        ctrl_mean = self.ctrl_stats[ctype_name]['mean']
        ctrl_var = self.ctrl_stats[ctype_name]['var']

        return {
            'cell_type': torch.tensor(sample['cell_type_idx'], dtype=torch.long),
            'knockout': torch.tensor(sample['knockout_idx'], dtype=torch.long),
            'ctrl_mean': ctrl_mean,  # Input: Control Prior
            'ctrl_var': ctrl_var,    # Input: Control Prior
            'target_prob': sample['marginal_prob'] # Target: Soft Label (0~1)
        }

def return_mask_dataloader(adata,cell_type,adata_ctrl=None,mole_embed=None,mole_list=None,gene_name=None,source_model=True,data_name="adamson",pert_type="molecular",batch_size=32):
    if pert_type=="molecular":
        return DataLoader(TargetModelDataset_Molecular(adata,adata_ctrl=adata_ctrl,cell_list=cell_type,mole_embed=mole_embed,mole_list=mole_list),batch_size=batch_size,shuffle=True)
    else:
        return DataLoader(MaskModelDataset_Gene(adata,adata_ctrl=adata_ctrl,cell_list=cell_type,gene_list=gene_name,data_name=data_name),batch_size=batch_size,shuffle=True)
