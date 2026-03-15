'''
This code is adapted from :
https://github.com/siyuh/Squidiff/tree/main

with modifications and extensions based on our specific requirements.
'''
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import math

# from Dataset.scFoundation.DeepCDR.prog.process_drug import features
from nn import timestep_embedding

from torch_geometric.nn import GATConv
from torch_geometric.utils import from_networkx
from torch_geometric.data import Batch,Data
# from dgl.nn.pytorch import GATConv


class MLP(nn.Module):
    def __init__(self,
                 sizes,
                 batch_norm=True,
                 last_layer_act='linear',
                 append_layer_width=None,
                 append_layer_position=None,
                 act="SiLU",
                 dropout_rate=0.1,
                 ):
        super(MLP, self).__init__()
        self.batch_norm = batch_norm
        self.layers = []

        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))

            if batch_norm and i < len(sizes) - 2:
                self.layers.append(nn.BatchNorm1d(sizes[i + 1]))

            if i < len(sizes) - 2:
                if act == "ReLU":
                    self.layers.append(nn.ReLU())
                elif act == "SiLU":
                    self.layers.append(nn.SiLU())

                if dropout_rate > 0.0:
                    self.layers.append(nn.Dropout(dropout_rate))


        if append_layer_width is not None:
            if append_layer_position == "first":
                self.layers.insert(0, nn.Linear(sizes[0], append_layer_width))
                self.layers.insert(1, nn.BatchNorm1d(append_layer_width))
                self.layers.insert(2, nn.ReLU())
            elif append_layer_position == "last":
                self.layers.append(nn.Linear(sizes[-1], append_layer_width))
                self.layers.append(nn.BatchNorm1d(append_layer_width))
                self.layers.append(nn.ReLU())



        if last_layer_act == "ReLU":
            self.layers.append(nn.ReLU())
        elif last_layer_act == "linear":
            self.layers.append(nn.Identity())
        elif last_layer_act == "LeakyReLU":
            self.layers.append(nn.LeakyReLU(negative_slope=0.01))

        self.network = nn.Sequential(*self.layers)

    def forward(self, x):
        is_single = False

        if x.dim() == 3:
            b, l, d = x.shape
            if b == 1 and self.batch_norm:
                is_single = True
                x = x.repeat(2, 1, 1)
                b = 2
            x = x.view(b * l, d)
            x = self.network(x)
            x = x.view(b, l, -1)
            if is_single:
                x = x[0:1]
        else:
            if x.shape[0] == 1 and self.batch_norm:
                is_single = True
                x = x.repeat(2, 1)
            x = self.network(x)
            if is_single:
                x = x[0:1]

        return x


class Block_A(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 time_embed_dim,
                 cell_type_embed_dim,
                 depth=3,
                 ):
        super(Block_A, self).__init__()
        '''
        time_embed_dim equals to cell_type_embed_dim
        '''
        # self.mlp=MLP([gene_init_dim+1]+2*[2*hidden_dim]+[output_dim])
        self.mlp1 = MLP([input_dim] + depth*[output_dim],act="SiLU")
        self.time_encoder = MLP([time_embed_dim]+2*[output_dim],act="SiLU")
        self.mlp2 = MLP([output_dim]+(depth)*[output_dim],act="SiLU")
        self.cell_type_encoder=MLP([cell_type_embed_dim]+2*[output_dim],act="SiLU")
        self.mlp3 = MLP((depth+1)*[output_dim],act="SiLU")

    def forward(self,x_l,time_emb,cell_type=None):
        '''
        :param x_l: input, shape:[batch_size,gene_num]
        '''
        f=self.mlp1(x_l)
        f=f+self.time_encoder(time_emb)
        f=self.mlp2(f)
        if cell_type is not None:
            f=f+self.cell_type_encoder(cell_type)
        f=self.mlp3(f)
        return f


class Block_B(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 time_embed_dim,
                 cell_type_embed_dim,
                 depth=3,
                 ):
        super(Block_B, self).__init__()
        '''
        time_embed_dim equals to cell_type_embed_dim
        '''
        # self.mlp=MLP([gene_init_dim+1]+2*[2*hidden_dim]+[output_dim])
        self.mlp1 = MLP([input_dim] + depth*[output_dim],act="SiLU")
        self.time_encoder_1 = MLP([time_embed_dim]+depth*[output_dim],act="SiLU")
        self.cell_type_encoder_1=MLP([cell_type_embed_dim]+depth*[output_dim],act="SiLU")
        self.mlp2 = MLP((depth+1)*[output_dim],act="SiLU")
        self.time_encoder_2=MLP([time_embed_dim]+depth*[output_dim],act="SiLU")
        self.cell_type_encoder_2=MLP([cell_type_embed_dim]+depth*[output_dim],act="SiLU")
        self.mlp3 = MLP((depth+1)*[output_dim],act="SiLU",last_layer_act="ReLU")

    def forward(self,x_l,time_emb,cell_type=None):
        '''
        :param x_l: input, shape:[batch_size,gene_num]
        '''
        f=self.mlp1(x_l)
        f=f+self.time_encoder_1(time_emb)
        if cell_type is not None:
            f=f+self.cell_type_encoder_1(cell_type)
        f=self.mlp2(f)
        f = f + self.time_encoder_2(time_emb)
        if cell_type is not None:
            f = f + self.cell_type_encoder_2(cell_type)
        f=self.mlp3(f)
        return f


class Dosager(nn.Module):
    def __init__(self,
                 dim,
                 ):
        """
        The `dosager` module combines molecular representations (`x`) and
            dosage information (`dosage`) to model the effect of dosage on the molecular response.
        """
        super(Dosager, self).__init__()
        self.mlp=MLP(sizes=[dim+1,dim,dim,1])

    def forward(self, x, dosage):
        x_dosage = th.cat((x, dosage), dim=1)
        return self.mlp(x_dosage).sigmoid()


class Mole_encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 ):
        """
        :param input_dim: dimension of input molecular features extracted by Uni-Mol
        :param output_dim: dimension of output molecular features, equal to gene embedding dimension
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dosager = Dosager(dim=output_dim)
        self.mole_mlp = MLP(sizes=[input_dim,input_dim,output_dim])

    def forward(self, mole_embedding, dosage):
        """
        :param
         mole_embedding: molecule embedding extracted by pretrained Uni-Mol
         dosage: dosage of molecular drug
        """
        # mole_embedding shape [b,output_dim]
        mole_embedding = self.mole_mlp(mole_embedding)

        # dosage_embed shape [b,1]
        dosage_embed = self.dosager(mole_embedding, dosage)

        # mole_dosage_embed shape [b,output_dim]
        mole_dosage_embed = dosage_embed*mole_embedding

        return mole_dosage_embed


from torch_geometric.nn import GATConv
from torch.nn import LayerNorm

class MultiLayerGAT(nn.Module):
    def __init__(self, dim, heads=[4, 4], dropout=0.1, use_residual=False, use_norm=True):
        super().__init__()
        self.layers = nn.ModuleList()
        self.heads = heads
        self.dropout = dropout
        self.use_residual = use_residual
        self.use_norm = use_norm

        self.norms = nn.ModuleList()

        for i in range(len(dim) - 1):
            in_dim = dim[i]
            out_dim = dim[i + 1] // heads[i]
            self.layers.append(GATConv(in_dim, out_dim, heads=heads[i], concat=True))
            if self.use_norm:
                self.norms.append(LayerNorm(dim[i + 1]))


    def forward(self, g, feature):  # feature: [B, N, D]
        B, N, D = feature.shape
        x = feature.reshape(B * N, D)  # Flattening the feature

        # Create the batched edge_index using PyG's Batch
        edge_index = g.edge_index  # [2, E]

        # Batch the graph data using Batch.from_data_list
        data_list = []
        for b in range(B):
            data = Data(x=x[b * N: (b + 1) * N], edge_index=edge_index)
            data_list.append(data)

        batch = Batch.from_data_list(data_list)
        x = batch.x  # [B*N, D]
        edge_index = batch.edge_index  # [2, E]

        for i, layer in enumerate(self.layers):
            h = layer(x, edge_index)  # [B*N, hidden_dim]

            if self.use_residual:
                if h.shape == x.shape:
                    h = h + x  # Add residual connection if the dimensions match
            if self.use_norm:
                h = self.norms[i](h)

            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

            x = h

        x = x.view(B, N, -1)  # Reshaping the result back to [B, N, hidden_dim]
        return x


class GRN_conditional_network(nn.Module):
    def __init__(self,
                 gene_num,
                 init_gene_emb,
                 gene_dim,
                 mole_dim,
                 output_dim,
                 GRN,
                 hidden_dim,
                 cell_type_embed_dim,
                 cell_type_num,
                 pert_type="gene",
                 ):
        """
        :param gene_num:
        :param gene_init_dim: dimension of initial gene embedding and self.mole_gene_specific_encoder's output
        :param mole_dim: dimension of molecular features extracted by Uni-Mol
        :param output_dim: output dimension of self.gat
        :param GRN: graph data structure
        :param pert_type: "gene" or "molecular"
        """
        super(GRN_conditional_network, self).__init__()
        self.gene_num = gene_num
        self.gene_init_dim = gene_dim
        self.pert_type = pert_type
        self.scale = math.sqrt(hidden_dim)

        self.grn=GRN
        if pert_type=="gene":
            self.ln_gene = nn.LayerNorm(gene_dim)
            self.gene_embedding=nn.Embedding(gene_num,gene_dim)
            pretrained_weight = th.from_numpy(init_gene_emb).float()
            self.gene_embedding.weight.data.copy_(pretrained_weight)
            self.gene_embedding.weight.requires_grad = False

        if cell_type_num >1:
            self.cell_type_encoder=MLP(sizes=[cell_type_embed_dim]+2*[hidden_dim])

        """
        time embed and cell type embed
        """
        self.ln = nn.LayerNorm(hidden_dim)
        self.gnn=MultiLayerGAT(dim=[gene_dim] + 2*[hidden_dim],heads=[2,2])
        self.decoder=MLP(sizes=[hidden_dim]+3*[output_dim],act="SiLU")

        self.control_token = nn.Parameter(th.randn(1, hidden_dim)) 
        nn.init.xavier_normal_(self.control_token)

        # self.mlp_1 = MLP(sizes=[2 * gene_num] + 2*[gene_num],last_layer_act='relu')
        if self.pert_type == "molecular":
            self.mole_encoder = Mole_encoder(input_dim=mole_dim, output_dim=hidden_dim)
            self.gene_mole_encoder=MLP(sizes=[2*gene_dim] + 2*[gene_dim])


    def forward(self,x_t=None,x_noisy=None,time_emb =None,cell_type_emb=None,mole=None,dosage=None,knockout=None,single=True):

        if mole is None and dosage is None and knockout is None: # control cells
            output = self.control_token.repeat(time_emb.shape[0],1)

        else:
            if self.pert_type == "gene":
                gene_initialization = self.gene_embedding.weight.repeat(time_emb.shape[0], 1, 1) # shape: [gene_num,d]
                gene_initialization = self.ln_gene(gene_initialization)
                res = self.gnn(feature=gene_initialization, g=self.grn)
                batch_idx = th.arange(res.size(0)).to(res.device)
                if knockout.shape[-1]==2:
                    output = res[batch_idx.reshape(-1,1),knockout,:]
                else:
                    output = res[batch_idx,knockout.squeeze(),:]

            elif self.pert_type == "molecular":
                output = self.mole_encoder(mole, dosage)

        if cell_type_emb is not None:
                output += self.ln(self.cell_type_encoder(cell_type_emb))

        if output.shape[1]==2:
            output = self.decoder(output).sum(dim=1)
        else:
            output = self.decoder(output)
                    
        return output


class SourceModel(nn.Module):
    """
    only consider cell type
    Source model is used to reconstruct control cells
    """
    def __init__(self,
                 gene_num,
                 hidden_dim,
                 GRN,
                 time_pos_dim=1024,
                 cell_type_num=None,
                 cell_type_embed_dim=1024,
                 data_name="adamson",
                 ):
        super().__init__()
        self.GRN=GRN
        self.gene_num = gene_num
        self.data_name = data_name
        self.time_embed = None
        self.time_pos_dim = time_pos_dim
        self.cell_type_num = cell_type_num # number of cell type

        self.block_1=Block_A(
            input_dim=gene_num,
            output_dim=hidden_dim,
            time_embed_dim=time_pos_dim,
            cell_type_embed_dim=cell_type_embed_dim,
        )

        self.decoder_control=MLP(sizes=[hidden_dim]+3*[2*hidden_dim]+[gene_num])
        self.decoder_treated=MLP(sizes=[hidden_dim]+3*[2*hidden_dim]+[gene_num])

    def forward(self,x,x_noisy=None,cond=None,time_emb=None,cell_type_emb=None):
        h=self.block_1(x, time_emb=time_emb, cell_type=cell_type_emb)
        f = h+cond
        if x_noisy is None:
            res=self.decoder_control(f)
        else:
            res=self.decoder_treated(f)+x_noisy

        return res


class TargetModel(nn.Module):
    """
    only consider cell type
    Source model is used to reconstruct control cells
    """
    def __init__(self,
                 gene_num,
                 GRN,
                 init_gene_emb,
                 gene_dim=32,
                 hidden_dim=512,
                 time_pos_dim=512,
                 gene_wise_embed_dim=512,
                 cell_type_num=None,
                 time_embed_dim=512,
                 cell_type_embed_dim=512,
                 data_name="adamson",
                 mole_dim=512,
                 pert_type="gene",
                 ):
        super().__init__()
        """
        latent_dim: dimension for cell type embedding
        """
        self.gene_num = gene_num
        self.data_name = data_name
        self.time_embed_dim = time_embed_dim
        self.hidden_dim = hidden_dim
        # self.time_encoder = MLP(sizes=[time_pos_dim]+2*[time_embed_dim],act="SiLU")
        self.cell_type_num = cell_type_num # number of cell type
        self.time_pos_dim = time_pos_dim

        if cell_type_num >1:
            self.cell_type_embedding = nn.Embedding(cell_type_num, cell_type_embed_dim)
        else:
            self.cell_type_embedding = None

        self.source_model=SourceModel(
            gene_num=gene_num,
            hidden_dim=hidden_dim,
            GRN=GRN,
            time_pos_dim=time_pos_dim,
            cell_type_num=cell_type_num,
            cell_type_embed_dim=cell_type_embed_dim,
            data_name=data_name,
        )

        self.control_net=GRN_conditional_network(gene_num=gene_num,
                                                 init_gene_emb=init_gene_emb,
                                                 gene_dim=gene_dim,
                                                 mole_dim=mole_dim,
                                                 output_dim=hidden_dim,
                                                 GRN=GRN,
                                                 hidden_dim=gene_wise_embed_dim,
                                                 cell_type_num=cell_type_num,
                                                 cell_type_embed_dim=cell_type_embed_dim,
                                                 pert_type=pert_type,
                                                 )


    def load_pretrained_source_model(self, pretrained_source_model_path):
        self.source_model.load_state_dict(th.load(pretrained_source_model_path))


    def forward(self, x,timesteps=None,knockout=None,cell_type=None,mole=None, dosage=None,ctrl_mean=None, ctrl_var=None,single=True):
        '''
        add random noise to x0
        '''

        B=x.shape[0]
        if ctrl_var is not None and ctrl_mean is not None:
            x_noisy = ctrl_mean + th.sqrt(ctrl_var)* th.randn_like(x)
        elif ctrl_mean is not None:
            x_noisy = ctrl_mean
        else:
            x_noisy = None

        h_all = th.zeros_like(x)

        if single is True:
            is_ctrl_mole = (dosage.squeeze() == 0) if dosage is not None else th.zeros(B, dtype=th.bool, device='cuda')
            is_ctrl_gene = (knockout.squeeze() == -1) if knockout is not None else th.zeros(B, dtype=th.bool, device='cuda')
            is_ctrl = is_ctrl_mole | is_ctrl_gene
        else:
            is_ctrl = th.zeros(B, dtype=th.bool, device='cuda')

        idx_ctrl = th.nonzero(is_ctrl).squeeze(-1)
        idx_cond = th.nonzero(~is_ctrl).squeeze(-1)


        if len(idx_ctrl) > 0:
            cond_info_cond = self.control_net(
                x_t=x[idx_ctrl],
                x_noisy=None,
                # time_emb = self.time_encoder(timestep_embedding(timesteps[idx_ctrl], self.time_pos_dim)),
                time_emb=timestep_embedding(timesteps[idx_ctrl], self.time_pos_dim),
                cell_type_emb = self.cell_type_embedding(cell_type[idx_ctrl].view(-1)) if self.cell_type_embedding is not None else None,
                mole=None,
                dosage=None,
                knockout=None,
            )

            h_ctrl = self.source_model(
                x=x[idx_ctrl],
                cond = cond_info_cond,
                # time_emb=self.time_encoder(timestep_embedding(timesteps[idx_ctrl], self.time_pos_dim)),
                time_emb=timestep_embedding(timesteps[idx_ctrl], self.time_pos_dim),
                cell_type_emb=self.cell_type_embedding(
                    cell_type[idx_ctrl].view(-1)) if self.cell_type_embedding is not None else None,
            )

            h_all[idx_ctrl] = h_ctrl

        if len(idx_cond) > 0:
            cond_info_cond = self.control_net(
                x_t=x[idx_cond],
                x_noisy=x_noisy[idx_cond],
                # time_emb=self.time_encoder(timestep_embedding(timesteps[idx_cond], self.time_pos_dim)),
                time_emb=timestep_embedding(timesteps[idx_cond], self.time_pos_dim),
                cell_type_emb=self.cell_type_embedding(cell_type[idx_cond].view(-1)) if self.cell_type_embedding is not None else None,
                mole=mole[idx_cond] if mole is not None else None,
                dosage=dosage[idx_cond] if dosage is not None else None,
                knockout=knockout[idx_cond] if knockout is not None else None,
                single=single,
            )
            h_cond = self.source_model(
                x=x[idx_cond],
                x_noisy=x_noisy[idx_cond],
                cond=cond_info_cond,
                # time_emb=self.time_encoder(timestep_embedding(timesteps[idx_cond], self.time_pos_dim)),
                time_emb=timestep_embedding(timesteps[idx_cond], self.time_pos_dim),
                cell_type_emb=self.cell_type_embedding(
                    cell_type[idx_cond].view(-1)) if self.cell_type_embedding is not None else None,
            )
            h_all[idx_cond] = h_cond

        # return h_all+ctrl_base
        return h_all


class GRN_conditional_network_for_mask(nn.Module):
    def __init__(self,
                 gene_num,
                 gene_init_dim,
                 mole_dim,
                 output_dim,
                 GRN,
                 cell_type_embed_dim,
                 cell_type_num,
                 pert_type="gene",
                 ):
        """
        :param gene_num:
        :param gene_init_dim: dimension of initial gene embedding and self.mole_gene_specific_encoder's output
        :param mole_dim: dimension of molecular features extracted by Uni-Mol
        :param output_dim: output dimension of self.gat
        :param GRN: graph data structure
        :param pert_type: "gene" or "molecular"
        """
        super(GRN_conditional_network_for_mask, self).__init__()
        self.gene_num = gene_num
        self.gene_init_dim = gene_init_dim
        self.pert_type = pert_type

        self.grn = GRN
        self.gene_embedding = nn.Embedding(gene_num, gene_init_dim)
        self.indv_w = nn.Parameter(th.rand(1, gene_num, output_dim))
        #
        nn.init.xavier_normal_(self.indv_w)
        self.indv_b = nn.Parameter(th.rand(1, gene_num))
        nn.init.xavier_normal_(self.indv_b)
        # nn.init.xavier_normal_(self.indv_b)

        if cell_type_num > 1:
            self.cell_type_encoder = MLP(sizes=[cell_type_embed_dim] + 2 * [gene_init_dim])

        self.mlp_1=MLP(sizes=3*[gene_num])
        self.mlp_2=MLP(sizes=3*[gene_num])

        self.gene_encoder = MLP(sizes=[2*gene_init_dim]+3 * [gene_init_dim])
        self.x_noisy_encoder = MLP(sizes=[1] + 2 * [gene_init_dim])
        self.decoder=MLP(sizes=4*[output_dim])
        self.ln = nn.LayerNorm(gene_init_dim)

        self.ko_token = nn.Parameter(th.randn(1, 1, gene_init_dim)) 
        nn.init.xavier_normal_(self.ko_token)

        self.gnn = MultiLayerGAT(dim=[gene_init_dim] +  [output_dim], heads=[2])
        # self.mlp_1 = MLP(sizes=[2 * gene_num] + 2*[gene_num],last_layer_act='relu')
        if self.pert_type == "molecular":
            self.mole_encoder = Mole_encoder(input_dim=mole_dim, output_dim=gene_init_dim)
            self.gene_mole_encoder = MLP(sizes=[2 * gene_init_dim] + 2 * [gene_init_dim])

    def forward(self, x_noisy=None, cell_type_emb=None, mole=None, dosage=None, knockout=None,single=True):
        """
        :param x_l: latent feature
        :return: node embeddings
        """
        gene_initialization = self.gene_embedding.weight.repeat(x_noisy.shape[0], 1, 1)
        gene_initialization = self.ln(gene_initialization)
        gene_initialization = (gene_initialization +self.ln(self.x_noisy_encoder(x_noisy.unsqueeze(-1))))

        if cell_type_emb is not None:
            gene_initialization = gene_initialization + self.ln(self.cell_type_encoder(cell_type_emb.squeeze()).unsqueeze(1))

        if self.pert_type == "gene":
            """
            mask corresponding gene node feature
            """
            if single is True:
                mask = th.ones(len(knockout), self.gene_num, self.gene_init_dim).to('cuda')
                mask.scatter_(1, knockout.view(-1, 1, 1).expand(-1, -1, self.gene_init_dim), 0)
                # masked_gene_initialization = gene_initialization * mask
                # res = self.gnn(feature=masked_gene_initialization, g=self.grn)  # shape: [b,gene_num,output_dim]
            else:
                mask = th.ones(len(knockout), self.gene_num, self.gene_init_dim).to('cuda')
                expanded_idx = knockout.unsqueeze(-1).expand(-1, -1, self.gene_init_dim)
                mask.scatter_(1, expanded_idx, 0)
                # masked_gene_initialization = gene_initialization * mask
                # res = self.gnn(feature=masked_gene_initialization, g=self.grn)
            masked_gene_initialization = gene_initialization * mask + self.ko_token * (1 - mask)
            res = self.gnn(feature=masked_gene_initialization, g=self.grn)

        else:
            mole_dosage_emb = self.mole_encoder(mole, dosage)
            gene_initialization = self.gene_mole_encoder(
                th.cat([gene_initialization, mole_dosage_emb.unsqueeze(1).expand(-1, self.gene_num, -1)], dim=2))
            res = self.gnn(feature=gene_initialization, g=self.grn)  # shape: [b,gene_num,output_dim]


        # res = self.gene_encoder(res)
        f = self.decoder(res)
        gwf = (f * self.indv_w.squeeze(-1)).sum(dim=-1, keepdim=True)
        gwf = gwf.view(x_noisy.shape[0], -1)+self.indv_b
        # output = gwf+x_noisy

        # h=self.mlp_1(gwf+x_noisy)
        # output=self.mlp_2(h+x_noisy)

        output=gwf+x_noisy

        return output


class MaskModel(nn.Module):
    """
    predict where is zero
    """
    def __init__(self,
                 gene_num,
                 GRN,
                 gene_init_dim=64,
                 cell_type_num=None,
                 cell_type_embed_dim=512,
                 data_name="adamson",
                 mole_dim=512,
                 pert_type="gene",
                 ):
        super().__init__()
        """
        latent_dim: dimension for cell type embedding
        """
        self.gene_num = gene_num
        self.data_name = data_name
        self.cell_type_num = cell_type_num # number of cell type

        if cell_type_num >1:
            self.cell_type_embedding = nn.Embedding(cell_type_num, cell_type_embed_dim)
        else:
            self.cell_type_embedding = None

        self.control_net=GRN_conditional_network_for_mask(gene_num=gene_num,
                                                          gene_init_dim=gene_init_dim,
                                                          mole_dim=mole_dim,
                                                          output_dim=gene_init_dim,
                                                          GRN=GRN,
                                                          cell_type_num=cell_type_num,
                                                          cell_type_embed_dim=cell_type_embed_dim,
                                                          pert_type=pert_type,
                                                          )

    def forward(self, x, knockout=None,cell_type=None,mole=None, dosage=None,ctrl_mean=None, ctrl_var=None):
        '''
        add random noise to x0
        '''
        if ctrl_var is not None and ctrl_mean is not None:
            x_noisy = ctrl_mean + th.sqrt(ctrl_var)* th.randn_like(x)
        elif ctrl_mean is not None:
            x_noisy = ctrl_mean
        else:
            x_noisy = None


        h = self.control_net(
            x_noisy=x_noisy,
            # time_emb=self.time_encoder(timestep_embedding(timesteps[idx_cond], self.time_pos_dim)),
            cell_type_emb=self.cell_type_embedding(cell_type) if self.cell_type_embedding is not None else None,
            mole=mole if mole is not None else None,
            dosage=dosage if dosage is not None else None,
            knockout=knockout if knockout is not None else None
        )
        prob=th.sigmoid(h)
        label=(th.tensor(x)!=0).float()
        loss_fc=nn.BCELoss()
        loss = loss_fc(prob, label)

        return loss

    def predict(self,x,knockout=None,cell_type=None,mole=None, dosage=None,threshold=0.5):
        with th.no_grad():
            x_noisy = x
            h = self.control_net(
                x_noisy=x_noisy,
                # time_emb=self.time_encoder(timestep_embedding(timesteps[idx_cond], self.time_pos_dim)),
                cell_type_emb=self.cell_type_embedding(cell_type) if self.cell_type_embedding is not None else None,
                mole=mole if mole is not None else None,
                dosage=dosage if dosage is not None else None,
                knockout=knockout if knockout is not None else None
            )
            prob=th.sigmoid(h)

        return prob


    def predict_double(self,x,knockout=None,cell_type=None,mole=None, dosage=None,threshold=0.5):
        with th.no_grad():
            x_noisy = x
            h = self.control_net(
                x_noisy=x_noisy,
                # time_emb=self.time_encoder(timestep_embedding(timesteps[idx_cond], self.time_pos_dim)),
                cell_type_emb=self.cell_type_embedding(cell_type) if self.cell_type_embedding is not None else None,
                mole=mole if mole is not None else None,
                dosage=dosage if dosage is not None else None,
                knockout=knockout if knockout is not None else None,
                single=False
            )
            prob=th.sigmoid(h)

        return prob

