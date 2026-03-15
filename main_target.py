from train import *
import argparse
from Dataset.Preprocess import *
from Dataset.Datasets import *
import yaml


if __name__ == "__main__":
    args_train = parse_args()

    args_train.logger_path = "./result/logs/target_" + args_train.data_name + "_" + str(args_train.gene_num)
    args_train.resume_checkpoint = "./result/target_" + args_train.data_name + "_" + str(args_train.gene_num)

    if not os.path.exists(args_train.resume_checkpoint):
        os.makedirs(args_train.resume_checkpoint)

    print('**************training args*************')
    print(args_train)
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)

    if args_train.pert_type=='molecular':
        pert_data=PertData(hvg_num=args_train.gene_num,pert_type=args_train.pert_type,data_name=args_train.data_name,path=current_dir)

        data=return_dataloader(pert_data.train_cell,adata_ctrl=pert_data.train_cell,cell_type=pert_data.cell_type,
                               mole_embed=pert_data.mole_embed,mole_list=pert_data.mole,source_model=args_train.source_model,
                               pert_type=args_train.pert_type,batch_size=args_train.batch_size)
        losses_target = run_training(data=data,cell_type_num=pert_data.cell_type_num,GRN=pert_data.GRN,args=args_train,init_gene_emb=None)
        
    else:
        pert_data=PertData(hvg_num=args_train.gene_num,pert_type=args_train.pert_type,data_name=args_train.data_name,
                           path=current_dir,threshold=args_train.threshold,threshold_co=args_train.threshold_co)
        data=return_dataloader(pert_data.train_cell,adata_ctrl=pert_data.train_cell,cell_type=pert_data.cell_type,
                               gene_name=pert_data.gene_name,source_model=args_train.source_model,pert_type=args_train.pert_type,
                               batch_size=args_train.batch_size)
        losses_target = run_training(data=data,cell_type_num=pert_data.cell_type_num,
                            GRN=pert_data.GRN,init_gene_emb=pert_data.gene_emb,args=args_train)

    
    plot_loss(losses_target, args_train)