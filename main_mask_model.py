from model import MaskModel
from train import *
import argparse
from Dataset.Preprocess import *
from Dataset.Datasets import *
from torch.optim import AdamW


if __name__ == "__main__":
    args_train = parse_args()

    main_cfg = {}
    if args_train.config is not None:
        with open(args_train.config) as f:
            cfg = yaml.safe_load(f)

        main_cfg = cfg.get("main", {})

    args_train.epoch = main_cfg['epoch']


    args_train.resume_checkpoint = "./result/target_" + args_train.data_name + "_" + str(args_train.gene_num)

    if not os.path.exists(args_train.resume_checkpoint):
        os.makedirs(args_train.resume_checkpoint)

    print('**************training args*************')

    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)

    if args_train.pert_type=='molecular':
        pert_data = PertData(hvg_num=args_train.gene_num, pert_type=args_train.pert_type,
                             data_name=args_train.data_name, path=current_dir, threshold=args_train.threshold,
                             threshold_co=args_train.threshold_co)
        data = return_dataloader(pert_data.train_cell_treated, adata_ctrl=pert_data.train_cell, cell_type=pert_data.cell_type,
                                 mole_embed=pert_data.mole_embed, mole_list=pert_data.mole,
                                 source_model=False, pert_type=args_train.pert_type,
                                 batch_size=args_train.batch_size)

    else:
        pert_data = PertData(hvg_num=args_train.gene_num,pert_type=args_train.pert_type,data_name=args_train.data_name,
                           path=current_dir,threshold=args_train.threshold,threshold_co=args_train.threshold_co)
        data = return_dataloader(pert_data.train_cell_treated,adata_ctrl=pert_data.train_cell,cell_type=pert_data.cell_type,
                               gene_name=pert_data.gene_name,source_model=False,pert_type=args_train.pert_type,
                               batch_size=args_train.batch_size)
        # data=return_dataloader(pert_data.train_cell_treated,adata_ctrl=pert_data.train_cell,cell_type=pert_data.cell_type,gene_name=pert_data.gene_name,source_model=args_train.source_model,pert_type=args_train.pert_type,batch_size=args_train.batch_size)

    mask_model = MaskModel(
        gene_num=args_train.gene_num,
        GRN=pert_data.GRN,
        cell_type_num=pert_data.cell_type_num,
        data_name=args_train.data_name,
        mole_dim=args_train.mole_dim,
        pert_type=args_train.pert_type,
    ).to('cuda')

    mask_model.train()

    optimizer = AdamW(mask_model.parameters(), lr=1e-2, weight_decay=1e-2)
    loss_list = []

    for e in range(args_train.epoch):
        pbar = tqdm(data, desc=f"Epoch {e + 1}")
        for batch in pbar:
            if args_train.pert_type=="gene":
                loss=mask_model(
                    x=batch['feature'],
                    knockout=batch['knockout'],
                    cell_type=batch['cell_type'],
                    ctrl_mean=batch['ctrl_mean'],
                    ctrl_var=batch['ctrl_var'],
                )
            elif args_train.pert_type=="molecular":
                loss=mask_model(
                    x=batch['feature'],
                    mole=batch['mole'],
                    dosage=batch['dosage'].to('cuda'),
                    cell_type=batch['cell_type'],
                    ctrl_mean=batch['ctrl_mean'],
                    ctrl_var=batch['ctrl_var'],
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())

            pbar.set_postfix(loss=f"{loss.item():.4f}")

    th.save(mask_model.state_dict(), args_train.resume_checkpoint+"/mask_model.pt")

    import matplotlib.pyplot as plt

    plt.plot(loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training Loss Curve")
    plt.savefig("loss_curve.png")
    plt.close()