from prometheus_client import samples

from test_util import *
from train import *
from Dataset.Preprocess import *
import os
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.special import kl_div
from collections import defaultdict

if __name__ == "__main__":
    args_test = parse_args()
    
    main_cfg = {}

    if args_test.config is not None:
        with open(args_test.config) as f:
            cfg = yaml.safe_load(f)

        main_cfg = cfg.get("main", {})

    from_source = main_cfg['from_source']
    pred_ood = main_cfg['pred_ood']


    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)

    if args_test.pert_type == 'molecular':
        pert_data = PertData(hvg_num=args_test.gene_num, pert_type=args_test.pert_type, data_name=args_test.data_name,
                            path=current_dir, threshold=args_test.threshold, threshold_co=args_test.threshold_co)

    else:
        pert_data = PertData(hvg_num=args_test.gene_num, pert_type=args_test.pert_type, data_name=args_test.data_name,
                            path=current_dir, threshold=args_test.threshold,threshold_co=args_test.threshold_co)


    model_path = "./result/target_" + args_test.data_name + "_" + str(args_test.gene_num) + "/model.pt"
    source_path="./result/source_"+args_test.data_name+"_"+str(args_test.gene_num)+"/model.pt"

    if from_source==False:
        result_path="./result/prediction/"+args_test.data_name+".json"
    else:
        if pred_ood==True:
            result_path="./result/prediction/"+args_test.data_name+"_from_source_only_ood.json"

        else:
            result_path = "./result/prediction/" + args_test.data_name + "_from_source.json"


    if os.path.exists(result_path):
        with open(result_path, "r", encoding="utf-8") as f:
            result = json.load(f)
    else:
        cond_dict = defaultdict(dict)
        if args_test.pert_type=="gene":
            for cond in tqdm(pert_data.train_cond, desc="Perturbation conditions"):
                if args_test.data_name == "norman":
                    current_test_cell = pert_data.adata[pert_data.adata.obs['knockout'] == cond]
                else:
                    current_test_cell = pert_data.adata[pert_data.adata.obs['condition'] == cond]
                current_cell_type = list(current_test_cell.obs['cell_type'].unique())
                
                for cell_type in current_cell_type:
                    specific_type_cell = current_test_cell[current_test_cell.obs['cell_type'] == cell_type]
                    key = args_test.pert_type + "_" + cond + "_" + cell_type

                    sample_mask = (specific_type_cell.X != 0).astype(float)  # [num_samples, gene_num]
                    
                    overall_prob = np.mean(sample_mask, axis=0)  # [gene_num]

                    cond_dict[key]['sample_mask'] = sample_mask
                    cond_dict[key]['overall_prob'] = overall_prob
        else:
            train_cells = pert_data.adata[pert_data.adata.obs['unlasting_split'] == 'train']
            train_smiles = list(train_cells.obs['SMILES'].unique())

            for mole in tqdm(train_smiles, desc="Training drug covariate conditions"):
                current_train_cell = train_cells[train_cells.obs['SMILES'] == mole]
                current_cell_type = list(current_train_cell.obs['cell_type'].unique())

                for cell_type in current_cell_type:
                    specific_type_cell = current_train_cell[current_train_cell.obs['cell_type'] == cell_type]
                    all_dosage = list(specific_type_cell.obs['dose_val'].unique())
                    cell_type_idx = pert_data.cell_type.index(cell_type)
                    for dosage in all_dosage:
                        specific_type_dosage_cell = specific_type_cell[specific_type_cell.obs['dose_val'] == dosage]
                        key = args_test.pert_type + "_" + mole + "_" + cell_type + "_" + str(dosage)
                        sample_mask = (specific_type_dosage_cell.X != 0).astype(float)  # [num_samples, gene_num]
                        overall_prob = np.mean(sample_mask, axis=0)  # [gene_num]

                        cond_dict[key]['sample_mask'] = sample_mask
                        cond_dict[key]['overall_prob'] = overall_prob

        result = predict(pert_data=pert_data,
                        model_path=model_path,
                        source_path=source_path,
                        args=args_test,
                        from_source=from_source,
                        pred_ood=pred_ood,
                        sample_num=100,
                        cond_dict=cond_dict)

    EMD=[]
    RMSE_all=[]
    E_distance_all=[]

    RMSE_DE20=[]
    EMD_DE20=[]
    E_distance_DE20=[]

    RMSE_DE40=[]
    E_distance_DE40=[]
    EMD_DE40=[]

    max=pert_data.max
    pert_data.recover()


    if args_test.pert_type=='molecular':
        print("see test_sciplex3.py for details")
        if pred_ood==True:
            for mole in tqdm(pert_data.ood_cond, desc="OOD drugs conditions"):
                current_test_cell = pert_data.adata[pert_data.adata.obs['SMILES'] == mole]
                current_cell_type = list(current_test_cell.obs['cell_type'].unique())

                for cell_type in current_cell_type:
                    specific_type_cell = current_test_cell[current_test_cell.obs['cell_type'] == cell_type]

                    all_dosage = list(specific_type_cell.obs['dose_val'].unique())
                    cell_type_idx = pert_data.cell_type.index(cell_type)
                    for dosage in all_dosage:
                        specific_type_dosage_cell = specific_type_cell[specific_type_cell.obs['dose_val'] == dosage]

                        key = args_test.pert_type + "_" + mole + "_" + cell_type + "_" + str(dosage)
                        pred_cell_expr = result[key]
                        pred_cell_expr = np.array(pred_cell_expr)
                        # recover
                        pred_cell_expr = pred_cell_expr * max
                        pred_cell_expr[pred_cell_expr < 0] = 0

                        control_type_cell = pert_data.adata[pert_data.adata.obs['cell_type'] == cell_type]
                        control_type_cell = control_type_cell[control_type_cell.obs['condition'] == 'ctrl']

                        # calculate E-distance
                        E = compute_e_distance(np.array(pred_cell_expr), specific_type_dosage_cell.X.toarray())
                        E_distance_all.append(E)

                        """
                        DE
                        """
                        genes=pert_data.adata.var.index.tolist()
                        DE=pert_data.adata.uns['lincs_DEGs']
                        DE20_idx=[genes.index(g) for g in DE[cell_type+'_'+specific_type_dosage_cell.obs['condition'][0]+'_'+str(dosage)][:20]]
                        DE40_idx=[genes.index(g) for g in DE[cell_type+'_'+specific_type_dosage_cell.obs['condition'][0]+'_'+str(dosage)][:40]]

                        # calculate E-distance
                        E_DE20 = compute_e_distance(np.array(pred_cell_expr)[:, DE20_idx],
                                                    specific_type_dosage_cell.X.toarray()[:, DE20_idx])
                        E_distance_DE20.append(E_DE20)

                        E_DE40 = compute_e_distance(np.array(pred_cell_expr)[:, DE40_idx],
                                                    specific_type_dosage_cell.X.toarray()[:, DE40_idx])
                        E_distance_DE40.append(E_DE40)

                        real_array = specific_type_dosage_cell.X.toarray()
                        pred_array = np.array(pred_cell_expr)

                        EMD.append(np.mean([
                            wasserstein_distance(real_array[:, i], pred_array[:, i])
                            for i in range(real_array.shape[1])
                        ]))

                        # DE20
                        EMD_DE20.append(np.mean([
                            wasserstein_distance(real_array[:, i], pred_array[:, i])
                            for i in DE20_idx
                        ]))

                        # DE40
                        EMD_DE40.append(np.mean([
                            wasserstein_distance(real_array[:, i], pred_array[:, i])
                            for i in DE40_idx
                        ]))

                        RMSE_all.append(compute_rmse(real_array, pred_array))

                        RMSE_DE20.append(compute_rmse(real_array[:, DE20_idx], pred_array[:, DE20_idx]))

                        RMSE_DE40.append(compute_rmse(real_array[:, DE40_idx], pred_array[:, DE40_idx]))

        else:
            '''unseen drug covariate'''
            test_cells = pert_data.adata[pert_data.adata.obs['unlasting_split'] == 'test']
            test_smiles = list(test_cells.obs['SMILES'].unique())

            for mole in tqdm(test_smiles, desc="Unseen drug covariate conditions"):
                current_test_cell = test_cells[test_cells.obs['SMILES'] == mole]
                current_cell_type = list(current_test_cell.obs['cell_type'].unique())

                for cell_type in current_cell_type:
                    specific_type_cell = current_test_cell[current_test_cell.obs['cell_type'] == cell_type]

                    all_dosage = list(specific_type_cell.obs['dose_val'].unique())
                    cell_type_idx = pert_data.cell_type.index(cell_type)
                    for dosage in all_dosage:
                        pcc_delta_value = []

                        specific_type_dosage_cell = specific_type_cell[specific_type_cell.obs['dose_val'] == dosage]

                        if specific_type_dosage_cell.shape[0] < 10:
                            continue

                        key = args_test.pert_type + "_" + mole + "_" + cell_type + "_" + str(dosage)

                        pred_cell_expr = result[key]
                        pred_cell_expr=np.array(pred_cell_expr)

                        # recover
                        pred_cell_expr=pred_cell_expr*max
                        pred_cell_expr[pred_cell_expr < 0] = 0

                        control_type_cell = pert_data.adata[pert_data.adata.obs['cell_type'] == cell_type]
                        control_type_cell = control_type_cell[control_type_cell.obs['condition'] == 'ctrl']

                        # calculate E-distance
                        E = compute_e_distance(np.array(pred_cell_expr), specific_type_dosage_cell.X.toarray())
                        E_distance_all.append(E)


                        """
                        DE
                        """
                        genes=pert_data.adata.var.index.tolist()
                        DE=pert_data.adata.uns['lincs_DEGs']
                        DE20_idx=[genes.index(g) for g in DE[cell_type+'_'+specific_type_dosage_cell.obs['condition'][0]+'_'+str(dosage)][:20]]
                        DE40_idx=[genes.index(g) for g in DE[cell_type+'_'+specific_type_dosage_cell.obs['condition'][0]+'_'+str(dosage)][:40]]

                        # calculate E-distance
                        E_DE20 = compute_e_distance(np.array(pred_cell_expr)[:, DE20_idx],
                                                    specific_type_dosage_cell.X.toarray()[:, DE20_idx])
                        E_distance_DE20.append(E_DE20)

                        E_DE40 = compute_e_distance(np.array(pred_cell_expr)[:, DE40_idx],
                                                    specific_type_dosage_cell.X.toarray()[:, DE40_idx])
                        E_distance_DE40.append(E_DE40)

                        real_array = specific_type_dosage_cell.X.toarray()
                        pred_array = np.array(pred_cell_expr)

                        EMD.append(np.mean([
                            wasserstein_distance(real_array[:, i], pred_array[:, i])
                            for i in range(real_array.shape[1])
                        ]))

                        # DE20
                        EMD_DE20.append(np.mean([
                            wasserstein_distance(real_array[:, i], pred_array[:, i])
                            for i in DE20_idx
                        ]))

                        # DE40
                        EMD_DE40.append(np.mean([
                            wasserstein_distance(real_array[:, i], pred_array[:, i])
                            for i in DE40_idx
                        ]))

                        E = compute_e_distance(real_array[:, DE20_idx], pred_array[:, DE20_idx])
                        E_distance_DE20.append(E)

                        E = compute_e_distance(real_array[:, DE40_idx], pred_array[:, DE40_idx])
                        E_distance_DE40.append(E)

                        RMSE_all.append(compute_rmse(real_array, pred_array))

                        RMSE_DE20.append(compute_rmse(real_array[:, DE20_idx], pred_array[:, DE20_idx]))

                        RMSE_DE40.append(compute_rmse(real_array[:, DE40_idx], pred_array[:, DE40_idx]))

    else:
        for cond in tqdm(pert_data.test_cond, desc="Perturbation conditions"):
            current_test_cell = pert_data.adata[pert_data.adata.obs['knockout'] == cond]
            current_cell_type = list(current_test_cell.obs['cell_type'].unique())
            for cell_type in current_cell_type:
                specific_type_cell = current_test_cell[current_test_cell.obs['cell_type'] == cell_type]
                key = args_test.pert_type + "_" + cond + "_" + cell_type
                pred_cell_expr = result[key]
                pred_cell_expr = np.array(pred_cell_expr)
                pred_cell_expr = pred_cell_expr * max
                pred_cell_expr[pred_cell_expr < 0] = 0

                E=compute_e_distance(pred_cell_expr, specific_type_cell.X.toarray())
                E_distance_all.append(E)

                """
                Calcualte Differential Expression Gene
                """
                gene_list=pert_data.adata.var.index.tolist()
                if args_test.data_name=="adamson":
                    DE_gene=pert_data.adata.uns['rank_genes_groups_cov_all'][cell_type+'_'+cond+'+ctrl_1+1']
                elif args_test.data_name=="norman":
                    DE_gene = pert_data.adata.uns['rank_genes_groups_cov_all'][cell_type + '_' + cond + '_1+1']
                DE_gene=[de for de in DE_gene if de in gene_list]

                DE20_idx = [gene_list.index(g) for g in DE_gene[:20]]
                DE40_idx = [gene_list.index(g) for g in DE_gene[:40]]

                real_array = specific_type_cell.X.toarray()
                pred_array = np.array(pred_cell_expr)

                EMD.append(np.mean([
                    wasserstein_distance(real_array[:, i], pred_array[:, i])
                    for i in range(real_array.shape[1])
                ]))

                E=compute_e_distance(pred_array[:,DE20_idx],real_array[:,DE20_idx])
                E_distance_DE20.append(E)

                E=compute_e_distance(pred_array[:,DE40_idx],real_array[:,DE40_idx])
                E_distance_DE40.append(E)

                distances = [
                    wasserstein_distance(specific_type_cell.X.toarray()[:, i], np.array(pred_cell_expr)[:, i])
                    for i in DE20_idx
                ]
                EMD_DE20.append(np.mean(distances))

                distances = [
                    wasserstein_distance(specific_type_cell.X.toarray()[:, i], np.array(pred_cell_expr)[:, i])
                    for i in DE40_idx
                ]
                EMD_DE40.append(np.mean(distances))

                RMSE_all.append(compute_rmse(real_array, pred_array))

                RMSE_DE20.append(compute_rmse(real_array[:, DE20_idx], pred_array[:, DE20_idx]))

                RMSE_DE40.append(compute_rmse(real_array[:, DE40_idx], pred_array[:, DE40_idx]))

    print("Aver EMD of prediction under each condition: {:.4f} ± {:.4f}".format(
        np.mean(EMD), np.std(EMD)))
    print("Aver E-distance of prediction under each condition: {:.4f} ± {:.4f}".format(
        np.mean(E_distance_all), np.std(E_distance_all)))
    print("Aver RMSE of prediction under each condition: {:.4f} ± {:.4f}".format(
        np.mean(RMSE_all), np.std(RMSE_all)))

    print("Aver EMD of DE20 prediction under each condition: {:.4f} ± {:.4f}".format(
        np.mean(EMD_DE20), np.std(EMD_DE20)))
    print("Aver E-distance of DE20 prediction under each condition: {:.4f} ± {:.4f}".format(
        np.mean(E_distance_DE20), np.std(E_distance_DE20)))
    print("Aver RMSE of DE20 prediction under each condition: {:.4f} ± {:.4f}".format(
        np.mean(RMSE_DE20), np.std(RMSE_DE20)))

    print("Aver EMD of DE40 prediction under each condition: {:.4f} ± {:.4f}".format(
        np.mean(EMD_DE40), np.std(EMD_DE40)))
    print("Aver E-distance of DE40 prediction under each condition: {:.4f} ± {:.4f}".format(
        np.mean(E_distance_DE40), np.std(E_distance_DE40)))
    print("Aver RMSE of DE40 prediction under each condition: {:.4f} ± {:.4f}".format(
        np.mean(RMSE_DE40), np.std(RMSE_DE40)))