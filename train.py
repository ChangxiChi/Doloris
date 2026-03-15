
import io
import os
import socket
import sys

import torch as th
import torch.distributed as dist
import argparse
from datetime import datetime
import dist_util, logger
from model import SourceModel,TargetModel

from resample import create_named_schedule_sampler
from respace import SpacedDiffusion,space_timesteps

from train_util import TrainLoop, plot_loss
import diffusion
import yaml

def create_gaussian_diffusion(
        *,
        steps=1000,
        learn_sigma=False,
        sigma_small=False,
        noise_schedule="linear",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        timestep_respacing="ddim50",
):
    print('diffusion num of steps = ', steps)
    betas = diffusion.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = diffusion.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = diffusion.LossType.RESCALED_MSE
    else:
        loss_type = diffusion.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            diffusion.ModelMeanType.EPSILON if not predict_xstart else diffusion.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                diffusion.ModelVarType.FIXED_LARGE
                if not sigma_small
                else diffusion.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else diffusion.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def create_model_and_diffusion(
    gene_num,
    GRN,
    init_gene_emb,
    hidden_dim=1024,
    time_pos_dim=1024,
    gene_wise_embed_dim=512,
    cell_type_num=1,
    time_embed_dim=1024,
    cell_type_embed_dim=1024,
    data_name="adamson",
    gene_init_dim=256,
    mole_dim=512,
    pert_type="gene",
    learn_sigma=False,
    diffusion_steps=1000,
    noise_schedule="linear",
    timestep_respacing="ddim50",
    use_kl=False,
    predict_xstart=True,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
):
    model = TargetModel(
        gene_num=gene_num,
        GRN=GRN,
        gene_dim=gene_init_dim,
        init_gene_emb=init_gene_emb,
        hidden_dim=hidden_dim,
        time_pos_dim=time_pos_dim,
        gene_wise_embed_dim=gene_wise_embed_dim,
        cell_type_num=cell_type_num,
        time_embed_dim=time_embed_dim,
        cell_type_embed_dim=cell_type_embed_dim,
        data_name=data_name,
        mole_dim=mole_dim,
        pert_type=pert_type,
    )

    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion


def run_training(data,cell_type_num,init_gene_emb,GRN,args):
    logger.configure(dir=args.logger_path)
    logger.log("*********creating model and diffusion**********")
    model, diffusion = create_model_and_diffusion(
        GRN=GRN,
        init_gene_emb=init_gene_emb,
        cell_type_num=cell_type_num,
        pert_type=args.pert_type,
        data_name=args.data_name,
        gene_num=args.gene_num,
        time_pos_dim=args.time_pos_dim,
        gene_wise_embed_dim=args.gene_wise_embed_dim,
        time_embed_dim=args.time_embed_dim,
        cell_type_embed_dim=args.cell_type_embed_dim,
        gene_init_dim=args.gene_init_dim,
        mole_dim=args.mole_dim,
        learn_sigma=args.learn_sigma,
        diffusion_steps=args.diffusion_steps,
        timestep_respacing=args.timestep_respacing,
        use_kl=args.use_kl,
        predict_xstart=args.predict_xstart,
        rescale_timesteps=args.rescale_timesteps,
        rescale_learned_sigmas=args.rescale_learned_sigmas,
    )

    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    # logger.log(f'with gpu {dist_util.dev()}')
    start_time = datetime.now()
    logger.log(f'**********training started at {start_time} **********')
    train_ = TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        pert_type=args.pert_type,
        source_model=args.source_model,
    )
    train_.run_loop()

    end_time = datetime.now()

    during_time = (end_time - start_time).seconds / 60

    logger.log(f'start time: {start_time} end_time: {end_time} time:{during_time} min')

    return train_.loss_list



def parse_args():
    # Initialize argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/sciplex3/sciplex3_test.yaml")

    # Add arguments
    parser.add_argument('--gene_num', type=int, default=2000, help='Number of genes')
    parser.add_argument('--data_name', type=str, default="sciplex3", help='Name of dataset')
    parser.add_argument('--pert_type', type=str, default="molecular", help='Perturbation type')

    parser.add_argument('--time_pos_dim', type=int, default=1024, help='Dimensionality of time position embedding')
    parser.add_argument('--gene_wise_embed_dim', type=int, default=256, help='Dimensionality of global gene features')
    parser.add_argument('--time_embed_dim', type=int, default=256, help='Time embedding dimension')
    parser.add_argument('--cell_type_embed_dim', type=int, default=256, help='Cell type embedding dimension')
    parser.add_argument('--load_trained_source_model', action='store_true',default=False,help='Whether to load a trained source model')
    parser.add_argument('--source_trainable', action='store_true',default=True, help='Whether the source model is trainable in target model')
    parser.add_argument('--gene_init_dim', type=int, default=32, help='Initial gene embedding dimension')
    parser.add_argument('--mole_dim', type=int, default=512, help='Molecular feature dimension extracted by Uni-Mol')
    parser.add_argument('--learn_sigma', action='store_true', default=False, help='Whether to learn noise sigma in diffusion')
    parser.add_argument('--diffusion_steps', type=int, default=500, help='Number of diffusion steps')
    parser.add_argument('--timestep_respacing', type=str, default='ddim50', help='Timestep respacing strategy')
    parser.add_argument('--use_kl', action='store_true', default=False, help='Whether to use KL divergence loss')
    parser.add_argument('--predict_xstart', action='store_true', default=True, help='Whether to predict x_0 directly')
    parser.add_argument('--rescale_timesteps', action='store_true', default=False, help='Whether to rescale timesteps')
    parser.add_argument('--rescale_learned_sigmas', action='store_true', default=False, help='Whether to rescale learned sigmas')
    parser.add_argument('--source_model', action='store_true', default=True, help='is a source model or not')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--fp16_scale_growth', type=float, default=1e-3, help='')
    parser.add_argument('--use_fp16', action='store_true',default=False, help='')
    parser.add_argument('--microbatch', type=int,default=-1, help='')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (default: 0.0)')
    parser.add_argument('--lr_anneal_steps', type=int, default=int(1e5),help='Number of steps for learning rate annealing (default: 100000)')
    parser.add_argument('--ema_rate', type=float, default=0.9999,help='Exponential moving average rate (default: 0.9999)')
    parser.add_argument('--log_interval', type=int, default=int(1e4), help='Logging interval (default: 10000)')
    parser.add_argument('--save_interval', type=int, default=int(1e4), help='Model saving interval (default: 10000)')
    parser.add_argument('--schedule_sampler',type=str, default='uniform', help='')
    parser.add_argument('--resume_checkpoint', type=str, default='')
    parser.add_argument('--threshold', type=float, default=0.4, help='')
    parser.add_argument('--threshold_co', type=float, default=0.4, help='')
    parser.add_argument('--output_dim',type=int,default=1)
    # Parse arguments

    # Detect if running inside Jupyter (ipykernel)
    if any("ipykernel" in arg for arg in sys.argv):
        args = parser.parse_args([])   # ignore Jupyter's extra args
    else:
        args = parser.parse_args()

    if args.config is not None:

        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)

        if "args" in cfg:
                    for k, v in cfg["args"].items():
                        if hasattr(args, k):
                            setattr(args, k, v)


    return args