import os
import torch
import random
import argparse
import numpy as np
from datasets import *
from config import _C as cfg
from models import *
from trainer import Trainer
import utils

def arg_parse():
    parser = argparse.ArgumentParser(description='Training parameters')
    parser.add_argument('--cfg', required=True, help='config file', type=str)
    parser.add_argument('--init', type=str, help='init model from', default='')
    parser.add_argument('--exp-name', type=str, help='exp name')
    parser.add_argument('--seed', type=int, help='random seed', default=0)
    return parser.parse_args()


def main():
    # ---- setup training environment
    args = arg_parse()
    rng_seed = args.seed
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
    else:
        raise NotImplementedError

    # ---- setup config files
    cfg.merge_from_file(args.cfg)
    cfg.DATA_ROOT = utils.get_data_root()
    cfg.freeze()

    # ---- setup model
    model = dyn_model.Net()
    model.to(torch.device('cuda'))

    # ---- setup optimizer
    optim = torch.optim.Adam(
        model.parameters(),
        lr=cfg.SOLVER.BASE_LR,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )

    # ---- if resume experiments, use --init ${model_name}
    if args.init:
        print(f'loading pretrained model from {args.init}')
        cp = torch.load(args.init)
        model.load_state_dict(cp['model'], False)

    # ---- setup dataset in the last, and avoid non-deterministic in data shuffling order
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    train_set = eval(f'{cfg.DATASET_ABS}')(data_dir=os.path.join(cfg.DATA_ROOT, cfg.TRAIN_DIR), phase='train')
    val_set = eval(f'{cfg.DATASET_ABS}')(data_dir=os.path.join(cfg.DATA_ROOT, cfg.VAL_DIR), phase='val')
    kwargs = {'pin_memory': False, 'num_workers': 4}

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True, **kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False, **kwargs,
    )
    print(f'size: train {len(train_loader)} / test {len(val_loader)}')

    # ---- setup trainer
    kwargs = {'device': torch.device('cuda'),
              'model': model,
              'optim': optim,
              'train_loader': train_loader,
              'val_loader': val_loader,
              'exp_name': args.exp_name,
              'max_iters': cfg.SOLVER.MAX_ITERS}
    trainer = Trainer(**kwargs)

    trainer.train()

if __name__ == '__main__':
    main()
