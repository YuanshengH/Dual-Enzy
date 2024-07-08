import os
import argparse
import math
import torch
import wandb
import pickle
import pandas as pd
import numpy as np
import torch.distributed.nn 
import torch.distributed as dist
from tqdm import tqdm
from loguru import logger
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers.optimization import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from dataset import EnzymeDataset, collate_fn
from utils.build_utils import same_seed, Feature_Bank
from model.Enzymatic_reaction import EnzymaticModel

def main(args):
    # ddp
    os.environ["WANDB_API_KEY"] = args.wandb_api_key
    os.environ["WANDB_MODE"] = "offline"
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device(f'cuda:{local_rank}')

    print(args)
    same_seed(args.seed)
    df = pd.read_csv(args.data_path)
    rxntoi = {r:idx for idx, r in enumerate(sorted(list(set(df['reaction'].values.tolist()))))}
    ectoi = {ec:idx for idx, ec in enumerate(sorted(list(set(df['EC'].values.tolist()))))}
    with open(args.unimol_dict, 'rb') as f:
        itosmiles, smilestoi = pickle.load(f)
    df['reaction_id'] = df['reaction'].apply(lambda x:rxntoi[x])
    df['EC_id'] = df['EC'].apply(lambda x:ectoi[x])
    df['reactant_id'] = df['reactant'].apply(lambda x: [smilestoi[i] for i in x.split('.')])
    df['product_id'] = df['product'].apply(lambda x: [smilestoi[i] for i in x.split('.')])
    train_df,valid_df = train_test_split(df, train_size=0.8, random_state=args.seed)
    valid_df, test_df = train_test_split(valid_df, train_size=0.5, random_state=args.seed)

    train_set = EnzymeDataset(reactant_id=train_df['reactant_id'].values.tolist(), product_id=train_df['product_id'].values.tolist(),
                              reaction_id=train_df['reaction_id'].values.tolist(), ec=train_df['EC_id'].values.tolist(),
                                uni_id=train_df['Uniprot_ID'].values.tolist(), mol_env_path=args.mol_env_path, esm_env_path=args.esm_env_path)
    sampler = DistributedSampler(train_set)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=4, pin_memory=True, sampler=sampler)

    valid_set = EnzymeDataset(reactant_id=valid_df['reactant_id'].values.tolist(), product_id=valid_df['product_id'].values.tolist(),
                              reaction_id=valid_df['reaction_id'].values.tolist(), ec=valid_df['EC_id'].values.tolist(),
                                uni_id=valid_df['Uniprot_ID'].values.tolist(), mol_env_path=args.mol_env_path, esm_env_path=args.esm_env_path)
    valid_sampler = DistributedSampler(valid_set)
    valid_loader = DataLoader(dataset=valid_set, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=4, pin_memory=True, sampler=valid_sampler)

    model = EnzymaticModel(num_layers=args.num_layer, hidden_dim=args.hidden_dim, out_dim=args.output_dim)
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    criterion = torch.nn.CrossEntropyLoss()
    num_class = len(list(set(df['EC_id'].values.tolist())))
    feature_bank = Feature_Bank(num_class=num_class, m=args.m)
    
    # scheduler
    total_steps = (len(train_loader) * args.epoch)
    warmup_steps = math.ceil(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    start_epoch = 0
    feats_array = np.load(args.feat_path)
    lb_array = np.load(args.lb_path)
    feature_bank.init_feats(feats=feats_array, lbs=lb_array)
    dist.barrier()

    # wandb setting
    if local_rank == 0:
        wandb_name = 'train'
        logger.add('./log/training_log_{time}.log')
        wandb.init(config=args, project='Enzymatic reaction', name=wandb_name)
        wandb.define_metric("mystep")
        wandb.define_metric("total_train_loss", step_metric="mystep")
        wandb.define_metric("reaction_loss", step_metric="mystep")
        wandb.define_metric("rxn_loss", step_metric="mystep")
        wandb.define_metric("ec_loss", step_metric="mystep")
        wandb.define_metric("prototype_loss", step_metric="mystep")
        wandb.define_metric("total_valid_loss", step_metric="mystep")
        wandb.define_metric("valid_reaction_loss", step_metric="mystep")
        wandb.define_metric("valid_rxn_loss", step_metric="mystep")
        wandb.define_metric("valid_ec_loss", step_metric="mystep")
        wandb.define_metric("valid_prototype_loss", step_metric="mystep")
        wandb.define_metric("val_product_mrr", step_metric="mystep")
        wandb.define_metric("val_enzyme_mrr", step_metric="mystep")
        wandb.define_metric("test_product_mrr", step_metric="mystep")
        wandb.define_metric("test_enzyme_mrr", step_metric="mystep")
        wandb.watch(models=model)

    train_global_step = 0
    valid_global_step = 0
    valid_reaction_losses = AverageMeter('Valid Reaction Loss', ':.4e')
    valid_rxn_losses = AverageMeter('Valid RXN Loss', ':.4e')
    valid_ec_losses = AverageMeter('Valid EC Loss', ':.4e')
    valid_prototype_losses = AverageMeter('Vild Product Loss', ':.4e')

    for i in range(1, args.epoch+1):
        i += start_epoch
        train_loader.sampler.set_epoch(i)
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {i}:"):
            train_global_step += 1
            r_embedding, p_embedding, esm_embedding, r_padding_mask, p_padding_mask, esm_padding_mask, reaction_id, ec = batch
            r_embedding, p_embedding, esm_embedding, r_padding_mask, p_padding_mask, esm_padding_mask = r_embedding.to(device), p_embedding.to(device), esm_embedding.to(device),\
                                                                        r_padding_mask.to(device), p_padding_mask.to(device), esm_padding_mask.to(device)
            reaction_id, ec = reaction_id.to(device), ec.to(device)

            reactant_emb, enzyme_emb, product_emb = model(esm_emb=esm_embedding, reactant=r_embedding, product=p_embedding, 
                                                        esm_padding_mask=esm_padding_mask, reactant_padding_mask=r_padding_mask, product_padding_mask=p_padding_mask)
            
            all_reactant_emb = torch.cat(torch.distributed.nn.all_gather(reactant_emb), dim=0)
            all_product_emb = torch.cat(torch.distributed.nn.all_gather(product_emb), dim=0)
            all_enzyme_emb = torch.cat(torch.distributed.nn.all_gather(enzyme_emb), dim=0)
            all_reaction_id = torch.cat(torch.distributed.nn.all_gather(reaction_id), dim=0)
            all_ec_id = torch.cat(torch.distributed.nn.all_gather(ec), dim=0)

            rxn_loss, ec_loss, reaction_loss = ReactionLoss(all_reactant_emb, all_enzyme_emb, all_product_emb, rxn_label=all_reaction_id, ec_label=all_ec_id, margin1=args.margin1, margin2=args.margin2)
            
            # EC prototype updata
            feats = torch.Tensor(feature_bank.get_feats()).to(device)
            logit = torch.mm(enzyme_emb, feats.transpose(1,0)) / args.T
            prototype_loss = criterion(logit, ec.long())
            feature_bank.update_a_batch(batch_feats=enzyme_emb.detach().cpu().numpy(), lbs=ec.detach().cpu().numpy())

            loss = reaction_loss + rxn_loss + ec_loss + prototype_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if local_rank == 0:
                wandb.log({'mystep':train_global_step, 'total_train_loss':loss.item(), 'reaction_loss':reaction_loss.item(), 
                           'rxn_loss':rxn_loss.item(), 'ec_loss':ec_loss.item(), 'prototype_loss': prototype_loss.item()})
            
        if local_rank == 0:
            logger.info(f"Train Loss:{loss.item()}")
            save_checkpoint(model, optimizer, epoch=i, checkpoint_path=args.checkpoint, feature_bank=feature_bank)

        torch.distributed.barrier()
        valid_global_step += 1
        valid_reaction_losses.reset()
        valid_rxn_losses.reset()
        valid_ec_losses.reset()
        valid_prototype_losses.reset()
        model.eval()
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f"Epoch {i}:"):
                r_embedding, p_embedding, esm_embedding, r_padding_mask, p_padding_mask, esm_padding_mask, reaction_id, ec = batch
                r_embedding, p_embedding, esm_embedding, r_padding_mask, p_padding_mask, esm_padding_mask = r_embedding.to(device), p_embedding.to(device), esm_embedding.to(device),\
                                                                            r_padding_mask.to(device), p_padding_mask.to(device), esm_padding_mask.to(device)
                reaction_id, ec = reaction_id.to(device), ec.to(device)
                reactant_emb, enzyme_emb, product_emb = model(esm_emb=esm_embedding, reactant=r_embedding, product=p_embedding, 
                                                            esm_padding_mask=esm_padding_mask, reactant_padding_mask=r_padding_mask, product_padding_mask=p_padding_mask)

                all_reactant_emb = torch.cat(torch.distributed.nn.all_gather(reactant_emb), dim=0)
                all_product_emb = torch.cat(torch.distributed.nn.all_gather(product_emb), dim=0)
                all_enzyme_emb = torch.cat(torch.distributed.nn.all_gather(enzyme_emb), dim=0)
                all_reaction_id = torch.cat(torch.distributed.nn.all_gather(reaction_id), dim=0)
                all_ec_id = torch.cat(torch.distributed.nn.all_gather(ec), dim=0)

                valid_rxn_loss, valid_ec_loss, valid_reaction_loss = ReactionLoss(all_reactant_emb, all_enzyme_emb, all_product_emb, rxn_label=all_reaction_id, ec_label=all_ec_id, margin1=args.margin1, margin2=args.margin2)

                # EC prototype updata --> enzyme_emb
                feats = torch.Tensor(feature_bank.get_feats()).to(device)
                logit = torch.mm(enzyme_emb, feats.transpose(1,0)) / args.T
                valid_prototype_loss = criterion(logit, ec.long())

                valid_reaction_losses.update(valid_reaction_loss.item(), ec.size(0))
                valid_rxn_losses.update(valid_rxn_loss.item(), ec.size(0))
                valid_ec_losses.update(valid_ec_loss.item(), ec.size(0))
                valid_prototype_losses.update(valid_prototype_loss.item(), ec.size(0))

        if local_rank == 0:
            valid_total_loss = valid_reaction_losses.avg + valid_rxn_losses.avg + valid_ec_losses.avg + valid_prototype_losses.avg
            wandb.log({'mystep':valid_global_step, 'total_valid_loss':valid_total_loss, 'valid_reaction_loss':valid_reaction_losses.avg, \
                    'valid_rxn_loss':valid_rxn_losses.avg, 'valid_ec_loss':valid_ec_losses.avg, 'valid_prototype_loss':valid_prototype_losses.avg})
            logger.info(f"valid Loss:{valid_total_loss}")

def ReactionLoss(reactant_emb, enzyme_emb, product_emb, rxn_label, ec_label, margin1, margin2):
    # [reactant+enzyme] <--> [product] large margin
    fuse_emb1 = reactant_emb + enzyme_emb
    rxn_label = rxn_label.contiguous().view(-1, 1)
    rxn_mask = torch.eq(rxn_label, rxn_label.T).int().cuda()
    pos_num = rxn_mask.sum(dim=1)
    neg_num = (1-rxn_mask).sum(dim=1)
    dist = torch.cdist(fuse_emb1.double(), product_emb.double(), p=2)
    pos = rxn_mask * dist
    pos = pos.sum(dim=1) / pos_num
    neg = (1 - rxn_mask) * dist + rxn_mask * margin1
    neg = torch.relu(margin1 - neg)
    neg = neg.sum(dim=1) / neg_num
    rxn_loss = pos.mean() + neg.mean()

    # [product_emb - reactant_emb] <--> [enzyme] small margin
    fuse_emb2 = product_emb - reactant_emb
    ec_label = ec_label.contiguous().view(-1, 1)
    ec_mask = torch.eq(ec_label, ec_label.T).int().cuda()
    pos_num = ec_mask.sum(dim=1)
    neg_num = (1-ec_mask).sum(dim=1)
    dist = torch.cdist(fuse_emb2.double(), enzyme_emb.double(), p=2)
    pos = ec_mask * dist
    pos = pos.sum(dim=1) / pos_num
    neg = (1 - ec_mask) * dist + ec_mask * margin2
    neg = torch.relu(margin2 - neg)
    neg = neg.sum(dim=1) / neg_num
    ec_loss = pos.mean() + neg.mean()

    # [reactant] <--> [product] small margin
    reaction_dist = torch.linalg.norm((reactant_emb-product_emb), dim=-1)   
    reaction_loss = torch.relu(margin2-reaction_dist).mean()
    return rxn_loss, ec_loss, reaction_loss
    
def save_checkpoint(model, optimizer, epoch, checkpoint_path, feature_bank):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'feat_array': feature_bank.feats,
    }
    file_name = os.path.join(checkpoint_path, f'checkpoint_{epoch}.pt')
    torch.save(checkpoint, file_name)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='setting')
    parser.add_argument('--m', default=0.9, type=float, help="Feature bank updata rate")
    parser.add_argument('--T', type=float, default=0.07, help="temperature")
    parser.add_argument('--margin1', default=12, type=float, help="contrastive loss margin")
    parser.add_argument('--margin2', default=3, type=float, help="contrastive loss margin")
    parser.add_argument('--output_dim', default=512, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_layer', type=int, default=1)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--wandb_api_key', type=str, default="")
    parser.add_argument('--checkpoint', default='./ckpt')
    parser.add_argument('--data_path', default='./data/rhea_processed_data.csv')
    parser.add_argument('--unimol_dict', default='./data/train_data/unimol_smile_dict.pk')
    parser.add_argument('--mol_env_path', default='./data/train_data/unimol.lmdb')
    parser.add_argument('--esm_env_path', default='./data/train_data/esm_rhea.lmdb')
    parser.add_argument('--feat_path', default='./data/feat.npy', help="Feature bank initialized features")
    parser.add_argument('--lb_path', default='./data/lb.npy', help="Feature bank initialized label")
    args = parser.parse_args()
    main(args)
