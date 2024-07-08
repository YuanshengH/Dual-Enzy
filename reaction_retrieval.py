import argparse
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from loguru import logger
from dataset import EvalDataset, collate_fn_eval
from utils.build_utils import same_seed
from model.Enzymatic_reaction import EnzymaticModel

def main(args):
    print(args)
    device = torch.device('cuda')
    same_seed(42)

    model = EnzymaticModel()
    model = model.to(device)
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(args.checkpoint, map_location='cuda:0')['model_state_dict'].items()})

    test_path = args.test_path
    logger.add('./log/test_mrr_map.log')
    logger.info(args.checkpoint)
    data_df = pd.read_csv('./data/evaluate/test_split/test.csv')
    data_df['reactant_id'] = data_df['reactant_id'].apply(lambda x:eval(x))
    data_df['product_id'] = data_df['product_id'].apply(lambda x:eval(x))

    model.eval()
    with torch.no_grad():
        eval_dataset = EvalDataset(reactant_id=data_df['reactant_id'].values.tolist(), product_id=data_df['product_id'].values.tolist(),
                                     reaction_id=data_df['reaction_id'].values.tolist(), ec=data_df['EC_id'].values.tolist(),
                                      uni_id=data_df['Uniprot_ID'].values.tolist(), mol_env_path=args.mol_env_path, esm_env_path=args.esm_env_path)
        eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=args.batch_size, collate_fn=collate_fn_eval, num_workers=8, shuffle=False)

        ec2uni_dict = {}
        for ec,uid in zip(data_df['EC_id'].values.tolist(), data_df['Uniprot_ID'].values.tolist()):
            if ec not in ec2uni_dict.keys():
                ec2uni_dict[ec] = set()
                ec2uni_dict[ec].add(uid)
            else:
                ec2uni_dict[ec].add(uid)

        product_dict = {}
        enzyme_dict = {}

        for batch in tqdm(eval_dataloader, desc='eval'):
            r_embedding, p_embedding, esm_embedding, r_padding_mask, p_padding_mask, esm_padding_mask, reaction_id, ec, reactant_id, product_id, uni_id = batch
            r_embedding, p_embedding, esm_embedding, r_padding_mask, p_padding_mask, esm_padding_mask = r_embedding.to(device), p_embedding.to(device), esm_embedding.to(device),\
                                                                        r_padding_mask.to(device), p_padding_mask.to(device), esm_padding_mask.to(device)
            reaction_id, ec = reaction_id.to(device), ec.to(device)
            reactant_emb, enzyme_emb, product_emb = model(esm_emb=esm_embedding, reactant=r_embedding, product=p_embedding, 
                                                        esm_padding_mask=esm_padding_mask, reactant_padding_mask=r_padding_mask, product_padding_mask=p_padding_mask)

            for i in range(enzyme_emb.size(0)):
                if str(product_id[i]) not in product_dict.keys():
                    product_dict[str(product_id[i])] = product_emb[i]
                if uni_id[i] not in enzyme_dict.keys():
                    enzyme_dict[uni_id[i]] = enzyme_emb[i]

        product_keys = list(product_dict.keys())
        product_tensor = torch.stack(list(product_dict.values()), dim=0).cpu()
        enzyme_keys = list(enzyme_dict.keys())
        enzyme_tensor = torch.stack(list(enzyme_dict.values()), dim=0).cpu()

    for i in os.listdir(test_path):     
        if i[-3:] != 'csv':
            continue 
        logger.info(i)  
        test_df = pd.read_csv(os.path.join(test_path, i))
        test_df['reactant_id'] = test_df['reactant_id'].apply(lambda x:eval(x))
        test_df['product_id'] = test_df['product_id'].apply(lambda x:eval(x))
        evaluate(model, test_df, device, 'test', args, logger, product_keys, product_tensor, enzyme_keys, enzyme_tensor, ec2uni_dict)

def evaluate(model, test_df, device, mode, args, logger, product_keys, product_tensor, enzyme_keys, enzyme_tensor, ec2uni_dict):
    model.eval()
    with torch.no_grad():
        all_product_embeddings = []
        all_reactant_embeddings = []
        all_enzyme_embeddings = []
        all_ec = []
        all_product_id = []
        all_reaction_id = []
        test_dataset = EvalDataset(reactant_id=test_df['reactant_id'].values.tolist(), product_id=test_df['product_id'].values.tolist(),
                                     reaction_id=test_df['reaction_id'].values.tolist(), ec=test_df['EC_id'].values.tolist(),
                                      uni_id=test_df['Uniprot_ID'].values.tolist(), mol_env_path=args.mol_env_path, esm_env_path=args.esm_env_path)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, collate_fn=collate_fn_eval, num_workers=8, shuffle=False)

        for batch in tqdm(test_dataloader, desc='eval'):
            r_embedding, p_embedding, esm_embedding, r_padding_mask, p_padding_mask, esm_padding_mask, reaction_id, ec, reactant_id, product_id, uni_id = batch
            r_embedding, p_embedding, esm_embedding, r_padding_mask, p_padding_mask, esm_padding_mask = r_embedding.to(device), p_embedding.to(device), esm_embedding.to(device),\
                                                                        r_padding_mask.to(device), p_padding_mask.to(device), esm_padding_mask.to(device)
            reaction_id, ec = reaction_id.to(device), ec.to(device)
            reactant_emb, enzyme_emb, product_emb = model(esm_emb=esm_embedding, reactant=r_embedding, product=p_embedding, 
                                                        esm_padding_mask=esm_padding_mask, reactant_padding_mask=r_padding_mask, product_padding_mask=p_padding_mask)

            all_ec.append(ec.detach().cpu())
            all_reaction_id.append(reaction_id.detach().cpu())
            all_reactant_embeddings.append(reactant_emb.detach().cpu())
            all_enzyme_embeddings.append(enzyme_emb.detach().cpu())
            all_product_embeddings.append(product_emb.detach().cpu())

            for p_id in product_id:
                all_product_id.append(str(p_id))

    all_ec = torch.cat(all_ec, dim=0)
    all_reaction_id = torch.cat(all_reaction_id, dim=0)
    all_product_embeddings = torch.cat(all_product_embeddings, dim=0)
    all_reactant_embeddings = torch.cat(all_reactant_embeddings, dim=0)
    all_enzyme_embeddings = torch.cat(all_enzyme_embeddings, dim=0)

    fuse_embeddings = all_reactant_embeddings + all_enzyme_embeddings
    dist = torch.cdist(fuse_embeddings.double(), product_tensor.double(), p=2)
    ground_truth = torch.tensor([product_keys.index(i) for i in all_product_id]).unsqueeze(1)
    sorted_indices = torch.argsort(dist, dim=1)
    product_all_ranks = np.array(((sorted_indices == ground_truth).nonzero()[:, 1] + 1).tolist())

    product_mean_reciprocal_rank = float(np.mean(1 / product_all_ranks))
    product_mean_rank = float(np.mean(product_all_ranks))
    product_h1 = float(np.mean(product_all_ranks<=1))
    print('Match product:  %s  product_mr: %.4f product_mrr: %.4f product_h1: %.4f' % (mode, product_mean_rank, product_mean_reciprocal_rank,product_h1))
    logger.info('Match product:  %s  product_mr: %.4f product_mrr: %.4f product_h1: %.4f' % (mode, product_mean_rank, product_mean_reciprocal_rank,product_h1))

    fuse_embeddings = all_reactant_embeddings - all_product_embeddings
    dist = torch.cdist(fuse_embeddings.double(), -enzyme_tensor.double(), p=2)
    sorted_indices = torch.argsort(dist, dim=1)

    all_enzyme_ap = []
    enzyme_keys = {v:idx for idx,v in enumerate(enzyme_keys)}
    searched_reaction = set()
    for idx, data in tqdm(enumerate(zip(all_ec, all_reaction_id)), desc='enzyme mrr:'):
        ec, rid = data
        if rid.item() not in searched_reaction:
            searched_reaction.add(rid.item())
        else:
            continue
        match_uni = ec2uni_dict[ec.item()]
        ground_truth = torch.tensor([enzyme_keys[i] for i in match_uni])
        rankings = np.sort(np.array((sorted_indices[idx] == ground_truth.unsqueeze(1)).nonzero()[:, 1] + 1))
        ap = (np.arange(1, len(match_uni)+1) / rankings).sum() / len(match_uni)
        all_enzyme_ap.append(ap)
    enzyme_mean_average_precision = float(np.mean(all_enzyme_ap))
    print('Match enzyme:  %s  enzyme_mean_average_precision: %.4f' % (mode, enzyme_mean_average_precision))
    logger.info('Match enzyme:  %s  enzyme_mean_average_precision: %.4f' % (mode, enzyme_mean_average_precision))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='setting')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--checkpoint', default='./ckpt/checkpoint.pt')
    parser.add_argument('--mol_env_path', default='./data/train_data/unimol.lmdb')
    parser.add_argument('--esm_env_path', default='./data/train_data/esm_rhea.lmdb')
    parser.add_argument('--test_path', default='./data/evaluate/test_split')
    parser.add_argument('--unimol_dict', default='./data/train_data/unimol_smile_dict.pk')
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()
    main(args)
