import argparse
import os
import torch
import lmdb
import esm
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict
from unimol_tools import UniMolRepr
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

    if not os.path.exists('./data/test_split/enzyme_emb.lmdb'):
        df = pd.read_csv('./data/test_split/test.csv')
        df.drop_duplicates(subset=['Uniprot_ID'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        print(f"Sequence number: {len(df)}")
        device = torch.device('cuda')
        esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        esm_model.to(device)
        batch_converter = alphabet.get_batch_converter()
        esm_model.eval()
        lmdb_path = './data/test_split/enzyme_emb.lmdb'
        env = lmdb.open(lmdb_path, map_size=2199023255556)
        with env.begin(write=True) as txn:
            for j in tqdm(range(len(df))):
                enzyme = df['Uniprot_ID'][j]
                sequence = df['Sequence'][j]
                data = [(enzyme, sequence)]
                batch_labels, batch_strs, batch_tokens = batch_converter(data)
                batch_tokens = batch_tokens.to(device)
                with torch.no_grad():
                    results = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
                token_representations = results["representations"][33].squeeze(0)
                token_representations = token_representations.cpu()
                txn.put(enzyme.encode(), pickle.dumps(token_representations))
        env.close()

    if not os.path.exists('./data/test_split/reaction_emb.lmdb'):
        clf = UniMolRepr(data_type='molecule', remove_hs=False, use_gpu=True)
        df = pd.read_csv('./data/test_split/test.csv')
        reaction_list = list(set(df['reaction'].values.tolist()))
        smiles_list = [j for i in reaction_list for j in i.split('>>')]
        smiles_list = [j for i in smiles_list for j in i.split('.')]
        smiles_list = list(sorted(list(set(smiles_list))))
        itosmiles = set()
        for s in smiles_list:
            itosmiles.add(s)
        itosmiles = sorted(list(itosmiles))
        smilestoi = {itosmiles[i]:i for i in range(len(itosmiles))}
        with open('./data/test_split/unimol_smile_dict.pk', 'wb') as f:
            pickle.dump([itosmiles,smilestoi],f)

        batchsize = 64
        env = lmdb.open('./data/test_split/reaction_emb.lmdb', map_size=2199023255556)
        with env.begin(write=True) as txn:
            for i in tqdm(range(len(smiles_list)//batchsize + 1)):
                input_smiles = smiles_list[i*batchsize : min((i+1)*batchsize, len(smiles_list))]
                idx_list = [smilestoi[i] for i in input_smiles] 
                reprs = clf.get_repr(input_smiles, return_atomic_reprs=True)
                cls_repr_tensor = torch.tensor(reprs['cls_repr'])
                for j in range(len(reprs['atomic_reprs'])):
                    atom_repr_tensor = torch.tensor(reprs['atomic_reprs'][j])
                    result_tensor = torch.cat([cls_repr_tensor[j].unsqueeze(0), atom_repr_tensor], dim=0)
                    idx = idx_list[j]
                    txn.put(str(idx).encode(), pickle.dumps(result_tensor))
        env.close()

    for i in os.listdir(test_path):     
        if i[-3:] != 'csv':
            continue 
        logger.info(i)  
        test_df = pd.read_csv(os.path.join(test_path, i))
        test_df['reactant_id'] = test_df['reactant_id'].apply(lambda x:eval(x))
        test_df['product_id'] = test_df['product_id'].apply(lambda x:eval(x))
        evaluate(model, test_df, device, args, logger)

def evaluate(model, test_df, device, args, logger):
    ec2uni_dict = defaultdict(set)
    for ec, uid in zip(test_df['EC_id'].values.tolist(), test_df['Uniprot_ID'].values.tolist()):
        ec2uni_dict[ec].add(uid)

    model.eval()
    with torch.no_grad():
        all_product_embeddings = []
        all_reactant_embeddings = []
        all_enzyme_embeddings = []
        all_ec = []
        all_product_id = []
        all_reaction_id = []
        product_dict = {}
        enzyme_dict = {}

        with open('./data/test_split/unimol_smile_dict.pk', 'rb') as f:
            itosmiles, smilestoi = pickle.load(f)

        test_df['reactant_id'] = test_df['reactant'].apply(lambda x: [smilestoi[i] for i in x.split('.')])
        test_df['product_id'] = test_df['product'].apply(lambda x: [smilestoi[i] for i in x.split('.')])
        
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

            for i in range(enzyme_emb.size(0)):
                if str(product_id[i]) not in product_dict.keys():
                    product_dict[str(product_id[i])] = product_emb[i]
                if uni_id[i] not in enzyme_dict.keys():
                    enzyme_dict[uni_id[i]] = enzyme_emb[i]

    product_keys = list(product_dict.keys())
    product_tensor = torch.stack(list(product_dict.values()), dim=0).cpu()
    enzyme_keys = list(enzyme_dict.keys())
    enzyme_tensor = torch.stack(list(enzyme_dict.values()), dim=0).cpu()    

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
    product_h1 = float(np.mean(product_all_ranks<=1))
    print('Match_product_mrr: %.4f Match_product_h1: %.4f' % (product_mean_reciprocal_rank, product_h1))
    logger.info('Match_product_mrr: %.4f Match_product_h1: %.4f' % (product_mean_reciprocal_rank, product_h1))

    fuse_embeddings = all_reactant_embeddings - all_product_embeddings
    dist = torch.cdist(fuse_embeddings.double(), -enzyme_tensor.double(), p=2)
    sorted_indices = torch.argsort(dist, dim=1)

    all_enzyme_ap = []
    enzyme_keys = {v:idx for idx,v in enumerate(enzyme_keys)}
    searched_reaction = set()
    for idx, data in tqdm(enumerate(zip(all_ec, all_reaction_id)), desc='enzyme mrr:'):
        ec, rid = data
        if rid.item() not in searched_reaction:     # 一个化学反应只计算一次（fuse_embedding相同）
            searched_reaction.add(rid.item())
        else:
            continue
        match_uni = ec2uni_dict[ec.item()]
        ground_truth = torch.tensor([enzyme_keys[i] for i in match_uni])
        rankings = np.sort(np.array((sorted_indices[idx] == ground_truth.unsqueeze(1)).nonzero()[:, 1] + 1))
        ap = (np.arange(1, len(match_uni)+1) / rankings).sum() / len(match_uni)
        all_enzyme_ap.append(ap)
    enzyme_mean_average_precision = float(np.mean(all_enzyme_ap))
    print('enzyme_mean_average_precision: %.4f' % (enzyme_mean_average_precision))
    logger.info('enzyme_mean_average_precision: %.4f' % (enzyme_mean_average_precision))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='setting')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--checkpoint', default='./ckpt/checkpoint.pt')
    parser.add_argument('--mol_env_path', default='./data/test_split/reaction_emb.lmdb')
    parser.add_argument('--esm_env_path', default='./data/test_split/enzyme_emb.lmdb')
    parser.add_argument('--test_path', default='./data/test_split')
    parser.add_argument('--unimol_dict', default='./data/test_split/unimol_smile_dict.pk')
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()
    main(args)
