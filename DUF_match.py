import os
import torch 
import pickle   
import argparse                                           
import pandas as pd
import numpy as np
from loguru import logger
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from dataset import SubstrateDataset, collate_substrate
from model.Enzymatic_reaction import EnzymaticModel
from utils.build_utils import same_seed

def main(args):
    same_seed(args.seed)
    device = torch.device('cuda')
    df = pd.read_csv('./data/DUF_dataset/DUF_data.csv')
    rxntoi = {r:idx for idx, r in enumerate(sorted(list(set(df['reaction'].values.tolist()))))}
    with open('./data/DUF_dataset/DUF_reaction_dict.pk', 'rb') as f:
        itosmiles, smilestoi = pickle.load(f)
    df['reaction_id'] = df['reaction'].apply(lambda x:rxntoi[x])
    df['reactant'] = df['reaction'].apply(lambda x:x.split('>>')[0])
    df['product'] = df['reaction'].apply(lambda x:x.split('>>')[-1])
    df['reactant_id'] = df['reactant'].apply(lambda x: [smilestoi[i] for i in x.split('.')])
    df['product_id'] = df['product'].apply(lambda x: [smilestoi[i] for i in x.split('.')])
    df['seq_id'] = df['seq_id'].apply(lambda x: str(x))

    if args.split_method == 'random':
        train_df,test_df = train_test_split(df, train_size=0.7, random_state=42)
        valid_df, test_df = train_test_split(test_df, train_size=(1/3), random_state=42)

    elif args.split_method == 'enzyme':
        enzyme_list = df.seq_id.unique().tolist()
        np.random.shuffle(enzyme_list)
        test_size = int(len(enzyme_list)*0.3)
        test_enzyme = enzyme_list[:test_size]
        train_enzyme = enzyme_list[test_size:]
        train_df = df[df['seq_id'].isin(train_enzyme)]
        test_df = df[df['seq_id'].isin(test_enzyme)]
        valid_df, test_df = train_test_split(test_df, train_size=(1/3), random_state=42)

    elif args.split_method == 'substrate':
        substrates_list = df.reactant.unique().tolist()
        np.random.shuffle(substrates_list)
        test_size = int(len(substrates_list)*0.3)
        test_substrates = substrates_list[:test_size]
        train_substrates = substrates_list[test_size:]
        train_df = df[df['reactant'].isin(train_substrates)]
        test_df = df[df['reactant'].isin(test_substrates)]
        valid_df, test_df = train_test_split(test_df, train_size=(1/3), random_state=42)

    else:
        raise ValueError("Invalid split_method. Choose from 'random', 'enzyme', or 'substrate'.")

    valid_dataset = SubstrateDataset(reactant_id=valid_df['reactant_id'].values.tolist(), product_id=valid_df['product_id'].values.tolist(),
                                reaction_id=valid_df['reaction_id'].values.tolist(),uni_id=valid_df['seq_id'].values.tolist(), activity=valid_df['Activity'].values.tolist(),
                                mol_env_path='./data/DUF_dataset/reaction_emb.lmdb', esm_env_path='./data/DUF_dataset/enzyme_emb.lmdb')
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=args.batchsize, collate_fn=collate_substrate, num_workers=0)

    test_dataset = SubstrateDataset(reactant_id=test_df['reactant_id'].values.tolist(), product_id=test_df['product_id'].values.tolist(),
                                reaction_id=test_df['reaction_id'].values.tolist(),uni_id=test_df['seq_id'].values.tolist(), activity=test_df['Activity'].values.tolist(),
                                mol_env_path='./data/DUF_dataset/reaction_emb.lmdb', esm_env_path='./data/DUF_dataset/enzyme_emb.lmdb')
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batchsize, collate_fn=collate_substrate, num_workers=0)

    model = EnzymaticModel(num_layers=1, hidden_dim=1024, out_dim=512)
    model = model.to(device)

    args.ckpt = os.path.join(args.ckpt, f'duf_ckpt_{args.split_method}.pt')     
    model.load_state_dict(torch.load(args.ckpt, map_location=torch.device('cpu')))

    logger.add(args.log_path)
    model.to(device)

    dist_list = []
    activity_list = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(valid_dataloader):
            r_embedding, p_embedding, esm_embedding, r_padding_mask, p_padding_mask, esm_padding_mask, reaction_id, activity, uid = batch
            r_embedding, p_embedding, esm_embedding, r_padding_mask, p_padding_mask, esm_padding_mask = r_embedding.to(device), p_embedding.to(device), esm_embedding.to(device),\
                                                                        r_padding_mask.to(device), p_padding_mask.to(device), esm_padding_mask.to(device)
            reaction_id, activity = reaction_id.to(device), activity.to(device)
            reactant_emb, enzyme_emb, product_emb = model(esm_emb=esm_embedding, reactant=r_embedding, product=p_embedding, 
                                                        esm_padding_mask=esm_padding_mask, reactant_padding_mask=r_padding_mask, product_padding_mask=p_padding_mask)
            
            fuse_emb = reactant_emb + enzyme_emb
            dist = torch.linalg.norm((fuse_emb-product_emb), dim=-1)   
            dist_list.append(dist)
            activity_list.append(activity)


    dist = torch.cat(dist_list,dim=0).detach().cpu().numpy()
    activity = torch.cat(activity_list,dim=0).detach().cpu().numpy()
    label = np.array(activity)

    max_mcc = 0
    idx = 0
    for i in range(300):
        t = 0.01 * i 
        ll = [1 if i<t else 0 for i in dist]
        ll = np.array(ll)
        mcc = matthews_corrcoef(label, ll)
        if mcc>max_mcc:
            max_mcc = mcc
            idx=i 

    pred_label = np.array([1 if i<0.01*idx else 0 for i in dist])
    acc = np.mean((label==pred_label))
    roc_auc = roc_auc_score(label, 1/dist)
    thred = 0.01*idx

    dist_list = []
    label_list = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            r_embedding, p_embedding, esm_embedding, r_padding_mask, p_padding_mask, esm_padding_mask, reaction_id, label, uid = batch
            r_embedding, p_embedding, esm_embedding, r_padding_mask, p_padding_mask, esm_padding_mask = r_embedding.to(device), p_embedding.to(device), esm_embedding.to(device),\
                                                                        r_padding_mask.to(device), p_padding_mask.to(device), esm_padding_mask.to(device)
            reaction_id, label = reaction_id.to(device), label.to(device)
            reactant_emb, enzyme_emb, product_emb = model(esm_emb=esm_embedding, reactant=r_embedding, product=p_embedding, 
                                                        esm_padding_mask=esm_padding_mask, reactant_padding_mask=r_padding_mask, product_padding_mask=p_padding_mask)
            fuse_emb = reactant_emb + enzyme_emb
            dist = torch.linalg.norm((fuse_emb-product_emb), dim=-1)   
            dist_list.append(dist)
            label_list.append(label)
    
    dist = torch.cat(dist_list,dim=0).detach().cpu().numpy()
    label = torch.cat(label_list,dim=0).detach().cpu().numpy()
    label = np.array(label)

    print(args.ckpt)
    pred_label = np.array([1 if i<thred else 0 for i in dist])
    acc = np.mean((label==pred_label))
    print(f"acc:{acc}")
    mcc = matthews_corrcoef(label, pred_label)
    print(f'MCC:{mcc}')
    roc_auc = roc_auc_score(label, 1/dist)
    print(f'ROC_AUC:{roc_auc}')
    logger.info(f'ACC:{acc}, ROC_AUC:{roc_auc}, MCC:{mcc}, Thredhold:{thred}')

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='setting')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--split_method', default='enzyme', type=str, choices=['random', 'enzyme', 'substrate'])
    parser.add_argument('--log_path', default='./log/DUF_match_train.log', type=str)
    parser.add_argument('--task', default='fine-tune')
    parser.add_argument('--lr', default=0.00005, type=float)
    parser.add_argument('--batchsize', type=int, default=1024)
    parser.add_argument('--epoch', default=2000, type=int)
    parser.add_argument('--early_stop_patience', default=50, type=int)
    parser.add_argument('--ckpt', default='./ckpt', type=str)
    args = parser.parse_args()
    main(args)