import argparse
import torch
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset import EnzymeDataset, collate_fn
from utils.build_utils import same_seed
from model.Enzymatic_reaction import EnzymaticModel

def main(args):
    device = torch.device('cuda')
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
    train_df,valid_df = train_test_split(df, train_size=0.8, random_state=42)
    valid_df, test_df = train_test_split(valid_df, train_size=0.5, random_state=42)

    train_set = EnzymeDataset(reactant_id=train_df['reactant_id'].values.tolist(), product_id=train_df['product_id'].values.tolist(),
                              reaction_id=train_df['reaction_id'].values.tolist(), ec=train_df['EC_id'].values.tolist(),
                                uni_id=train_df['Uniprot_ID'].values.tolist(), mol_env_path=args.mol_env_path, esm_env_path=args.esm_env_path)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=8, pin_memory=True)

    model = EnzymaticModel(num_layers=args.num_layer, hidden_dim=args.hidden_dim, out_dim=args.output_dim)
    model = model.to(device)

    model.eval()
    feats_array = np.zeros((len(train_df), args.output_dim))  
    lb_array = np.ones(len(train_df))
    start = 0
    with torch.no_grad():
        for batch in tqdm(train_loader, desc='fearture bank init'):
            r_embedding, p_embedding, esm_embedding, r_padding_mask, p_padding_mask, esm_padding_mask, reaction_id, ec = batch
            r_embedding, p_embedding, esm_embedding, r_padding_mask, p_padding_mask, esm_padding_mask = r_embedding.to(device), p_embedding.to(device), esm_embedding.to(device),\
                                                                        r_padding_mask.to(device), p_padding_mask.to(device), esm_padding_mask.to(device)
            reaction_id, ec = reaction_id.to(device), ec.to(device)
            _, enzyme_emb, _ = model(esm_emb=esm_embedding, reactant=r_embedding, product=p_embedding, 
                                                        esm_padding_mask=esm_padding_mask, reactant_padding_mask=r_padding_mask, product_padding_mask=p_padding_mask)
            
            ec = ec.long()
            feats_array[start:min(start+args.batch_size,len(train_df)), :] = enzyme_emb.cpu().detach().numpy()
            lb_array[start:min(start+args.batch_size,len(train_df))] = ec.cpu().detach().numpy()
            start += args.batch_size
            
    np.save(args.feat_path, feats_array)
    np.save(args.lb_path, lb_array)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='setting')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--output_dim', default=512, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    parser.add_argument('--num_layer', type=int, default=1)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--data_path', default='./data/rhea_processed_data.csv')
    parser.add_argument('--unimol_dict', default='./data/unimol_smile_dict.pk')
    parser.add_argument('--mol_env_path', default='./data/unimol.lmdb')
    parser.add_argument('--esm_env_path', default='./data/esm_rhea.lmdb')
    parser.add_argument('--feat_path', default='./data/feat.npy')
    parser.add_argument('--lb_path', default='./data/lb.npy')
    args = parser.parse_args()
    main(args)
