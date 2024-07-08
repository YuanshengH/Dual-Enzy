import os
import lmdb
import torch
import pickle
import esm
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from unimol_tools import UniMolRepr
from model.Enzymatic_reaction import EnzymaticModel
from sklearn.metrics import confusion_matrix
from dataset import RXNMapperDataset, collate_fn_RXNMapper

def calculate_metrics_sklearn(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  
    overlap_score = tp / (tp + fn)
    return overlap_score, fpr,

def main():
    df = pd.read_csv('./data/site_pred_data/rxnaamapper.csv')
    device = torch.device('cuda')
    if not os.path.exists('./data/site_pred_data/esm.lmdb'):
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        model.to(device)
        batch_converter = alphabet.get_batch_converter()
        model.eval()
        new_df = df.drop_duplicates('pdb-id')
        env = lmdb.open('./data/site_pred_data/esm.lmdb', map_size=1099511627776)
        with env.begin(write=True) as txn:
            for idx, row in new_df.iterrows():
                enzyme = row['pdb-id']
                seq = row['aa_sequence']
                data = [(enzyme, seq)]
                batch_labels, batch_strs, batch_tokens = batch_converter(data)
                batch_tokens = batch_tokens.to(device)
                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[33], return_contacts=False)
                token_representations = results["representations"][33].squeeze(0)
                token_representations = token_representations.cpu()
                txn.put(enzyme.encode(), pickle.dumps(token_representations))     
        env.close()   

    if not os.path.exists('./data/site_pred_data/unimol.lmdb'):
        clf = UniMolRepr(data_type='molecule', remove_hs=False, use_gpu=True)
        rxn_list = df.rxn.unique()
        rxn_list = [j for i in rxn_list for j in i.split('>>')]
        rxn_list = [i.split('|')[0] for i in rxn_list]
        rxn_list = [j for i in rxn_list for j in i.split('.')]
        rxn_list = sorted(list(set(rxn_list)))
        itosmiles = set()
        for s in rxn_list:
            itosmiles.add(s)
        itosmiles = sorted(list(itosmiles))
        smilestoi = {itosmiles[i]:i for i in range(len(itosmiles))}
        with open('./data/site_pred_data/rxnaamapper_dict.pk', 'wb') as f:
            pickle.dump([itosmiles,smilestoi],f)
        env = lmdb.open('./data/site_pred_data/unimol.lmdb', map_size=1099511627776)
        with env.begin(write=True) as txn:
            for smi in tqdm(rxn_list):
                idx = smilestoi[smi]
                reprs = clf.get_repr(smi, return_atomic_reprs=True)
                cls_repr_tensor = torch.tensor(reprs['cls_repr'])
                atom_repr_tensor = torch.tensor(reprs['atomic_reprs'])
                result_tensor = torch.cat([cls_repr_tensor, atom_repr_tensor.squeeze(0)], dim=0)
                txn.put(str(idx).encode(), pickle.dumps(result_tensor))
        env.close()     
    with open('./data/site_pred_data/rxnaamapper_dict.pk', 'rb') as f:
        itosmiles, smilestoi = pickle.load(f) 
    df['reactant'] = df['rxn'].apply(lambda x:x.split('>>')[0].split('|')[0])
    df['product'] = df['rxn'].apply(lambda x:x.split('>>')[-1])
    df['reactant_id'] = df['reactant'].apply(lambda x: [smilestoi[i] for i in x.split('.')])
    df['product_id'] = df['product'].apply(lambda x: [smilestoi[i] for i in x.split('.')])
    df['length'] = df['aa_sequence'].apply(lambda x:len(x))
    batch_size = 64

    dataset = RXNMapperDataset(reactant_id=df['reactant_id'].values.tolist(), product_id=df['product_id'].values.tolist(), length=df['length'],
                               ec=df['EC number'].values.tolist(), uni_id=df['pdb-id'].values.tolist(), site_label=df['active_site'],
                               mol_env_path='./data/site_pred_data/unimol.lmdb', esm_env_path='./data/site_pred_data/esm.lmdb')
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=collate_fn_RXNMapper, )
    model = EnzymaticModel(num_layers=1, hidden_dim=1024, out_dim=512, require_attn=True)

    checkpoint = torch.load('./ckpt/checkpoint.pt', map_location='cpu')
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()})
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    for head in [4]:
        with torch.no_grad():
            for batch in tqdm(dataloader):
                r_embedding, p_embedding, esm_embedding, r_padding_mask, p_padding_mask, esm_padding_mask, ec, site = batch
                r_embedding, p_embedding, esm_embedding, r_padding_mask, p_padding_mask, esm_padding_mask = r_embedding.to(device), p_embedding.to(device), esm_embedding.to(device),\
                                                                            r_padding_mask.to(device), p_padding_mask.to(device), esm_padding_mask.to(device)
                _, _, _, enzyme_self_attn, _, _, _ = model(esm_emb=esm_embedding, reactant=r_embedding, product=p_embedding, 
                                                            esm_padding_mask=esm_padding_mask, reactant_padding_mask=r_padding_mask, product_padding_mask=p_padding_mask)
                enzyme_self_attn = enzyme_self_attn[:,:,1:-1,1:-1]
                pred_site_onehot = torch.zeros(site.size())
                pred_site_onehot[(torch.sum(enzyme_self_attn[:,head],dim=1)>0.10)] = 1
                site = site.view(-1)
                pred_site_onehot = pred_site_onehot.view(-1)
                mask = (site!=-1)
                test_predictions = pred_site_onehot[mask]
                test_sites = site[mask]
                all_preds.extend(test_predictions.cpu().numpy())
                all_labels.extend(test_sites.cpu().numpy())
                print() 
                
        results = calculate_metrics_sklearn(all_labels, all_preds)
        print(f"Head:{str(head)}")
        print("Recall: {:.5f}, FPR: {:.5f}".format(*results))
        print()

if __name__=="__main__":
    main()