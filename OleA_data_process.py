import os
import lmdb
import torch
import pickle
import esm
import pandas as pd
from tqdm import tqdm
from unimol_tools import UniMolRepr

df = pd.read_csv('./data/OleA_dataset/OleA_data.csv')

if not os.path.exists('./data/OleA_dataset/enzyme_emb.lmdb'):
    env = lmdb.open('./data/OleA_dataset/enzyme_emb.lmdb', map_size=1099511627776)
    device = torch.device('cuda')
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.to(device)
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    seq_df = df.drop_duplicates(['seq_id'])
    seq_df.reset_index(drop=True, inplace=True)

    with env.begin(write=True) as txn:
        for i in tqdm(range(len(seq_df))):
            seq = seq_df['SEQ'][i]
            uid = seq_df['seq_id'][i]
            data = [(uid, seq)]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(device)
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33].squeeze(0)
            token_representations = token_representations.cpu()
            txn.put(str(uid).encode(), pickle.dumps(token_representations))
    env.close()

if not os.path.exists('./data/OleA_dataset/reaction_emb.lmdb'):
    env = lmdb.open('./data/OleA_dataset/reaction_emb.lmdb', map_size=1099511627776)
    reaction_list = list(set(df['reaction'].values.tolist()))
    smiles_list = [j for i in reaction_list for j in i.split('>>')]
    smiles_list = [j for i in smiles_list for j in i.split('.')]
    smiles_list = list(sorted(list(set(smiles_list))))

    itosmiles = set()
    for s in smiles_list:
        itosmiles.add(s)
    itosmiles = sorted(list(itosmiles))
    smilestoi = {itosmiles[i]:i for i in range(len(itosmiles))}
    with open('./data/OleA_dataset/OleA_reaction_dict.pk', 'wb') as f:
        pickle.dump([itosmiles,smilestoi],f)

    clf = UniMolRepr(data_type='molecule', remove_hs=False, use_gpu=True)
    with env.begin(write=True) as txn:
        for smi in tqdm(smiles_list):
            reprs = clf.get_repr(smi, return_atomic_reprs=True)
            cls_repr_tensor = torch.tensor(reprs['cls_repr'])
            atom_repr_tensor = torch.tensor(reprs['atomic_reprs'])
            result_tensor = torch.cat([cls_repr_tensor, atom_repr_tensor.squeeze(0)], dim=0)
            idx = smilestoi[smi]
            txn.put(str(idx).encode(), pickle.dumps(result_tensor))
    env.close()
