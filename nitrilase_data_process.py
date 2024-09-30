import os
import lmdb
import torch
import pickle
import esm
import pandas as pd
from unimol_tools import UniMolRepr

df = pd.read_csv('./data/nitrilase_dataset/nitrilase_data.csv')

if not os.path.exists('./data/nitrilase_dataset/enzyme_emb.lmdb'):
    env = lmdb.open('./data/nitrilase_dataset/enzyme_emb.lmdb', map_size=1099511627776)
    device = torch.device('cuda')
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.to(device)
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    tmp = set()
    with env.begin(write=True) as txn:
        for idx in range(len(df)):
            enzyme = df['enzyme'][idx]
            sequence = df['Sequence'][idx]
            if enzyme not in tmp:
                tmp.add(enzyme)
            else:
                continue
            data = [(enzyme, sequence)]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(device)
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33].squeeze(0)
            token_representations = token_representations.cpu()
            txn.put(enzyme.encode(), pickle.dumps(token_representations))
    env.close()

if not os.path.exists('./data/nitrilase_dataset/reaction_emb.lmdb'):
    env = lmdb.open('./data/nitrilase_dataset/reaction_emb.lmdb', map_size=1099511627776)
    clf = UniMolRepr(data_type='molecule', remove_hs=False, use_gpu=True)
    reaction_list = list(set(df['Reaction'].values.tolist()))
    smiles_list = [j for i in reaction_list for j in i.split('>>')]
    smiles_list = [j for i in smiles_list for j in i.split('.')]
    smiles_list = list(sorted(list(set(smiles_list))))
    itosmiles = set()
    for s in smiles_list:
        itosmiles.add(s)
    itosmiles = sorted(list(itosmiles))
    smilestoi = {itosmiles[i]:i for i in range(len(itosmiles))}
    with open('./data/nitrilase_dataset/nitrilase_reaction_dict.pk', 'wb') as f:
        pickle.dump([itosmiles,smilestoi],f)

    with env.begin(write=True) as txn:
        for smi in smiles_list:
            key = smilestoi[smi]
            reprs = clf.get_repr(smi, return_atomic_reprs=True)
            cls_repr_tensor = torch.tensor(reprs['cls_repr'])
            atom_repr_tensor = torch.tensor(reprs['atomic_reprs'])
            result_tensor = torch.cat([cls_repr_tensor, atom_repr_tensor.squeeze(0)], dim=0)
            txn.put(str(key).encode(), pickle.dumps(result_tensor))
    env.close()
    print('Finish')
