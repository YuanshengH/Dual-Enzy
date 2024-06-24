import os
import lmdb
import torch
import pickle
import esm
import pandas as pd
from tqdm import tqdm
from unimol_tools import UniMolRepr

df = pd.read_excel('/mnt/Dual_Enzy/data/aminotransferases_dataset/aminotransferase_data.xlsx')
seq_df = pd.read_csv('/mnt/Dual_Enzy/data/aminotransferases_dataset/aminotransferase_sequence_file.txt',sep='\t')
new_df = pd.DataFrame(columns=['Entry','Sequence','Substrate','Label'])
aa_list = df.iloc[0,2:].tolist()
for i in range(len(df)):
    if i == 0:
        continue
    entry = df.iloc[i,1].strip()
    seq = seq_df.loc[seq_df.Entry==entry]['Sequence'].values[0]
    for j in range(len(aa_list)):
        sub = aa_list[j]
        activity = df.iloc[i,2+j]
        if activity>=0.1:
            label=1
        else:
            label=0
        tem_list = [entry, seq, sub, label]
        new_df.loc[len(new_df)] = tem_list

# reaction
aa_dict = {'C1=C(NC=N1)C[C@@H](C(=O)O)N': 'his', 'C(C(=O)O)N': 'gly', 'C[C@@H](C(=O)O)N': 'ala', 'CC(C)[C@@H](C(=O)O)N': 'val', 
           'CC(C)C[C@@H](C(=O)O)N': 'leu', 'CC[C@H](C)[C@@H](C(=O)O)N': 'ile', 'C1=CC=C(C=C1)C[C@@H](C(=O)O)N': 'phe', 
           'C1=CC(=CC=C1C[C@@H](C(=O)O)N)O': 'tyr', 'C1=CC=C2C(=C1)C(=CN2)C[C@@H](C(=O)O)N': 'trp', 'C([C@@H](C(=O)O)N)O': 'ser', 
           'C[C@H]([C@@H](C(=O)O)N)O': 'thr', 'C([C@@H](C(=O)O)N)S': 'cys', 'CSCC[C@@H](C(=O)O)N': 'met', 'C([C@@H](C(=O)O)N)C(=O)N': 'asn', 
           'C(CC(=O)N)[C@@H](C(=O)O)N': 'gln', 'C([C@@H](C(=O)O)N)C(=O)O': 'asp', 'C(CCN)C[C@@H](C(=O)O)N': 'lys', 'C(C[C@@H](C(=O)O)N)CN=C(N)N': 'arg'}

his = "O=C([O-])[C@@H]([NH3+])CC1=CN=CN1.[O-]C(=O)CCC(=O)C([O-])=O>>[NH3+][C@@H](CCC([O-])=O)C([O-])=O.O=C([O-])C(CC1=CN=CN1)=O"
gly = "O=C([O-])C[NH3+].[O-]C(=O)CCC(=O)C([O-])=O>>[NH3+][C@@H](CCC([O-])=O)C([O-])=O.O=C([O-])C=O"
ala = 'C[C@H]([NH3+])C([O-])=O.[O-]C(=O)CCC(=O)C([O-])=O>>[NH3+][C@@H](CCC([O-])=O)C([O-])=O.CC(C([O-])=O)=O'
val = 'CC([C@H]([NH3+])C([O-])=O)C.[O-]C(=O)CCC(=O)C([O-])=O>>[NH3+][C@@H](CCC([O-])=O)C([O-])=O.CC(C(C([O-])=O)=O)C'
leu = 'CC(C[C@H]([NH3+])C([O-])=O)C.[O-]C(=O)CCC(=O)C([O-])=O>>[NH3+][C@@H](CCC([O-])=O)C([O-])=O.CC(CC(C([O-])=O)=O)C'
ile = 'CC[C@@H]([C@H]([NH3+])C([O-])=O)C.[O-]C(=O)CCC(=O)C([O-])=O>>[NH3+][C@@H](CCC([O-])=O)C([O-])=O.CC[C@@H](C(C([O-])=O)=O)C'
phe = 'O=C([O-])[C@@H]([NH3+])CC1=CC=CC=C1.[O-]C(=O)CCC(=O)C([O-])=O>>[NH3+][C@@H](CCC([O-])=O)C([O-])=O.O=C([O-])C(CC2=CC=CC=C2)=O'
tyr = 'O=C([O-])[C@@H]([NH3+])CC1=CC=C(O)C=C1.[O-]C(=O)CCC(=O)C([O-])=O>>[NH3+][C@@H](CCC([O-])=O)C([O-])=O.O=C([O-])C(CC2=CC=C(O)C=C2)=O'
trp = 'O=C([O-])[C@@H]([NH3+])CC3=CNC4=CC=CC=C43.[O-]C(=O)CCC(=O)C([O-])=O>>[NH3+][C@@H](CCC([O-])=O)C([O-])=O.O=C([O-])C(CC1=CNC2=CC=CC=C21)=O'
thr = 'C[C@@H](O)[C@H]([NH3+])C([O-])=O.[O-]C(=O)CCC(=O)C([O-])=O>>[NH3+][C@@H](CCC([O-])=O)C([O-])=O.C[C@@H](O)C(C([O-])=O)=O'
cys = 'O=C([O-])[C@@H]([NH3+])CS.[O-]C(=O)CCC(=O)C([O-])=O>>[NH3+][C@@H](CCC([O-])=O)C([O-])=O.O=C([O-])C(CS)=O'
met = 'CSCC[C@H]([NH3+])C([O-])=O.[O-]C(=O)CCC(=O)C([O-])=O>>[NH3+][C@@H](CCC([O-])=O)C([O-])=O.CSCCC(C([O-])=O)=O'
asn = 'O=C([O-])[C@@H]([NH3+])CC(N)=O.[O-]C(=O)CCC(=O)C([O-])=O>>[NH3+][C@@H](CCC([O-])=O)C([O-])=O.O=C([O-])C(CC(N)=O)=O'
gln = 'O=C(N)CC[C@H]([NH3+])C([O-])=O.[O-]C(=O)CCC(=O)C([O-])=O>>[NH3+][C@@H](CCC([O-])=O)C([O-])=O.O=C(N)CCC(C([O-])=O)=O'
asp = 'O=C([O-])[C@@H]([NH3+])CC(O)=O.[O-]C(=O)CCC(=O)C([O-])=O>>[NH3+][C@@H](CCC([O-])=O)C([O-])=O.O=C([O-])C(CC(O)=O)=O'
lys = 'NCCCC[C@H]([NH3+])C([O-])=O.[O-]C(=O)CCC(=O)C([O-])=O>>[NH3+][C@@H](CCC([O-])=O)C([O-])=O.NCCCCC(C([O-])=O)=O'
arg = 'O=C([O-])[C@@H]([NH3+])CCC/N=C(N)/N.[O-]C(=O)CCC(=O)C([O-])=O>>[NH3+][C@@H](CCC([O-])=O)C([O-])=O.O=C([O-])C(CCC/N=C(N)/N)=O'
ser = 'O=C([O-])[C@@H]([NH3+])CO.[O-]C(=O)CCC(=O)C([O-])=O>>[NH3+][C@@H](CCC([O-])=O)C([O-])=O.O=C([O-])C(CO)=O'

reaction_dict = {'his':his,'gly':gly,'ala':ala,'val':val,'leu':leu,'ile':ile,'phe':phe,'tyr':tyr,'trp':trp,'thr':thr,'cys':cys,'met':met,'asn':asn,'gln':gln,'asp':asp,'lys':lys,'arg':arg, 'ser':ser}
new_df['aa'] = new_df['Substrate'].apply(lambda x: aa_dict[x])
new_df['reaction'] = new_df['aa'].apply(lambda x:reaction_dict[x])
print()

if not os.path.exists('/mnt/Dual_Enzy/data/aminotransferases_dataset/enzyme_emb.lmdb'):
    env = lmdb.open('/mnt/Dual_Enzy/data/aminotransferases_dataset/enzyme_emb.lmdb', map_size=1099511627776)
    device = torch.device('cuda')
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.to(device)
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    tmp = set()
    with env.begin(write=True) as txn:
        for i in range(len(seq_df)):
            seq = seq_df['Sequence'][i]
            uid = seq_df['Entry'][i]
            data = [(uid, seq)]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(device)
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33].squeeze(0)
            token_representations = token_representations.cpu()
            txn.put(uid.encode(), pickle.dumps(token_representations))
    env.close()

if not os.path.exists('/mnt/Dual_Enzy/data/aminotransferases_dataset/reaction_emb.lmdb'):
    env = lmdb.open('/mnt/Dual_Enzy/data/aminotransferases_dataset/reaction_emb.lmdb', map_size=1099511627776)
    reaction_list = list(set(new_df['reaction'].values.tolist()))
    smiles_list = [j for i in reaction_list for j in i.split('>>')]
    smiles_list = [j for i in smiles_list for j in i.split('.')]
    smiles_list = list(sorted(list(set(smiles_list))))

    itosmiles = set()
    for s in smiles_list:
        itosmiles.add(s)
    itosmiles = sorted(list(itosmiles))
    smilestoi = {itosmiles[i]:i for i in range(len(itosmiles))}
    with open('/mnt/Dual_Enzy/data/aminotransferases_dataset/aminotransferases_reaction_dict.pk', 'wb') as f:
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

# remove sample existed in training set
new_df = new_df[~((new_df.Entry=='P77434') & (new_df.aa=='ala'))]
new_df = new_df[~((new_df.Entry=='P00509') & (new_df.aa=='asp'))]
new_df = new_df[~((new_df.Entry=='P0A959') & (new_df.aa=='ala'))]
new_df = new_df[~((new_df.Entry=='P39576') & (new_df.aa=='ile'))]
new_df = new_df[~((new_df.Entry=='P39576') & (new_df.aa=='leu'))]
new_df = new_df[~((new_df.Entry=='P39576') & (new_df.aa=='val'))]
new_df.to_csv('/mnt/Dual_Enzy/data/aminotransferases_dataset/aminotransferases_data.csv', index=False)