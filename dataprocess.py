import lmdb
import torch
import argparse
import pickle
import pandas as pd
import esm
from tqdm import tqdm
from rdkit import Chem
from unimol_tools import UniMolRepr
from utils.build_utils import same_seed

def main(args):
    same_seed(42)
    if args.task == 'unimol':
    # get Uni-Mol Embedding
        clf = UniMolRepr(data_type='molecule', remove_hs=False, use_gpu=True)
        df = pd.read_csv(args.data_path)
        reaction_list = list(set(df['reaction'].values.tolist()))
        smiles_list = [j for i in reaction_list for j in i.split('>>')]
        smiles_list = [j for i in smiles_list for j in i.split('.')]
        smiles_list = list(sorted(list(set(smiles_list))))
        itosmiles = set()
        for s in smiles_list:
            itosmiles.add(s)
        itosmiles = sorted(list(itosmiles))
        smilestoi = {itosmiles[i]:i for i in range(len(itosmiles))}
        with open('./data/train_data/unimol_smile_dict.pk', 'wb') as f:
            pickle.dump([itosmiles,smilestoi],f)

        batchsize = 64
        env = lmdb.open('./data/train_data/unimol.lmdb', map_size=2199023255556)
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
        print('Finish')

    elif args.task == 'esm_extract':
        df = pd.read_csv(args.data_path)
        df.drop_duplicates(subset=['Uniprot_ID'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        print(f"Sequence number: {len(df)}")
        device = torch.device('cuda')
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        model.to(device)
        batch_converter = alphabet.get_batch_converter()
        model.eval()
        lmdb_path = './data/train_data/esm_rhea.lmdb'
        env = lmdb.open(lmdb_path, map_size=2199023255556)
        with env.begin(write=True) as txn:
            for j in tqdm(range(len(df))):
                enzyme = df['Uniprot_ID'][j]
                sequence = df['Sequence'][j]
                data = [(enzyme, sequence)]
                batch_labels, batch_strs, batch_tokens = batch_converter(data)
                batch_tokens = batch_tokens.to(device)
                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[33], return_contacts=False)
                token_representations = results["representations"][33].squeeze(0)
                token_representations = token_representations.cpu()
                txn.put(enzyme.encode(), pickle.dumps(token_representations))
        env.close()

    elif args.task == 'data_process':
        df = pd.read_csv(args.data_path)
        print(f'Data number: {len(df)}')

        # 获得产物和底物
        df['reactant'] = df['reaction'].apply(lambda x:x.split('>>')[0])
        df['product'] = df['reaction'].apply(lambda x:x.split('>>')[-1])

        # 去除原子数量大于256的样本
        l1 = list(set(df['reactant'].values.tolist()) | set(df['product'].values.tolist()))
        l2 = list(set([j for i in l1 for j in i.split('.')]))
        l3 = [i for i in l2 if Chem.AddHs(Chem.MolFromSmiles(i)).GetNumAtoms()>256]
        df = df[~df.apply(lambda row: any(compound in row['reactant'].split('.') or compound in row['product'].split('.') for compound in l3), axis=1)]

        # 去除底物和产物相同的样本
        df = df[df['product'] != df['reactant']]

        # 去除EC出现次数少于10的样本
        ec_count = df['EC'].value_counts()
        rare_ec = ec_count[ec_count < 10].index
        df = df[~df['EC'].isin(rare_ec)]

        df.reset_index(drop=True, inplace=True)
        print(f'Data number after process: {len(df)}')
        df.to_csv('./data/rhea_processed_data.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='esm_extract', choices=['unimol','esm_extract', 'data_process'])
    parser.add_argument('--data_path', default='./data/rhea_processed_data.csv')
    args = parser.parse_args()
    main(args)


