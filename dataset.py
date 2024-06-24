import lmdb
import torch
import pickle
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence                                     

def get_unimol_embedding(txn, id_list):
    repr_list = []
    id_list = sorted(id_list)
    for i in id_list:
        embedding = torch.Tensor(pickle.loads(txn.get(str(i).encode())))
        repr_list.append(embedding)
    final_repr = torch.cat(repr_list, dim=0)
    return final_repr

def collate_fn_RXNMapper(data):
    r_unimol, p_unimol, esm_embedding, ec, site = zip(*data)
    r_length = [r.size(0) for r in r_unimol]
    p_length = [p.size(0) for p in p_unimol]
    esm_length = [e.size(0) for e in esm_embedding]
    max_esm_length = max(esm_length)
    max_r_length = max(r_length)
    max_p_length = max(p_length)
    esm_padding_mask = torch.zeros(len(esm_length), max_esm_length)
    r_padding_mask = torch.zeros(len(r_length), max_r_length)
    p_padding_mask = torch.zeros(len(p_length), max_p_length)
    for i in range(len(r_length)):
        r_padding_mask[i, r_length[i]:] = 1
        p_padding_mask[i, p_length[i]:] = 1
        esm_padding_mask[i, esm_length[i]:] = 1
    r_padding_mask = (r_padding_mask == 1)
    p_padding_mask = (p_padding_mask == 1)
    esm_padding_mask = (esm_padding_mask == 1)
    new_r = pad_sequence(r_unimol, padding_value=1, batch_first=True)
    new_p = pad_sequence(p_unimol, padding_value=1, batch_first=True)
    esm_embedding = pad_sequence(esm_embedding, padding_value=1, batch_first=True)
    site = pad_sequence(site, batch_first=True, padding_value=-1)
    return new_r, new_p, esm_embedding, r_padding_mask, p_padding_mask, esm_padding_mask, ec, site

def collate_fn(data):
    r_unimol, p_unimol, esm_embedding, reaction_id, ec = zip(*data)
    r_length = [r.size(0) for r in r_unimol]
    p_length = [p.size(0) for p in p_unimol]
    esm_length = [e.size(0) for e in esm_embedding]
    max_esm_length = max(esm_length)
    max_r_length = max(r_length)
    max_p_length = max(p_length)
    esm_padding_mask = torch.zeros(len(esm_length), max_esm_length)
    r_padding_mask = torch.zeros(len(r_length), max_r_length)
    p_padding_mask = torch.zeros(len(p_length), max_p_length)
    for i in range(len(r_length)):
        r_padding_mask[i, r_length[i]:] = 1
        p_padding_mask[i, p_length[i]:] = 1
        esm_padding_mask[i, esm_length[i]:] = 1
    r_padding_mask = (r_padding_mask == 1)
    p_padding_mask = (p_padding_mask == 1)
    esm_padding_mask = (esm_padding_mask == 1)
    new_r = pad_sequence(r_unimol, padding_value=1, batch_first=True)
    new_p = pad_sequence(p_unimol, padding_value=1, batch_first=True)
    esm_embedding = pad_sequence(esm_embedding, padding_value=1, batch_first=True)
    return new_r, new_p, esm_embedding, r_padding_mask, p_padding_mask, esm_padding_mask, torch.Tensor(reaction_id).int(), torch.Tensor(ec).int()

def collate_fn_eval(data):
    r_unimol, p_unimol, esm_embedding, reaction_id, ec, reactant_id, product_id, uni_id = zip(*data)
    r_length = [r.size(0) for r in r_unimol]
    p_length = [p.size(0) for p in p_unimol]
    esm_length = [e.size(0) for e in esm_embedding]
    max_esm_length = max(esm_length)
    max_r_length = max(r_length)
    max_p_length = max(p_length)
    esm_padding_mask = torch.zeros(len(esm_length), max_esm_length)
    r_padding_mask = torch.zeros(len(r_length), max_r_length)
    p_padding_mask = torch.zeros(len(p_length), max_p_length)
    for i in range(len(r_length)):
        r_padding_mask[i, r_length[i]:] = 1
        p_padding_mask[i, p_length[i]:] = 1
        esm_padding_mask[i, esm_length[i]:] = 1
    r_padding_mask = (r_padding_mask == 1)
    p_padding_mask = (p_padding_mask == 1)
    esm_padding_mask = (esm_padding_mask == 1)
    new_r = pad_sequence(r_unimol, padding_value=1, batch_first=True)
    new_p = pad_sequence(p_unimol, padding_value=1, batch_first=True)
    esm_embedding = pad_sequence(esm_embedding, padding_value=1, batch_first=True)
    return new_r, new_p, esm_embedding, r_padding_mask, p_padding_mask, esm_padding_mask, torch.Tensor(reaction_id).int(), torch.Tensor(ec).int(), reactant_id, product_id, uni_id

def collate_substrate(data):
    r_unimol, p_unimol, esm_embedding, reaction_id, activity, uid = zip(*data)
    r_length = [r.size(0) for r in r_unimol]
    p_length = [p.size(0) for p in p_unimol]
    esm_length = [e.size(0) for e in esm_embedding]
    max_esm_length = max(esm_length)
    max_r_length = max(r_length)
    max_p_length = max(p_length)
    esm_padding_mask = torch.zeros(len(esm_length), max_esm_length)
    r_padding_mask = torch.zeros(len(r_length), max_r_length)
    p_padding_mask = torch.zeros(len(p_length), max_p_length)
    for i in range(len(r_length)):
        r_padding_mask[i, r_length[i]:] = 1
        p_padding_mask[i, p_length[i]:] = 1
        esm_padding_mask[i, esm_length[i]:] = 1
    r_padding_mask = (r_padding_mask == 1)
    p_padding_mask = (p_padding_mask == 1)
    esm_padding_mask = (esm_padding_mask == 1)
    new_r = pad_sequence(r_unimol, padding_value=1, batch_first=True)
    new_p = pad_sequence(p_unimol, padding_value=1, batch_first=True)
    esm_embedding = pad_sequence(esm_embedding, padding_value=1, batch_first=True)
    return new_r, new_p, esm_embedding, r_padding_mask, p_padding_mask, esm_padding_mask, torch.Tensor(reaction_id).int(), torch.Tensor(activity), uid
    
class EnzymeDataset(Dataset):
    def __init__(self, reactant_id, product_id, reaction_id, ec, uni_id, mol_env_path, esm_env_path):
        self.reactant = reactant_id
        self.product = product_id
        self.reaction_id = reaction_id
        self.ec = ec
        self.uni_id = uni_id
        self.mol_env_path = mol_env_path
        self.seq_env_path = esm_env_path 
        
    def open_lmdb(self):
        self.mol_env = lmdb.open(self.mol_env_path, readonly=True, create=False, max_readers=512,)
        self.seq_env = lmdb.open(self.seq_env_path, readonly=True, create=False, max_readers=512,)
        self.mol_txn = self.mol_env.begin()
        self.seq_txn = self.seq_env.begin()

    def __len__(self):
        return len(self.uni_id)
    
    def __getitem__(self, idx):
        if not hasattr(self, 'mol_txn'):
            self.open_lmdb()
        uni_id = self.uni_id[idx]
        # print(f"idx:{idx}, uid:{uni_id}")
        reactant_id = self.reactant[idx]    # list
        product_id = self.product[idx]
        r_embedding = get_unimol_embedding(self.mol_txn, reactant_id)
        p_embedding = get_unimol_embedding(self.mol_txn, product_id)
        esm_embedding = pickle.loads(self.seq_txn.get(uni_id.encode()))
        return r_embedding, p_embedding, esm_embedding, self.reaction_id[idx], self.ec[idx]
    
class EvalDataset(Dataset):
    def __init__(self, reactant_id, product_id, reaction_id, ec, uni_id, mol_env_path, esm_env_path):
        self.reactant = reactant_id
        self.product = product_id
        self.reaction_id = reaction_id
        self.ec = ec
        self.uni_id = uni_id
        self.mol_env_path = mol_env_path
        self.seq_env_path = esm_env_path 
        
    def open_lmdb(self):
        self.mol_env = lmdb.open(self.mol_env_path, readonly=True, create=False,
                                 meminit=False)
        self.seq_env = lmdb.open(self.seq_env_path, readonly=True, create=False,
                                 meminit=False)
        self.mol_txn = self.mol_env.begin(buffers=True)
        self.seq_txn = self.seq_env.begin(buffers=True)

    def __len__(self):
        return len(self.uni_id)
    
    def __getitem__(self, idx):
        if not hasattr(self, 'mol_txn'):
            self.open_lmdb()
        uni_id = self.uni_id[idx]
        reactant_id = self.reactant[idx]    # list
        product_id = self.product[idx]
        r_embedding = get_unimol_embedding(self.mol_txn, reactant_id)
        p_embedding = get_unimol_embedding(self.mol_txn, product_id)
        esm_embedding = pickle.loads(self.seq_txn.get(uni_id.encode()))
        return r_embedding, p_embedding, esm_embedding, self.reaction_id[idx], self.ec[idx], self.reactant[idx], self.product[idx], self.uni_id[idx]

class SubstrateDataset(Dataset):
    def __init__(self, reactant_id, product_id, reaction_id, uni_id, activity, mol_env_path, esm_env_path):
        self.reactant = reactant_id
        self.product = product_id
        self.reaction_id = reaction_id
        self.uni_id = uni_id
        self.activity = activity
        self.mol_env_path = mol_env_path
        self.seq_env_path = esm_env_path 
        
    def open_lmdb(self):
        self.mol_env = lmdb.open(self.mol_env_path, readonly=True, create=False, max_readers=512,)
        self.seq_env = lmdb.open(self.seq_env_path, readonly=True, create=False, max_readers=512,)
        self.mol_txn = self.mol_env.begin()
        self.seq_txn = self.seq_env.begin()

    def __len__(self):
        return len(self.uni_id)
    
    def __getitem__(self, idx):
        if not hasattr(self, 'mol_txn'):
            self.open_lmdb()
        uni_id = self.uni_id[idx]
        reactant_id = self.reactant[idx]    # list
        product_id = self.product[idx]
        r_embedding = get_unimol_embedding(self.mol_txn, reactant_id)
        p_embedding = get_unimol_embedding(self.mol_txn, product_id)
        esm_embedding = pickle.loads(self.seq_txn.get(uni_id.encode()))
        return r_embedding, p_embedding, esm_embedding, self.reaction_id[idx], self.activity[idx], uni_id
    
class RXNMapperDataset(Dataset):
    def __init__(self, reactant_id, product_id, site_label, ec, uni_id, length, mol_env_path, esm_env_path):
        self.reactant = reactant_id
        self.product = product_id
        self.ec = ec
        self.length = length
        self.site_label = site_label
        self.uni_id = uni_id
        self.mol_env_path = mol_env_path
        self.seq_env_path = esm_env_path 
        
    def open_lmdb(self):
        self.mol_env = lmdb.open(self.mol_env_path, readonly=True, create=False, max_readers=512,)
        self.seq_env = lmdb.open(self.seq_env_path, readonly=True, create=False, max_readers=512,)
        self.mol_txn = self.mol_env.begin()
        self.seq_txn = self.seq_env.begin()

    def calculate_active_sites(self, site_label, sequence_length):
        site_label = eval(site_label)  
        active_site = torch.zeros((int(sequence_length), ))
        for one_site in site_label:
            if len(one_site) == 1:
                active_site[one_site[0]-1] = 1
            elif len(one_site) == 2:
                b, e = one_site
                site_indices = [k-1 for k in range(b, e+1)]
                active_site[site_indices] = 1
            else:
                raise ValueError(
                    'The label of active site is not standard !!!')
        return active_site
    
    def __len__(self):
        return len(self.uni_id)
    
    def __getitem__(self, idx):
        if not hasattr(self, 'mol_txn'):
            self.open_lmdb()
        uni_id = self.uni_id[idx]
        reactant_id = self.reactant[idx]    # list
        product_id = self.product[idx]
        r_embedding = get_unimol_embedding(self.mol_txn, reactant_id)
        p_embedding = get_unimol_embedding(self.mol_txn, product_id)
        esm_embedding = pickle.loads(self.seq_txn.get(uni_id.encode()))
        site = self.calculate_active_sites(self.site_label[idx], self.length[idx])
        return r_embedding, p_embedding, esm_embedding, self.ec[idx], site