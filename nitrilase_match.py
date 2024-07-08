import torch 
import pickle   
import argparse                                           
import pandas as pd
import numpy as np
import torch.optim as optim
from loguru import logger
from tqdm import tqdm
from torch.utils.data import DataLoader 
from model.Enzymatic_reaction import EnzymaticModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from dataset import SubstrateDataset, collate_substrate
from utils.build_utils import same_seed

def main(args):
    same_seed(42)
    print(args)
    device = torch.device('cuda')
    df = pd.read_csv('./data/nitrilase_dataset/nitrilase_data.csv')
    df['label'] = df['activity'].apply(lambda x:1 if x>2 else 0)
    rxntoi = {r:idx for idx, r in enumerate(sorted(list(set(df['Reaction'].values.tolist()))))}
    with open('./data/nitrilase_dataset/nitrilase_reaction_dict.pk', 'rb') as f:
        itosmiles, smilestoi = pickle.load(f)
    df['reaction_id'] = df['Reaction'].apply(lambda x:rxntoi[x])
    df['reactant'] = df['Reaction'].apply(lambda x:x.split('>>')[0])
    df['product'] = df['Reaction'].apply(lambda x:x.split('>>')[-1])
    df['reactant_id'] = df['reactant'].apply(lambda x: [smilestoi[i] for i in x.split('.')])
    df['product_id'] = df['product'].apply(lambda x: [smilestoi[i] for i in x.split('.')])

    # split data randomly
    train_df,test_df = train_test_split(df, train_size=0.7, random_state=42)
    valid_df, test_df = train_test_split(test_df, train_size=(1/3), random_state=42)

    # 根据enzyme划分训练集和测试集
    # enzyme_list = df.enzyme.unique().tolist()
    # np.random.shuffle(enzyme_list)
    # test_size = int(len(enzyme_list)*0.3)
    # test_enzyme = enzyme_list[:test_size]
    # train_enzyme = enzyme_list[test_size:]
    # train_df = df[df['enzyme'].isin(train_enzyme)]
    # test_df = df[df['enzyme'].isin(test_enzyme)]
    # valid_df, test_df = train_test_split(test_df, train_size=(1/3), random_state=42)

    # 根据底物划分训练集和测试集
    # substrates_list = df.metabolite.unique().tolist()
    # np.random.shuffle(substrates_list)
    # test_size = int(len(substrates_list)*0.3)
    # test_substrates = substrates_list[:test_size]
    # train_substrates = substrates_list[test_size:]
    # train_df = df[df['metabolite'].isin(train_substrates)]
    # test_df = df[df['metabolite'].isin(test_substrates)]
    # valid_df, test_df = train_test_split(test_df, train_size=(1/3), random_state=42)

    train_dataset = SubstrateDataset(reactant_id=train_df['reactant_id'].values.tolist(), product_id=train_df['product_id'].values.tolist(),
                                reaction_id=train_df['reaction_id'].values.tolist(),uni_id=train_df['enzyme'].values.tolist(), activity=train_df['activity'].values.tolist(),
                                mol_env_path='./data/nitrilase_dataset/reaction_emb.lmdb', esm_env_path='./data/nitrilase_dataset/enzyme_emb.lmdb')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batchsize, collate_fn=collate_substrate)

    valid_dataset = SubstrateDataset(reactant_id=valid_df['reactant_id'].values.tolist(), product_id=valid_df['product_id'].values.tolist(),
                                reaction_id=valid_df['reaction_id'].values.tolist(),uni_id=valid_df['enzyme'].values.tolist(), activity=valid_df['activity'].values.tolist(),
                                mol_env_path='./data/nitrilase_dataset/reaction_emb.lmdb', esm_env_path='./data/nitrilase_dataset/enzyme_emb.lmdb')
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=args.batchsize, collate_fn=collate_substrate)

    test_dataset = SubstrateDataset(reactant_id=test_df['reactant_id'].values.tolist(), product_id=test_df['product_id'].values.tolist(),
                                reaction_id=test_df['reaction_id'].values.tolist(),uni_id=test_df['enzyme'].values.tolist(), activity=test_df['activity'].values.tolist(),
                                mol_env_path='./data/nitrilase_dataset/reaction_emb.lmdb', esm_env_path='./data/nitrilase_dataset/enzyme_emb.lmdb')
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batchsize, collate_fn=collate_substrate)

    model = EnzymaticModel(num_layers=1, hidden_dim=1024, out_dim=512)
    model = model.to(device)
    checkpoint = torch.load(args.ckpt, map_location=torch.device('cpu'))
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()})

    logger.add('./log/nitrilase.log')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    model.to(device)
    model.train()
    for i in range(args.epoch):
        for batch in tqdm(train_dataloader, desc=f'epoch:{i+1}'):
            r_embedding, p_embedding, esm_embedding, r_padding_mask, p_padding_mask, esm_padding_mask, reaction_id, activity, uid = batch
            r_embedding, p_embedding, esm_embedding, r_padding_mask, p_padding_mask, esm_padding_mask = r_embedding.to(device), p_embedding.to(device), esm_embedding.to(device),\
                                                                        r_padding_mask.to(device), p_padding_mask.to(device), esm_padding_mask.to(device)
            reaction_id, activity = reaction_id.to(device),activity.to(device)
            reactant_emb, enzyme_emb, product_emb = model(esm_emb=esm_embedding, reactant=r_embedding, product=p_embedding, 
                                                        esm_padding_mask=esm_padding_mask, reactant_padding_mask=r_padding_mask, product_padding_mask=p_padding_mask)

            activity = torch.where(activity<=2, 0, 1)
            rxn_loss, reaction_loss = ReactionLoss(reactant_emb, enzyme_emb, product_emb, reaction_id, activity,)
            loss = rxn_loss + reaction_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    dist_list = []
    activity_list = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(valid_dataloader):
            r_embedding, p_embedding, esm_embedding, r_padding_mask, p_padding_mask, esm_padding_mask, reaction_id, activity, uid = batch
            r_embedding, p_embedding, esm_embedding, r_padding_mask, p_padding_mask, esm_padding_mask = r_embedding.to(device), p_embedding.to(device), esm_embedding.to(device),\
                                                                        r_padding_mask.to(device), p_padding_mask.to(device), esm_padding_mask.to(device)
            reaction_id, activity = reaction_id.to(device),activity.to(device)
            reactant_emb, enzyme_emb, product_emb = model(esm_emb=esm_embedding, reactant=r_embedding, product=p_embedding, 
                                                        esm_padding_mask=esm_padding_mask, reactant_padding_mask=r_padding_mask, product_padding_mask=p_padding_mask)
            fuse_emb = reactant_emb + enzyme_emb
            dist = torch.linalg.norm((fuse_emb-product_emb), dim=-1)   
            dist_list.append(dist)
            activity_list.append(activity)
    
    dist = torch.cat(dist_list,dim=0).detach().cpu().numpy()
    activity = torch.cat(activity_list,dim=0).detach().cpu().numpy()
    label = [1 if i>2 else 0 for i in activity]
    label = np.array(label)

    max_mcc = 0
    idx = 0
    for i in range(400):
        t = 0.01 * i 
        ll = [1 if i<t else 0 for i in dist]
        ll = np.array(ll)
        mcc = matthews_corrcoef(label, ll)
        if mcc>max_mcc:
            max_mcc = mcc
            idx=i 
    print(f"max mcc:{max_mcc}")
    print(f"thredhold:{idx}") 

    pred_label = np.array([1 if i<0.01*idx else 0 for i in dist])
    acc = np.mean((label==pred_label))
    print(f"acc:{acc}")
    roc_auc = roc_auc_score(label, 1/dist)
    print(f'ROC_AUC:{roc_auc}')
    thred = 0.01*idx
    logger.info(f'Valid Epoch:{args.epoch}, Lr:{args.lr}, ACC:{acc}, MCC:{max_mcc}, ROC_AUC:{roc_auc}')
    print(f'thredhold:{thred}')

    dist_list = []
    activity_list = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            r_embedding, p_embedding, esm_embedding, r_padding_mask, p_padding_mask, esm_padding_mask, reaction_id, activity, uid = batch
            r_embedding, p_embedding, esm_embedding, r_padding_mask, p_padding_mask, esm_padding_mask = r_embedding.to(device), p_embedding.to(device), esm_embedding.to(device),\
                                                                        r_padding_mask.to(device), p_padding_mask.to(device), esm_padding_mask.to(device)
            reaction_id, activity = reaction_id.to(device),activity.to(device)
            reactant_emb, enzyme_emb, product_emb = model(esm_emb=esm_embedding, reactant=r_embedding, product=p_embedding, 
                                                        esm_padding_mask=esm_padding_mask, reactant_padding_mask=r_padding_mask, product_padding_mask=p_padding_mask)
            fuse_emb = reactant_emb + enzyme_emb
            dist = torch.linalg.norm((fuse_emb-product_emb), dim=-1)   
            dist_list.append(dist)
            activity_list.append(activity)
    
    dist = torch.cat(dist_list,dim=0).detach().cpu().numpy()
    activity = torch.cat(activity_list,dim=0).detach().cpu().numpy()
    label = [1 if i>2 else 0 for i in activity]
    label = np.array(label)

    # 使用验证集得到的thredhold
    print(args.ckpt)
    pred_label = np.array([1 if i<thred else 0 for i in dist])
    acc = np.mean((label==pred_label))
    print(f"acc:{acc}")
    mcc = matthews_corrcoef(label, pred_label)
    print(f'MCC:{mcc}')
    roc_auc = roc_auc_score(label, 1/dist)
    print(f'ROC_AUC:{roc_auc}')
    logger.info(f'Test Epoch:{args.epoch}, Lr:{args.lr}, ACC:{acc}, ROC_AUC:{roc_auc}, MCC:{mcc}, Thredhold:{thred}')

def ReactionLoss(reactant_emb, enzyme_emb, product_emb, rxn_label, activity, margin1=12, margin2=3):
    # [reactant+enzyme] <--> [product] large margin
    fuse_emb1 = reactant_emb + enzyme_emb
    dist = torch.cdist(fuse_emb1.double(), product_emb.double(), p=2)
    rxn_label = rxn_label.contiguous().view(-1, 1)
    rxn_mask = torch.eq(rxn_label, rxn_label.T).int().cuda()
    # 由于有负样本的存在，对角线上的负样本需要转为0
    zero_indices = (activity==0).nonzero(as_tuple=True)[0]
    rxn_mask = rxn_mask - torch.diag(1-activity)
    rxn_mask[zero_indices] = 0

    pos_num = rxn_mask.sum(dim=1)
    pos_num = pos_num[pos_num!=0]
    pos = rxn_mask * dist
    pos = pos.sum(dim=1)
    pos = pos[pos!=0] / pos_num

    # label为0的样本，化学反应是正确的，使用小的margin2
    enzyme_mask = torch.diag(1-activity)
    enzyme_neg = enzyme_mask * dist + (1 - enzyme_mask) * margin2
    enzyme_neg = torch.relu(margin2-enzyme_neg)
    enzyme_neg = torch.sum(enzyme_neg) / torch.sum(enzyme_mask)

    # 非对角线为0的样本是替换产物的样本，使用大的margin1
    reaction_mask = torch.eq(rxn_label, rxn_label.T).int().cuda()
    reaction_neg_num = (1 - reaction_mask).sum(dim=1)
    reaction_neg = (1 - reaction_mask) * dist + reaction_mask * margin1
    reaction_neg = torch.relu(margin1 - reaction_neg)
    reaction_neg = reaction_neg.sum(dim=1) / reaction_neg_num

    rxn_loss = pos.mean() + reaction_neg.mean() + enzyme_neg

    # [reactant] <--> [product] small margin
    reaction_dist = torch.linalg.norm((reactant_emb-product_emb), dim=-1)   
    reaction_loss = torch.relu(margin2-reaction_dist).mean()
    return rxn_loss, reaction_loss

if __name__ == "__main__":
    # random split epoch:100, lr:3e-5
    # sequence split epoch:100, lr:4e-5
    # substrate split epoch:100, lr:4e-5
    parser = argparse.ArgumentParser(description='setting')
    parser.add_argument('--task', default='fine-tune')
    parser.add_argument('--lr', default=0.00003, type=float)
    parser.add_argument('--batchsize', type=int, default=256)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--ckpt', default='./ckpt/checkpoint.pt', type=str)
    args = parser.parse_args()
    main(args)