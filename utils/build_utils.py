import random
import torch
import numpy as np
import pandas as pd
from Bio import SeqIO

def same_seed(seed): 
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class Feature_Bank():    
    def __init__(self, num_class, m=0.9):        
        self.m = m        
        self.num_class = num_class              
        self.feats = [[] for _ in range(num_class)]     

    def init_feats(self, feats, lbs):              
        for i in range(self.num_class):            
            self.feats[i] = np.mean(feats[np.where(lbs==i)[0], :], axis=0)        
        self.feats = np.stack(self.feats, axis=0)
        self.feats = np.nan_to_num(self.feats, nan=0)

    def update_a_batch(self, batch_feats, lbs):        
        for i, lb in enumerate(lbs):                          
            self.feats[lb] = self.m * self.feats[lb] + (1-self.m) * batch_feats[i]

    def get_feats(self):        
        return self.feats