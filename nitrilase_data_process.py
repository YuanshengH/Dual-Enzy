import os
import lmdb
import torch
import pickle
import esm
import pandas as pd
from unimol_tools import UniMolRepr

df = pd.read_excel('./data/nitrilase_dataset/nitrilase_descriptors.xlsx', sheet_name = 'neu50').copy()
df = pd.DataFrame(data= {"enzyme" : df["protein"], "metabolite" : df["nitriles"],
                        "activity" : df["activity"]})

# sequence
seq_3a2 = 'MTQHETTARRLAAVHAAPVFMDTDATIDKVIGFVEQAGREGIELLVFPETFVPGYPYWIECYPPLQQVAANAQYTDASVEVPGPEIKRVQAACARAGVEVVLGVSERLRGTRTCFNSQVFIDADGSLLGVHRKLQPTYVERIVWAQGGGATLSVFGSRSGRIGGLACWEHTMNLARQALLEQEQQIHAAAWPALSTMAGFETVADAQIEAMMKTHALTAQVFVICASNPVDGTCLEWMRDNLGEQKFVTAGGGWSAVIHPFNSFLGGPHTGLEEKLVSATIDFSDIRLVKAWVDSKGHYARPEVLRLAVDRKPLWHDECEVPGQAQVRTRAASLAVQEHPVVLPQGAARPAPQDWDTSAAQELTS'
seq_2a6 = 'MKEAIKVACVQAAPIYMDLEATVDKTIELMEEAARNNARLIAFPETWIPGYPWFLWLDSPAWAMQFVRQYHENSLELDGPQAKRISDAAKRLGIMVTLGMSERVGGTLYISQWFIGDNGDTIGARRKLKPTFVERTLFGEGDGSSLAVFETSVGRLGGLCCWEHLQPLTKYALYAQNEEIHCAAWPSFSLYPNAAKALGPDVNVAASRIYAVEGQCFVLASCALVSQSMIDMLCTDDEKHALLLAGGGHSRIIGPDGGDLVAPLAENEEGILYANLDPGVRILAKMAADPAGHYSRPDITRLLIDRSPKLPVVEIEGDLRPYALGKASETGAQLEEI'
seq_1b15 = 'MPKTVRAAAVQIAPDLTSRAGTVERVLNAIAEASDKGAELIVFPETFVPWYPYFSFVLPPVQQGPEHLRLYEEAVTVPSAETRAVADAARKRNAVIVLGVNERDHGSLYNTQLIFDADGSLKLKRRKITPTYHERMIWGQGDGAGLKVVETAIGRMGALACWEHYNPLARYALMAQHEEIHASHFPGSLVGPIFGEQIEVTMRHHALESGCFVVNATGWLSEEQIASIHPDPSLQKGLRDGCMTCIITPEGRHVVPPLTSGEGILIGDLDMRLITKRKRMMDSVGHYARPELLHLVHDTTPARAREQVGLSGDFSDAGQDKLFEEVQDA'
seq_1b16 = 'MNNSPPTIRAAAIQLSPVLFSRDGTTEKVLQAIASAAKEGAQLVVFPETFIPYYPYFSFIQPPVLMGKEHMRLYEEAVTVPGPVTDAVSRAARSYGMVVVLGVNERDGGSIYNTQLIFDADGTLLLKRRKITPTYHERMVWGQGDGAGLKVLDTAVGKVGALACWEHYNPLARFALMAQHEQIHCAQFPGSLVGQIFTDQIEVTIRHHALESGCFVVNATGWLSPEQVAQITTDEKLQRVLSGGCNTAIIGPEGNHLCPPITDGEGIAIADLDFSLITKRKRMMDCVGHYSRPDLLKLQLNATAWSVLAGEQGAGAREQGLGVPDAMLSTPKPEYSTLDQ'
seq_3wuy = 'GSHMLGKIMLNYTKNIRAAAAQISPVLFSQQGTMEKVLDAIANAAKKGVELIVFPETFVPYYPYFSFVEPPVLMGKSHLKLYQEAVTVPGKVTQAIAQAAKTHGMVVVLGVNEREEGSLYNTQLIFDADGALVLKRRKITPTYHERMVWGQGDGAGLRTVDTTVGRLGALACWEHYNPLARYALMAQHEQIHCGQFPGSMVGQIFADQMEVTMRHHALESGCFVINATGWLTAEQKLQITTDEKMHQALSGGCYTAIISPEGKHLCEPIAEGEGLAIADLDFSLIAKRKRMMDSVGHYARPDLLQLTLNNQPWSALEANPVTPNAIPAVSDPELTETIEALPNNPIFSH'
seq_1a27 = 'MAIIRAAAVQISPVLYSREGTVDKVCQQIITLGKQGVQFAVFPETVVPYYPYFSFVQPAFAMGAQHLKLLDQSVTVPSAATLAIGEACKQAGMVVSIGVNERDGGTIYNAQLLFDADGTLIQHRRKITPTYHERMVWGQGDGSGLRAIDSAVGRIGSLACWEHYNPLARYALMADGEQIHAAMFPGSLVGDIFAEQIEVTIRHHALESGCFVVNATAWLDADQQGQIMQDTGCGLGPISGGCFTAIVSPEGKLLGEPLRSGEGVVIADLDTALIDKRKRMMDSVGHYSRPELLSLLIDRSPTAHVHERAGFVSSNAGLQEVAHADQ'
seq_1a17 = 'MKNIKNSEKSSTVRVAAVQISPVLYNREATVQKVVNKILELGKQGVQFATFPETIVPYYPYFSFIQAPYAMGKEHLRLLEQSVTVPSAATDAISEAAKEANMVVSIGVNERDGGTIYNTQLLFDADGTLIQRRRKLTPTYHERMIWGQGDASGLRATDSAVGRIGQLACWEHYNPLFRYALIADGEQIHSAMYPGSFLGALHGEQTEINVRQHALESASFVVVATGWLDADQQAQIAKDTGGPIGPISGGCFTAVIGPDGQLIGEALTSGEGEVIADIDLAQIDARKRLMDASGHYNRPELLSLHIDHTPTAPMHERVVYTEPGLAKRQNENSSN'
seq_1a8 = 'MKVVKAAAVQISPVLYSREATVAKVVQKIHELGQKGVQFATFPETVVPYYPYFSAVQTGIELLSGTEHLRLLDQAVTVPSAATDAIGEAARKAGMVVSIGVNERDGGTLYNTQLLFDADGTLIQRRRKITPTHFERMIWGQGDGSGLRAVDSKVGRIGQLACFEHNNPLARYALIADGEQIHSAMYPGSAFGEGFAQRMEINIRQHALESGAFVVNATAWLDADQQAQIMKDTGCGIGPISGGCFTTIVSPDGMLMAEPLRSGEGEVIVDLDFTLIDRRKMLMDSAGHYNRPELLSLMIDRTPTAHVHERAARPVSGVEQNPEELRIPAA'
seq_1a2 = 'MKVVKAAAVQLSPVLYSREGTVERVVRKIHELGRQGVQFATFPETVVPYYPYFSFVQTPLQIIAGPEHLKLLDQAVTVPSPATDAISEAARQAGVVVSIGVNERDGGTLYNTQLLFDADGALIQRRRKITPTHFERMIWGQGDGSGLRAVDSKVGRIGQLACWEHNNPLARYAMIADGEQIHSAMYPGSMFGDPFAQKTEINIRQHALESACFVVCATAWLDADQQAQICKDTGCDIGPISGGCFTAIVAPDGTLLGEPIRSGEGMVIVDLDFTLIDKRKQVMDSRGHYNRPELLSLLIDRTPTAHVHDRAVRPESAAEQRSEELLATAV'
seq_1a1 = 'MSTIVKAAAVQISPVLYSREGTVERVVKKIRELGEKGVQFATFPETVIPYYPYFSFVQTPLQILAGPEHLKLLDQSVTVPSPATDAIGQAARQAGMVVSIGVNERDGGTLYNTQLLFDADGALIQRRRKIKPTHYERMIWGEGDGSGLRAVDSQVGRIGQLACWEHNNPLARYAMMADGEQIHSAMYPGSMFGDPFAQKTEINIRQHALESGCFVVCSTAWLDADQQAQIMQDTGCAIGPISGGCLTAIVAPDGTFLGEPLTSGEGEVIADLDFKLIDKRKQTMDSRGHYNRPELLSLLIDRTPTSNVHERAAHPKVEASQTAGDTERTREVL'
seq_pmi26 = 'MAIIRAAAVQISPVLYSREGTVEKVCQQIIALGRQGVQFAVFPETVVPYYPYFSFVQPPFAMGAQHLKLLDQSVTVPSAATLAIGEACKQVGIVVSIGVNERDGGTIYNAQLLFDADGSLIQHRRKITPTYHERMVWGQGDGSGLRAIDSAVGRIGSLACWEHYNPLARYALMADGEQIHAAMFPGSLVGDIFAEQIEVTIRHHALESGCFVVNATAWLDADQQGQIMQDTGCGLGPISGGCFTAIVSPEGKLLGEPLRSGEGVVIADLDLALIDKRKRMMDSVGHYSRPELLSLLIDRTPTAHVHERGGHFASVADEESDHANQ'
seq_pmi28 = 'MKIVKAAAVQISPVLYNREATVEKVVQKILELGQQGVQFATFPETVVPYYPYFSAVQTPMQLLAGTEYLKLLEQAVTVPSPATDAIGEAARKAGMVVSIGVNERDGGTLYNTQLLFDADGTLIQRRRKITPTHFERMIWGQGDGSGLCAVDSKIGRIGQLACFEHNNPLARYALIADGEQIHSAMYPGSAFGEGFAQRMEINIRQHALESGCFVVNATAWLDADQQAQIMKDTGCEIGPISGGCFTAIVAPDGMLLGEPLRSGEGEVIADLDFTLIDRRKMLMDSAGHYNRPELLSLLIDRTPTAHIHERTGQAATSAEQGADEPRATTVL'
sequence_dict = {'3a2':seq_3a2,'2a6':seq_2a6,'1b15':seq_1b15,'1b16':seq_1b16,'3wuy':seq_3wuy,'1a27':seq_1a27,'1a17':seq_1a17,'1a8':seq_1a8,'1a1':seq_1a1,'1a2':seq_1a2,'pmi26':seq_pmi26,'pmi28':seq_pmi28}

# reaction
smi_4_Hydroxybenzonitrile = 'N#Cc1ccc(O)cc1.O.O>>Oc1ccc(C([O-])=O)cc1.[NH4+]'
smi_4_Methoxybenzonitrile = 'COc1ccc(C#N)cc1.O.O>>COc1ccc(C([O-])=O)cc1.[NH4+]'
smi_Valeronitrile = 'CCCCC#N.O.O>>CCCCC([O-])=O.[NH4+]'
smi_4_Nitrophenylacetonitrile = 'N#CCc1ccc([N+](=O)[O-])cc1.O.O>>O=[N+]([O-])c1ccc(CC([O-])=O)cc1.[NH4+]'
smi_2_Aminobenzonitrile = 'N#Cc1ccccc1N.O.O>>Nc1c(C([O-])=O)cccc1.[NH4+]'
smi_Cinnamonitrile = 'N#C/C=C/c1ccccc1.O.O>>[O-]C(/C=C/c1ccccc1)=O.[NH4+]'
smi_alpha_methylbenzylcyanide = 'C[C@@H](C#N)c1ccccc1.O.O>>C[C@H](c1ccccc1)C([O-])=O.[NH4+]'
smi_2_6_dichlorobenzonitrile = 'N#Cc1c(Cl)cccc1Cl.O.O>>Clc1cccc(Cl)c1C([O-])=O.[NH4+]'
smi_Isophthalonitrile = 'N#Cc1cccc(C#N)c1.O.O.O.O>>[O-]C(c1cc(C([O-])=O)ccc1)=O.[NH4+].[NH4+]'
smi_4_Dimethylamino_benzonitrile = 'CN(C)c1ccc(C#N)cc1.O.O>>CN(c1ccc(C([O-])=O)cc1)C.[NH4+]'
smi_Acrylonitrile = 'C=CC#N.O.O>>C=CC([O-])=O.[NH4+]'
smi_4_Aminobenzonitrile = 'N#Cc1ccc(N)cc1.O.O>>Nc1ccc(C([O-])=O)cc1.[NH4+]'
smi_Benzoylacetonitrile = 'N#CCC(=O)c1ccccc1.O.O>>O=C(c1ccccc1)CC([O-])=O.[NH4+]'
smi_Cyclohexancarbonitrile = 'N#CC1CCCCC1.O.O>>[O-]C(C1CCCCC1)=O.[NH4+]'
smi_4_Chlorobenzonitrile = 'N#Cc1ccc(Cl)cc1.O.O>>Clc1ccc(C([O-])=O)cc1.[NH4+]'
smi_3_Phenylpropionitrile = 'N#CCCc1ccccc1.O.O>>[O-]C(CCc1ccccc1)=O.[NH4+]'
smi_Mandelonitrile = 'N#C[C@H](O)c1ccccc1.O.O>>O[C@H](c1ccccc1)C([O-])=O.[NH4+]'
smi_Benzonitrile = 'N#Cc1ccccc1.O.O>>[O-]C(c1ccccc1)=O.[NH4+]'
smi_2_Methylglutaronitrile = 'C[C@@H](C#N)CCC#N.O.O.O.O>>C[C@H](CCC([O-])=O)C([O-])=O.[NH4+].[NH4+]'
smi_Propionitrile = 'CCC#N.O.O>>CCC([O-])=O.[NH4+]'
reaction_dict = {'Propionitrile':smi_Propionitrile,'2,6-Dichlorobenzonitrile':smi_2_6_dichlorobenzonitrile,'4-(Dimethylamino)benzonitrile':smi_4_Dimethylamino_benzonitrile,'4-Methoxybenzonitrile':smi_4_Methoxybenzonitrile, 
'4-Chlorobenzonitrile':smi_4_Chlorobenzonitrile,'4-Hydroxybenzonitrile':smi_4_Hydroxybenzonitrile,'Isophthalonitrile':smi_Isophthalonitrile,'Cyclohexancarbonitrile':smi_Cyclohexancarbonitrile,'Acrylonitrile':smi_Acrylonitrile,
'Valeronitrile':smi_Valeronitrile,'Cinnamonitrile':smi_Cinnamonitrile,'2-Methylglutaronitrile':smi_2_Methylglutaronitrile,'2-Aminobenzonitrile':smi_2_Aminobenzonitrile,'Benzonitrile':smi_Benzonitrile,'Benzoylacetonitrile':smi_Benzoylacetonitrile,
'3-Phenylpropionitrile':smi_3_Phenylpropionitrile,'Mandelonitrile':smi_Mandelonitrile,'4-Nitrophenylacetonitrile':smi_4_Nitrophenylacetonitrile,'4-Aminobenzonitrile':smi_4_Aminobenzonitrile,'⍺-Methylbenzyl cyanide':smi_alpha_methylbenzylcyanide}

df['Sequence'] = df['enzyme'].apply(lambda x: sequence_dict[x])
df['Reaction'] = df['metabolite'].apply(lambda x: reaction_dict[x])
df.to_csv('./data/nitrilase_dataset/nitrilase_data.csv', index=False)

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
