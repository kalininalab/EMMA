import os
from os.path import join
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
sys.path.append("./../utilities")
from utilities.helper_functions import *
from utilities.thresholds import *

# Define the current directory
# https://www.guidetopharmacology.org/download.jsp
CURRENT_DIR = os.getcwd()
print("Current Directory:", CURRENT_DIR)
enzyme_interactions=pd.read_csv(join(CURRENT_DIR,"..","data","raw_data","enzyme_interactions.csv"),skiprows=1)
enzyme_interactions=enzyme_interactions[["Target UniProt ID","Ligand PubChem SID", 'Type', 'Action', 'Affinity Units','Original Affinity Median nm']]
enzyme_interactions.dropna(subset=['Original Affinity Median nm',"Target UniProt ID","Ligand PubChem SID","Action"], inplace=True)
print(enzyme_interactions["Type"].unique().tolist())
enzyme_interactions=enzyme_interactions[enzyme_interactions["Type"].isin(["Inhibitor", "Antagonist"])]
print(enzyme_interactions["Affinity Units"].unique().tolist())
# convert pIC50, pKi,pKd,pEC50 to IC50, Ki,Kd,EC50
# enzyme_interactions['standard_value'] = 10 ** (-enzyme_interactions['Original Affinity Median nm'])
# enzyme_interactions["standard_type"] = enzyme_interactions["Affinity Units"].str.replace(
#     r"^p", "", regex=True)

enzyme_interactions['standard_value'] = enzyme_interactions['Original Affinity Median nm']
enzyme_interactions["standard_type"] = enzyme_interactions["Affinity Units"].str.replace(
    r"^p", "", regex=True)
enzyme_interactions["Ligand PubChem SID"] = enzyme_interactions["Ligand PubChem SID"].astype(int)

# luphar_sids=list(set(enzyme_interactions["Ligand PubChem SID"].unique().tolist()))
# sid_cid_map = sids_to_cids(luphar_sids)
# sid_cid_map_df = pd.DataFrame(list(sid_cid_map.items()), columns=['Compound_SID', 'Compound_CID'])
# sid_cid_map_df.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data","luphar", "luphar_sid2cid.pkl"))

sid_cid_map_df = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data","luphar", "luphar_sid2cid.pkl"))
sid_cid_map_dict = dict(zip(sid_cid_map_df['Compound_SID'], sid_cid_map_df['Compound_CID']))
enzyme_interactions['molecule_ID']= enzyme_interactions['Ligand PubChem SID'].map(sid_cid_map_dict)

# luphar_cids=list(set(enzyme_interactions["molecule_ID"].unique().tolist()))
# cid_to_smiles = pubchem2smiles(luphar_cids)
# cid_to_smiles_df = pd.DataFrame(list(cid_to_smiles.items()), columns=['Compound_CID', 'SMILES'])
# cid_to_smiles_df.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data","luphar", "luphar_cid2smiles.pkl"))

cid_to_smiles_df = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data","luphar", "luphar_cid2smiles.pkl"))
cid_to_smiles_dict = dict(zip(cid_to_smiles_df['Compound_CID'], cid_to_smiles_df['SMILES']))
enzyme_interactions['SMILES']= enzyme_interactions['molecule_ID'].map(cid_to_smiles_dict)

#######################################
# Get sequence
#######################################
# uniprot_IDs = list(set(enzyme_interactions['Target UniProt ID'].unique().tolist()))
# sequences = get_protein_sequences_with_retry(uniprot_IDs)
# sequence_df = pd.DataFrame(list(sequences.items()), columns=['Uniprot_ID', 'Protein_Sequence'])
# sequence_df.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data","luphar", "luphar_proId2seq.pkl"))

sequence_df = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data","luphar", "luphar_proId2seq.pkl"))
enzyme_interactions['Protein_Sequence'] = np.nan
for _, row in sequence_df.iterrows():
    uniprot_id = row['Uniprot_ID']
    sequence = row['Protein_Sequence']
    if pd.notna(sequence):
        enzyme_interactions.loc[enzyme_interactions['Target UniProt ID'] == uniprot_id, 'Protein_Sequence'] = sequence

enzyme_interactions.rename(columns={'Target UniProt ID': 'Uni_SwissProt'}, inplace=True)
enzyme_interactions["EC_ID"] = None
enzyme_interactions["Evidence"]="DBA"
enzyme_interactions["activity_comment"]=None
enzyme_interactions = enzyme_interactions[['Uni_SwissProt', 'molecule_ID', 'Protein_Sequence','standard_type', 'standard_value', 'SMILES',"EC_ID","Evidence","activity_comment"]]

enzyme_interactions.dropna(subset=['Protein_Sequence','SMILES'], inplace=True)
##################################
# Remove small molecules
##################################
# check additional_code folder to see how cofactors_list.txt has been created
with open(join(CURRENT_DIR, "..", "data", "processed_data","cofactors_list.txt"), "r") as f:
    remove_cofactor_energy_ids = [line.strip() for line in f.readlines()]
enzyme_interactions["molecule_ID"] = enzyme_interactions["molecule_ID"].astype(str)
enzyme_interactions = enzyme_interactions.loc[~enzyme_interactions["molecule_ID"].isin(remove_cofactor_energy_ids)]
enzyme_interactions.reset_index(drop=True, inplace=True)


enz_inh = enzyme_interactions[(enzyme_interactions['standard_type'].isin(["IC50","Ki"])) &
                                           (enzyme_interactions['standard_value'].between(0.0001, inh_KiEIc_nM))]

enz_inh.loc[enz_inh['standard_type'] == "IC50", 'standard_value'] *= 0.50
enz_nonsub = enzyme_interactions[(enzyme_interactions['standard_type'].isin(["IC50","Ki",'Kd', 'EC50'])) &
                                      (enzyme_interactions['standard_value'] >= lap_KiEIc_nM)]
enz_nonsub.loc[enz_nonsub['standard_type'] == "IC50", 'standard_value'] *= 0.50


enz_nonsub.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data","luphar", "luphar_enz_nonsub.pkl"))
enz_inh.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data","luphar", "11-luphar_enz_inh.pkl"))
print(data_report(enz_inh))
print(data_report(enz_nonsub))
