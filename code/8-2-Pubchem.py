import requests
import pandas as pd
import sys
import os
import re
import json
from time import sleep
from urllib3.util.retry import Retry
sys.path.append("./../utilities")
from utilities.helper_functions import *
from utilities.thresholds import *

CURRENT_DIR = os.getcwd()
print(CURRENT_DIR)
pubchem = pd.read_csv(join(CURRENT_DIR, "..", "data", "processed_data","Pubchem", "8-1-pubchem_assays_combined.csv"))
print(data_report(pubchem))
pubchem = pubchem.rename(columns={'EC Number': 'EC_ID',"assay_type":"standard_type","Active/Inactive":"activity_comment" })
pubchem = pubchem[pubchem['Assay Description'].str.contains('inhibit|binding|affinity',case=False, na=False)]
pubchem['standard_value'] = pubchem['assay_value_uM'] * 1000
pubchem.loc[pubchem['standard_type'] == "IC50", 'standard_value'] *= 0.50
pubchem_lap=pubchem[pubchem['activity_comment']=='Inactive']
pubchem_lap=pubchem_lap[(pubchem_lap['standard_type'].isin(["Ki",'Kd',"IC50",'EC50'])) & (pubchem_lap['standard_value']>= lap_KiEIc_nM)]
pubchem_inh=pubchem[pubchem['activity_comment']=='Active']
pubchem_inh=pubchem_inh[pubchem_inh['standard_type'].isin(["IC50","Ki"]) & (pubchem_inh['standard_value'].between(0.0001, inh_KiEIc_nM))]
pubchem_inh = pubchem_inh.groupby('UniProt ID', group_keys=False).apply(downsample_pubchem)

print(data_report(pubchem_lap))
print(data_report(pubchem_inh))

# pubchem_sids=list(set(pubchem_inh["Compound_SID"].unique().tolist() + pubchem_lap["Compound_SID"].unique().tolist()))
# print(len(pubchem_sids))
# sid_cid_map = sids_to_cids(pubchem_sids)
# print(sid_cid_map)
# sid_cid_map_df = pd.DataFrame(list(sid_cid_map.items()), columns=['Compound_SID', 'Compound_CID'])
# sid_cid_map_df.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data","Pubchem", "pubchem_sid2cid.pkl"))

sid_cid_map_df = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data","Pubchem", "pubchem_sid2cid.pkl"))
sid_cid_map_dict = dict(zip(sid_cid_map_df['Compound_SID'], sid_cid_map_df['Compound_CID']))
pubchem_lap['Compound_CID']= pubchem_lap['Compound_SID'].map(sid_cid_map_dict)
pubchem_inh['Compound_CID']= pubchem_inh['Compound_SID'].map(sid_cid_map_dict)

# pubchem_cids=list(set(pubchem_inh["Compound_CID"].unique().tolist() + pubchem_lap["Compound_CID"].unique().tolist()))
# cid_to_smiles = pubchem2smiles(pubchem_cids)
# cid_to_smiles_df = pd.DataFrame(list(cid_to_smiles.items()), columns=['Compound_CID', 'SMILES'])
# cid_to_smiles_df.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data","Pubchem", "pubchem_cid2smiles.pkl"))

cid_to_smiles_df = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data","Pubchem", "pubchem_cid2smiles.pkl"))
cid_to_smiles_dict = dict(zip(cid_to_smiles_df['Compound_CID'], cid_to_smiles_df['SMILES']))
pubchem_lap['SMILES']= pubchem_lap['Compound_CID'].map(cid_to_smiles_dict)
pubchem_inh['SMILES']= pubchem_inh['Compound_CID'].map(cid_to_smiles_dict)
#######################################
# Get sequence
#######################################
# uniprot_IDs = list(set(pubchem_inh['UniProt ID'].unique().tolist() + pubchem_lap['UniProt ID'].unique().tolist()))
# sequences = get_protein_sequences_with_retry(uniprot_IDs)
# sequence_df = pd.DataFrame(list(sequences.items()), columns=['Uniprot_ID', 'Protein_Sequence'])
# sequence_df.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data","pubchem", "pubchem_proId2seq.pkl"))

sequence_df = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data","pubchem", "pubchem_proId2seq.pkl"))
pubchem_lap['Protein_Sequence'] = np.nan
pubchem_inh['Protein_Sequence'] = np.nan
for _, row in sequence_df.iterrows():
    uniprot_id = row['Uniprot_ID']
    sequence = row['Protein_Sequence']
    if pd.notna(sequence):
        pubchem_lap.loc[pubchem_lap['UniProt ID'] == uniprot_id, 'Protein_Sequence'] = sequence
        pubchem_inh.loc[pubchem_inh['UniProt ID'] == uniprot_id, 'Protein_Sequence'] = sequence

pubchem_lap.rename(columns={'UniProt ID': 'Uni_SwissProt','Compound_CID':'molecule_ID'}, inplace=True)
pubchem_inh.rename(columns={'UniProt ID': 'Uni_SwissProt','Compound_CID':'molecule_ID'}, inplace=True)
pubchem_lap["Evidence"]="DBA" # DBA stands for Direct Binding Assay
pubchem_inh["Evidence"]="DBA"
pubchem_lap = pubchem_lap[['Uni_SwissProt', 'molecule_ID', 'Protein_Sequence','standard_type', 'standard_value', 'SMILES',"EC_ID", "Evidence","activity_comment"]]
pubchem_inh = pubchem_inh[['Uni_SwissProt', 'molecule_ID', 'Protein_Sequence','standard_type', 'standard_value', 'SMILES',"EC_ID", "Evidence","activity_comment"]]

pubchem_lap.dropna(subset=['Protein_Sequence'], inplace=True)
pubchem_inh.dropna(subset=['Protein_Sequence'], inplace=True)
pubchem_lap.dropna(subset=['SMILES'], inplace=True)
pubchem_inh.dropna(subset=['SMILES'], inplace=True)
##################################
# Remove small molecules
##################################
# check additional_code folder to see how cofactors_list.txt has been created
with open(join(CURRENT_DIR, "..", "data", "processed_data","cofactors_list.txt"), "r") as f:
    remove_cofactor_energy_ids = [line.strip() for line in f.readlines()]
pubchem_lap["molecule_ID"] = pubchem_lap["molecule_ID"].astype(str)
pubchem_inh["molecule_ID"] = pubchem_inh["molecule_ID"].astype(str)
pubchem_lap = pubchem_lap.loc[~pubchem_lap["molecule_ID"].isin(remove_cofactor_energy_ids)]
pubchem_inh = pubchem_inh.loc[~pubchem_inh["molecule_ID"].isin(remove_cofactor_energy_ids)]
pubchem_lap.reset_index(drop=True, inplace=True)
pubchem_inh.reset_index(drop=True, inplace=True)
pubchem_lap.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data","Pubchem", "8-2-pubchem_lap.pkl"))
pubchem_inh.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data","Pubchem", "8-2-pubchem_enz_inh.pkl"))
print(data_report(pubchem_lap))
print(data_report(pubchem_inh))