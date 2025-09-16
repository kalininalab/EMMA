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
pubchem = pd.read_csv(join(CURRENT_DIR, "..", "data", "processed_data", "Pubchem", "8-1-pubchem_assays_combined.csv"))
print(data_report(pubchem))
pubchem = pubchem.rename(
    columns={'EC Number': 'EC_ID', "assay_type": "standard_type", "Active/Inactive": "activity_comment"})
pubchem = pubchem[pubchem['Assay Description'].str.contains('inhibit|binding|affinity', case=False, na=False)]
pubchem['standard_value'] = pubchem['assay_value_uM'] * 1000
pubchem.loc[pubchem['standard_type'] == "IC50", 'standard_value'] *= 0.50
pubchem_inactive = pubchem[pubchem['activity_comment'] == 'Inactive']
pubchem_enz_ni = pubchem_inactive[(pubchem_inactive['standard_type'].isin(["Ki", 'Kd', "IC50", 'EC50'])) & (
            pubchem_inactive['standard_value'] >= EnNi_KiECKdIC_nM)]
pubchem_enz_inh = pubchem[pubchem['activity_comment'] == 'Active']
pubchem_enz_inh = pubchem_enz_inh[
    pubchem_enz_inh['standard_type'].isin(["IC50", "Ki"]) &
    (pubchem_enz_inh['standard_value'] > 0) &
    (pubchem_enz_inh['standard_value'] <= EnInh_KiIcMic_nM)
]
pubchem_enz_inh = pubchem_enz_inh.groupby('UniProt ID', group_keys=False).apply(downsample_pubchem)

print(data_report(pubchem_enz_ni))
print(data_report(pubchem_enz_inh))

# pubchem_sids=list(set(pubchem_enz_inh["Compound_SID"].unique().tolist() + pubchem_enz_ni["Compound_SID"].unique().tolist()))
# print(len(pubchem_sids))
# sid_cid_map = sids_to_cids(pubchem_sids)
# print(sid_cid_map)
# sid_cid_map_df = pd.DataFrame(list(sid_cid_map.items()), columns=['Compound_SID', 'Compound_CID'])
# sid_cid_map_df.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data","Pubchem", "pubchem_sid2cid.pkl"))

sid_cid_map_df = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data", "Pubchem", "pubchem_sid2cid.pkl"))
sid_cid_map_dict = dict(zip(sid_cid_map_df['Compound_SID'], sid_cid_map_df['Compound_CID']))
pubchem_enz_ni['Compound_CID'] = pubchem_enz_ni['Compound_SID'].map(sid_cid_map_dict)
pubchem_enz_inh['Compound_CID'] = pubchem_enz_inh['Compound_SID'].map(sid_cid_map_dict)

# pubchem_cids=list(set(pubchem_enz_inh["Compound_CID"].unique().tolist() + pubchem_enz_ni["Compound_CID"].unique().tolist()))
# cid_to_smiles = pubchem2smiles(pubchem_cids)
# cid_to_smiles_df = pd.DataFrame(list(cid_to_smiles.items()), columns=['Compound_CID', 'SMILES'])
# cid_to_smiles_df.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data","Pubchem", "pubchem_cid2smiles.pkl"))

cid_to_smiles_df = pd.read_pickle(
    join(CURRENT_DIR, "..", "data", "processed_data", "Pubchem", "pubchem_cid2smiles.pkl"))
cid_to_smiles_dict = dict(zip(cid_to_smiles_df['Compound_CID'], cid_to_smiles_df['SMILES']))
pubchem_enz_ni['SMILES'] = pubchem_enz_ni['Compound_CID'].map(cid_to_smiles_dict)
pubchem_enz_inh['SMILES'] = pubchem_enz_inh['Compound_CID'].map(cid_to_smiles_dict)
#######################################
# Get sequence
#######################################
# uniprot_IDs = list(set(pubchem_enz_inh['UniProt ID'].unique().tolist() + pubchem_enz_ni['UniProt ID'].unique().tolist()))
# sequences = get_protein_sequences_with_retry(uniprot_IDs)
# sequence_df = pd.DataFrame(list(sequences.items()), columns=['Uniprot_ID', 'Protein_Sequence'])
# sequence_df.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data","pubchem", "pubchem_proId2seq.pkl"))

sequence_df = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data", "pubchem", "pubchem_proId2seq.pkl"))
pubchem_enz_ni['Protein_Sequence'] = np.nan
pubchem_enz_inh['Protein_Sequence'] = np.nan
for _, row in sequence_df.iterrows():
    uniprot_id = row['Uniprot_ID']
    sequence = row['Protein_Sequence']
    if pd.notna(sequence):
        pubchem_enz_ni.loc[pubchem_enz_ni['UniProt ID'] == uniprot_id, 'Protein_Sequence'] = sequence
        pubchem_enz_inh.loc[pubchem_enz_inh['UniProt ID'] == uniprot_id, 'Protein_Sequence'] = sequence

pubchem_enz_ni.rename(columns={'UniProt ID': 'Uni_SwissProt', 'Compound_CID': 'molecule_ID'}, inplace=True)
pubchem_enz_inh.rename(columns={'UniProt ID': 'Uni_SwissProt', 'Compound_CID': 'molecule_ID'}, inplace=True)
pubchem_enz_ni["Evidence"] = "BAA"
pubchem_enz_inh["Evidence"] = "BAA"
pubchem_enz_ni = pubchem_enz_ni[
    ['Uni_SwissProt', 'molecule_ID', 'Protein_Sequence', 'standard_type', 'standard_value', 'SMILES', "EC_ID",
     "Evidence", "activity_comment"]]
pubchem_enz_inh = pubchem_enz_inh[
    ['Uni_SwissProt', 'molecule_ID', 'Protein_Sequence', 'standard_type', 'standard_value', 'SMILES', "EC_ID",
     "Evidence", "activity_comment"]]

pubchem_enz_ni.dropna(subset=['Protein_Sequence'], inplace=True)
pubchem_enz_inh.dropna(subset=['Protein_Sequence'], inplace=True)
pubchem_enz_ni.dropna(subset=['SMILES'], inplace=True)
pubchem_enz_inh.dropna(subset=['SMILES'], inplace=True)
##################################
# Remove small molecules
##################################
# check additional_code folder to see how cofactors_list.txt has been created
with open(join(CURRENT_DIR, "..", "data", "processed_data", "cofactors_list.txt"), "r") as f:
    remove_cofactor_energy_ids = [line.strip() for line in f.readlines()]
pubchem_enz_ni["molecule_ID"] = pubchem_enz_ni["molecule_ID"].astype(str)
pubchem_enz_inh["molecule_ID"] = pubchem_enz_inh["molecule_ID"].astype(str)
pubchem_enz_ni = pubchem_enz_ni.loc[~pubchem_enz_ni["molecule_ID"].isin(remove_cofactor_energy_ids)]
pubchem_enz_inh = pubchem_enz_inh.loc[~pubchem_enz_inh["molecule_ID"].isin(remove_cofactor_energy_ids)]
pubchem_enz_ni.reset_index(drop=True, inplace=True)
pubchem_enz_inh.reset_index(drop=True, inplace=True)
pubchem_enz_ni.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data", "Pubchem", "8-2-pubchem_enz_ni.pkl"))
pubchem_enz_inh.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data", "Pubchem", "8-2-pubchem_enz_inh.pkl"))
print(data_report(pubchem_enz_ni))
print(data_report(pubchem_enz_inh))
