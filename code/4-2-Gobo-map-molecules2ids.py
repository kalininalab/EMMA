import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.append("./../utilities")
import pandas as pd
import os
import pickle
from os.path import join, exists
import math
from utilities.helper_functions import *

CURRENT_DIR = os.getcwd()
print(CURRENT_DIR)

df_obo_uniprot = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data", "GO", "4-1-gobo_enz_sub.pkl"))
#######################################
# Pubchem
#######################################
# sub_unique = df_obo_uniprot['Substrate'].unique().tolist()
# pubID_dict = get_pubchem_ids(sub_unique)
# cid_to_name_df = pd.DataFrame(list(pubID_dict.items()), columns=['Substrate', 'PubChem_ID'])
# cid_to_name_df.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data" ,"GO", "gobo_cid2name.pkl"))

cid_to_name_df = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data", "GO", "gobo_cid2name.pkl"))
cid_to_name_dict = dict(zip(cid_to_name_df['Substrate'], cid_to_name_df['PubChem_ID']))
df_obo_uniprot['molecule_ID'] = df_obo_uniprot['Substrate'].map(cid_to_name_dict)
print(data_report(df_obo_uniprot))

df_obo_uniprot["Substrate"] = [name.lower() for name in df_obo_uniprot["Substrate"]]
mol2ids = pd.read_pickle(join(CURRENT_DIR, "..", "data", "raw_data", "molecule_name_to_ids", "mol2ids.pkl"))
mol2ids_dict = dict(zip(mol2ids['molecule_name'], mol2ids['molecule_ID']))
df_obo_uniprot['molecule_ID'].fillna(df_obo_uniprot['Substrate'].map(mol2ids_dict), inplace=True)
print(data_report(df_obo_uniprot))
##################################
# Remove small molecules
##################################
# check additional_code folder to see how cofactors_list.txt has been created
with open(join(CURRENT_DIR, "..", "data", "processed_data", "cofactors_list.txt"), "r") as f:
    remove_cofactor_energy_ids = [line.strip() for line in f.readlines()]
df_obo_uniprot["molecule_ID"] = df_obo_uniprot["molecule_ID"].astype(str)
df_obo_uniprot = df_obo_uniprot.loc[~df_obo_uniprot["molecule_ID"].isin(remove_cofactor_energy_ids)]
df_obo_uniprot.dropna(subset=['molecule_ID'], inplace=True)
df_obo_uniprot.reset_index(drop=True, inplace=True)
print(data_report(df_obo_uniprot))

#######################################
# Get sequence
#######################################
# uniprot_IDs = df_obo_uniprot['Uniprot_ID'].unique()
# sequences = get_protein_sequences_with_retry(uniprot_IDs)
# sequence_df = pd.DataFrame(list(sequences.items()), columns=['UniProt_ID', 'Protein_Sequence'])
# sequence_df.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data","GO", "gobo_proId2seq.pkl"))

sequence_df = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data", "GO", "gobo_proId2seq.pkl"))
df_obo_uniprot['Protein_Sequence'] = np.nan
for _, row in sequence_df.iterrows():
    uniprot_id = row['UniProt_ID']
    sequence = row['Protein_Sequence']
    if pd.notna(sequence):
        df_obo_uniprot.loc[df_obo_uniprot['Uniprot_ID'] == uniprot_id, 'Protein_Sequence'] = sequence

df_obo_uniprot.rename(columns={'Uniprot_ID': 'Uni_SwissProt'}, inplace=True)
df_obo_uniprot.dropna(subset=['Uni_SwissProt'], inplace=True)
df_obo_uniprot.reset_index(drop=True, inplace=True)
df_obo_uniprot.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data", "GO", "4-2-gobo_enz_sub.pkl"))
print(data_report(df_obo_uniprot))
