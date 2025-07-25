import requests
import pandas as pd
import os
import sys
from os.path import join

sys.path.append("./../utilities")
from utilities.helper_functions import *

CURRENT_DIR = os.getcwd()
print(CURRENT_DIR)

uniprot_enz_sub = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data","uniprot", "2-1-uniprot_enz_sub.pkl"))
#######################################
# get Pubchem ID
#######################################
# unique_name = uniprot_enz_sub['Substrate'].unique().tolist()
# pubID_dict = get_pubchem_ids(unique_name)
# name_to_cid_df = pd.DataFrame(list(pubID_dict.items()), columns=['molecule_name', 'ID'])
# name_to_cid_df.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data","uniprot", "UniAct_MolName2cid.pkl"))

name_to_cid_df = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data","uniprot", "UniAct_MolName2cid.pkl"))
pubID_dict = dict(zip(name_to_cid_df['molecule_name'], name_to_cid_df['ID']))
uniprot_enz_sub['molecule_ID'] = uniprot_enz_sub['Substrate'].map(pubID_dict)
uniprot_enz_sub.reset_index(drop=True, inplace=True)
print(data_report(uniprot_enz_sub))

mol2ids = pd.read_pickle(join(CURRENT_DIR, "..", "data", "raw_data","molecule_name_to_ids", "mol2ids.pkl"))
uniprot_enz_sub["Substrate"] = [name.lower() for name in uniprot_enz_sub["Substrate"]]
mol2ids_dict = dict(zip(mol2ids['molecule_name'], mol2ids['molecule_ID']))
uniprot_enz_sub['molecule_ID'].fillna(uniprot_enz_sub['Substrate'].map(mol2ids_dict), inplace=True)
uniprot_enz_sub.dropna(subset=['molecule_ID'], inplace=True)
print(data_report(uniprot_enz_sub))
##############################
# separate different IDS
##############################
unique_molecule_ids = uniprot_enz_sub['molecule_ID'].dropna().unique()
# Initialize lists for each identifier type
chebi_ids = []
pubchem_ids = []
kegg_ids = []

# Categorize each identifier
for molecule_id in unique_molecule_ids:
    molecule_id=str(molecule_id)
    if re.match(r'^CHEBI:', molecule_id):
        chebi_ids.append(molecule_id)
    elif re.match(r'^\d+$', molecule_id):
        pubchem_ids.append(molecule_id)
    elif re.match(r'^C\d+', molecule_id):
        kegg_ids.append(molecule_id)
##############################
# Map ChEBI ID to SMILES
##############################
# dict_chebi2smiles = chebi2smiles(chebi_ids)
# chebi_to_smiles_df = pd.DataFrame(list(dict_chebi2smiles.items()), columns=['ChEBI_ID', 'SMILES'])
# chebi_to_smiles_df.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data","uniprot", "UniAct_chebi2smiles.pkl"))

chebi_to_smiles_df = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data","uniprot", "UniAct_chebi2smiles.pkl"))
dict_chebi2smiles = dict(zip(chebi_to_smiles_df['ChEBI_ID'], chebi_to_smiles_df['SMILES']))
uniprot_enz_sub['SMILES'] = uniprot_enz_sub['molecule_ID'].map(dict_chebi2smiles)
print(data_report(uniprot_enz_sub))
##############################
# Map PubChem ID  to SMILES
##############################
cid_to_smiles = pubchem2smiles(pubchem_ids)
cid_to_smiles_df = pd.DataFrame(list(cid_to_smiles.items()), columns=['PubChem_ID', 'SMILES'])
cid_to_smiles_df.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data","uniprot", "UniAct_cid2smiles.pkl"))

cid_to_smiles_df = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data","uniprot", "UniAct_cid2smiles.pkl"))
cid_to_smiles = dict(zip(cid_to_smiles_df['PubChem_ID'], cid_to_smiles_df['SMILES']))
uniprot_enz_sub['SMILES'].fillna(uniprot_enz_sub['molecule_ID'].map(cid_to_smiles), inplace=True)
print(data_report(uniprot_enz_sub))

##############################
# Map KEGG ID  to SMILES
##############################
dict_kegg2smiles = kegg2smiles(kegg_ids)
kegg_to_smiles_df = pd.DataFrame(list(dict_kegg2smiles.items()), columns=['KEGG', 'SMILES'])
kegg_to_smiles_df.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data","uniprot", "UniAct_kegg2smiles.pkl"))

kegg_to_smiles_df = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data","uniprot", "UniAct_kegg2smiles.pkl"))
dict_kegg2smiles = dict(zip(kegg_to_smiles_df['KEGG'], kegg_to_smiles_df['SMILES']))
uniprot_enz_sub['SMILES'].fillna(uniprot_enz_sub['molecule_ID'].map(cid_to_smiles), inplace=True)
uniprot_enz_sub.dropna(subset=['SMILES'], inplace=True)
uniprot_enz_sub.reset_index(drop=True, inplace=True)
print(data_report(uniprot_enz_sub))

uniprot_enz_sub.rename(columns={'ID': 'Uni_SwissProt', 'EC Number': 'EC_ID'}, inplace=True)

# check additional_code folder to see how cofactors_list.txt has been created
with open(join(CURRENT_DIR, "..", "data", "processed_data","cofactors_list.txt"), "r") as f:
    remove_cofactor_energy_ids = [line.strip() for line in f.readlines()]
uniprot_enz_sub["molecule_ID"] = uniprot_enz_sub["molecule_ID"].astype(str)
uniprot_enz_sub["Evidence"]="EEC" # EEC stand for Experimental Evidence Code
uniprot_enz_sub["activity_comment"]="CAS" # CAS stand for Classified As Substrate
uniprot_enz_sub = uniprot_enz_sub.loc[~uniprot_enz_sub["molecule_ID"].isin(remove_cofactor_energy_ids)]
uniprot_enz_sub = uniprot_enz_sub[['Uni_SwissProt', 'molecule_ID', 'Protein_Sequence', 'SMILES', 'EC_ID', 'Evidence', 'activity_comment']]
uniprot_enz_sub.reset_index(drop=True, inplace=True)
uniprot_enz_sub.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data","uniprot", "2-2-uniprot_enz-sub.pkl"))
print(data_report(uniprot_enz_sub))
