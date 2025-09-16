import pandas as pd
import os
from math import ceil
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.append("./../utilities")
import re
import ast
from os.path import join
from helper_functions import *

CURRENT_DIR = os.getcwd()
print(CURRENT_DIR)
##############################
# Load data
##############################
gobo_enz_sub = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data", "GO", "4-2-gobo_enz_sub.pkl"))
# Combine the molecule_ID columns
molecule_ids = gobo_enz_sub['molecule_ID'].dropna().unique()

# Initialize lists for each identifier type
chebi_ids = []
pubchem_ids = []
kegg_ids = []

# Categorize each identifier
for molecule_id in molecule_ids:
    molecule_id = str(molecule_id)
    if re.match(r'^CHEBI:', molecule_id):
        chebi_ids.append(molecule_id)
    elif re.match(r'^\d+$', molecule_id):
        pubchem_ids.append(molecule_id)
    elif re.match(r'^C\d+', molecule_id):
        kegg_ids.append(molecule_id)
##############################
# Map KEGG ID  to SMILES
##############################
# dict_kegg2smiles = kegg2smiles(kegg_ids)
# kegg_to_smiles_df = pd.DataFrame(list(dict_kegg2smiles.items()), columns=['KEGG', 'SMILES'])
# kegg_to_smiles_df.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data","GO", "gobo_kegg2smiles.pkl"))

kegg_to_smiles_df = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data", "GO", "gobo_kegg2smiles.pkl"))
dict_kegg2smiles = dict(zip(kegg_to_smiles_df['KEGG'], kegg_to_smiles_df['SMILES']))
gobo_enz_sub['SMILES'] = gobo_enz_sub['molecule_ID'].map(dict_kegg2smiles)
print(data_report(gobo_enz_sub))
##############################
# Map ChEBI ID  to SMILES
##############################
# dict_chebi2smiles = chebi2smiles(chebi_ids)
# chebi_to_smiles_df = pd.DataFrame(list(dict_chebi2smiles.items()), columns=['ChEBI_ID', 'SMILES'])
# chebi_to_smiles_df.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data","GO", "gobo_chebi2smiles.pkl"))

chebi_to_smiles_df = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data", "GO", "gobo_chebi2smiles.pkl"))
dict_chebi2smiles = dict(zip(chebi_to_smiles_df['ChEBI_ID'], chebi_to_smiles_df['SMILES']))
gobo_enz_sub['SMILES'].fillna(gobo_enz_sub['molecule_ID'].map(dict_chebi2smiles), inplace=True)
print(data_report(gobo_enz_sub))
##############################
# Map PubChem ID  to SMILES
##############################
# cid_to_smiles = pubchem2smiles(pubchem_ids)
# cid_to_smiles_df = pd.DataFrame(list(cid_to_smiles.items()), columns=['PubChem_ID', 'SMILES'])
# cid_to_smiles_df.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data","GO", "gobo_cid2smiles.pkl"))

cid_to_smiles_df = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data", "GO", "gobo_cid2smiles.pkl"))
cid_to_smiles_dict = dict(zip(cid_to_smiles_df['PubChem_ID'], cid_to_smiles_df['SMILES']))
gobo_enz_sub['SMILES'].fillna(gobo_enz_sub['molecule_ID'].map(cid_to_smiles_dict), inplace=True)
##############################
# Save data
##############################
gobo_enz_sub["Evidence"] = "EEC"  # EEC stands for Experimental Evidence Code
gobo_enz_sub["activity_comment"] = "CAS"  # CAS stand for Classified As Substrate
gobo_enz_sub = gobo_enz_sub[
    ['Uni_SwissProt', 'molecule_ID', 'Protein_Sequence', 'SMILES', 'EC_ID', 'Evidence', 'activity_comment']]
gobo_enz_sub.dropna(subset=['SMILES'], inplace=True)
gobo_enz_sub.reset_index(drop=True, inplace=True)
gobo_enz_sub.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data", "GO", "4-3-gobo_enz_sub.pkl"))
print(data_report(gobo_enz_sub))
