import requests
import pandas as pd
import os
import sys
import re
from os.path import join
import gc
from chemdataextractor import Document
import json

sys.path.append("./../utilities")
from utilities.helper_functions import *

CURRENT_DIR = os.getcwd()
print(CURRENT_DIR)

df_exploded = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data","uniprot", "7-1-uniprot_enz-inh.pkl"))
# print(data_report(df_exploded))

df_exploded["Ihibitors_description"] = df_exploded["Ihibitors_description"].str.replace(r"\s*\(PubMed:[^)]*\)", "",
                                                                                        regex=True)


def extract_inhibitors(description):
    inhibitors = []
    if match := re.search(r"inhibit(?:ed|ion)?\s+.*?\bby\b\s+(.*)", description, re.IGNORECASE):
        inhibitors_part = match.group(1)
        inhibitors_part = inhibitors_part.split("but not")[0].strip()
        inhibitors_part = inhibitors_part.split("cofactor")[0].strip()
        inhibitors_part = re.sub(
            r"(?<=\s)(?:with|where|which|that|an|the|inhibitor|"
            r"substrate|analog|primary|secondary|is|to|bind|for|analogs|"
            r"as|so|such|binding|a|by|activity|site|mediated|partialy|in|above|uM|nM|mM|only|"
            r"at|very|high|weak|weakly|drugs|low|Ki|0\.25|IC50|approximately|mm|from|of|small|"
            r"molecule|increasing|concentrations|metabolite|vitro|also)(?=\s|'\s*)", "", inhibitors_part).strip()
        inhibitors_part = re.sub(r"including|as well as", ",", inhibitors_part, flags=re.IGNORECASE)
        inhibitors_part = re.sub(r", and\s+", ", ", inhibitors_part)
        inhibitors_part = re.sub(r"\s+and\s+", ", ", inhibitors_part)
        doc = Document(inhibitors_part)
        for c in doc.cems:
            inhibitors.append(c.text)

    return inhibitors


df_exploded["Inhibitors"] = df_exploded["Ihibitors_description"].apply(extract_inhibitors)
uniprot_enz_inh = df_exploded.explode("Inhibitors")
uniprot_enz_inh.dropna(subset=["Inhibitors"], inplace=True)
uniprot_enz_inh.drop_duplicates(subset=["ID", "Inhibitors"], inplace=True)
# print(data_report(uniprot_enz_inh))

#######################################
# Pubchem
#######################################
# inh_unique = uniprot_enz_inh['Inhibitors'].unique().tolist()
# print(len(inh_unique))
# pubID_dict = get_pubchem_ids(inh_unique)
# name_to_cid_df = pd.DataFrame(list(pubID_dict.items()), columns=['molecule_name', 'ID'])
# name_to_cid_df.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data","uniprot", "UniInh_MolName2cid.pkl"))

name_to_cid_df = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data","uniprot", "UniInh_MolName2cid.pkl"))
pubID_dict = dict(zip(name_to_cid_df['molecule_name'], name_to_cid_df['ID']))
uniprot_enz_inh["molecule_ID"] = uniprot_enz_inh["Inhibitors"].map(pubID_dict)
uniprot_enz_inh["Inhibitors"] = uniprot_enz_inh["Inhibitors"].apply(
    lambda name: name.lower() if isinstance(name, str) else name)

mol2ids = pd.read_pickle(join(CURRENT_DIR, "..", "data", "raw_data", "molecule_name_to_ids", "mol2ids.pkl"))
mol2ids["molecule_name"] = mol2ids["molecule_name"].str.lower()

mol2ids_dict = dict(zip(mol2ids["molecule_name"], mol2ids["molecule_ID"]))
uniprot_enz_inh["molecule_ID"].fillna(uniprot_enz_inh["Inhibitors"].map(mol2ids_dict), inplace=True)
uniprot_enz_inh.dropna(subset=["molecule_ID"], inplace=True)
##############################
# separate different IDS
##############################
unique_molecule_ids = uniprot_enz_inh['molecule_ID'].dropna().unique()
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
# chebi_to_smiles_df.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data","uniprot", "UniInh_chebi2smiles.pkl"))

chebi_to_smiles_df = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data","uniprot", "UniInh_chebi2smiles.pkl"))
dict_chebi2smiles = dict(zip(chebi_to_smiles_df['ChEBI_ID'], chebi_to_smiles_df['SMILES']))
uniprot_enz_inh['SMILES'] = uniprot_enz_inh['molecule_ID'].map(dict_chebi2smiles)
# print(data_report(uniprot_enz_inh))
##############################
# Map PubChem ID  to SMILES
##############################
# cid_to_smiles = pubchem2smiles(pubchem_ids)
# cid_to_smiles_df = pd.DataFrame(list(cid_to_smiles.items()), columns=['PubChem_ID', 'SMILES'])
# cid_to_smiles_df.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data", "uniprot", "UniInh_cid2smiles.pkl"))

cid_to_smiles_df = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data","uniprot", "UniInh_cid2smiles.pkl"))
cid_to_smiles = dict(zip(cid_to_smiles_df['PubChem_ID'], cid_to_smiles_df['SMILES']))
uniprot_enz_inh['SMILES'].fillna(uniprot_enz_inh['molecule_ID'].map(cid_to_smiles), inplace=True)
#print(data_report(uniprot_enz_inh))

##############################
# Map KEGG ID  to SMILES
##############################
# dict_kegg2smiles = kegg2smiles(kegg_ids)
# kegg_to_smiles_df = pd.DataFrame(list(dict_kegg2smiles.items()), columns=['KEGG', 'SMILES'])
# kegg_to_smiles_df.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data", "uniprot", "UniInh_kegg2smiles.pkl"))

kegg_to_smiles_df = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data","uniprot", "UniInh_kegg2smiles.pkl"))
dict_kegg2smiles = dict(zip(kegg_to_smiles_df['KEGG'], kegg_to_smiles_df['SMILES']))
uniprot_enz_inh['SMILES'].fillna(uniprot_enz_inh['molecule_ID'].map(cid_to_smiles), inplace=True)
uniprot_enz_inh.dropna(subset=['SMILES'], inplace=True)
uniprot_enz_inh.reset_index(drop=True, inplace=True)
# print(data_report(uniprot_enz_inh))

uniprot_enz_inh.rename(columns={'ID': 'Uni_SwissProt', 'EC Number': 'EC_ID'}, inplace=True)
# check additional_code folder to see how cofactors_list.txt has been created
with open(join(CURRENT_DIR, "..", "data", "processed_data","cofactors_list.txt"), "r") as f:
    remove_cofactor_energy_ids = [line.strip() for line in f.readlines()]
uniprot_enz_inh["molecule_ID"] = uniprot_enz_inh["molecule_ID"].astype(str)
uniprot_enz_inh["Evidence"]="EEC" # EEC stand for Experimental Evidence Code
uniprot_enz_inh["activity_comment"] = "CAI"
uniprot_enz_inh = uniprot_enz_inh.loc[~uniprot_enz_inh["molecule_ID"].isin(remove_cofactor_energy_ids)]
uniprot_enz_inh = uniprot_enz_inh[
    ['Uni_SwissProt', 'molecule_ID', 'Protein_Sequence', 'SMILES', 'EC_ID','Evidence','activity_comment']]
uniprot_enz_inh.reset_index(drop=True, inplace=True)
uniprot_enz_inh.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data","uniprot", "7-2-uniprot_enz-inh.pkl"))
# print(data_report(uniprot_enz_inh))
