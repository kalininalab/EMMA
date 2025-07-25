import pandas as pd
import os
from math import ceil
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.append("./../Utilities")
import re
import ast
from os.path import join
from helper_functions import *
CURRENT_DIR = os.getcwd()
print(CURRENT_DIR)
##############################
# Map ChEBI ID to SMILES
##############################
brenda_enz_sub = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data","brenda", "5-2-brenda_enz_sub.pkl"))
brenda_enz_inh = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data","brenda", "5-2-brenda_enz_inh.pkl"))
# Combine the molecule_ID columns
combined_molecule_ids = pd.concat([brenda_enz_sub['molecule_ID'], brenda_enz_inh['molecule_ID']]).dropna().unique()
combined_molecule_ids = combined_molecule_ids[combined_molecule_ids != 'NIL']
# Initialize lists for each identifier type
chebi_ids = []
pubchem_ids = []
kegg_ids = []

for molecule_id in combined_molecule_ids:
    molecule_id = str(molecule_id)  # Ensure it's a string
    if re.match(r'^CHEBI:', molecule_id):
        chebi_ids.append(molecule_id)
    elif re.match(r'^\d+$', molecule_id):
        pubchem_ids.append(molecule_id)
    elif re.match(r'^C\d+', molecule_id):
        kegg_ids.append(molecule_id)
    else:
        continue

##############################
# Map KEGG ID  to SMILES
##############################
# dict_kegg2smiles = kegg2smiles(kegg_ids)
# kegg_to_smiles_df = pd.DataFrame(list(dict_kegg2smiles.items()), columns=['KEGG', 'SMILES'])
# kegg_to_smiles_df.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data", "brenda", "Brenda_kegg2smiles.pkl"))

kegg_to_smiles_df = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data", "brenda", "Brenda_kegg2smiles.pkl"))
dict_kegg2smiles = dict(zip(kegg_to_smiles_df['KEGG'], kegg_to_smiles_df['SMILES']))
brenda_enz_sub['SMILES'] = brenda_enz_sub['molecule_ID'].map(dict_kegg2smiles)
brenda_enz_inh['SMILES'] = brenda_enz_inh['molecule_ID'].map(dict_kegg2smiles)
print(data_report(brenda_enz_sub))
print(data_report(brenda_enz_inh))

##############################
# Map ChEBI ID  to SMILES
##############################
# dict_chebi2smiles = chebi2smiles(chebi_ids)
# chebi_to_smiles_df = pd.DataFrame(list(dict_chebi2smiles.items()), columns=['ChEBI_ID', 'SMILES'])
# chebi_to_smiles_df.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data", "brenda", "Brenda_chebi2smiles.pkl"))

chebi_to_smiles_df = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data", "brenda", "Brenda_chebi2smiles.pkl"))
dict_chebi2smiles = dict(zip(chebi_to_smiles_df['ChEBI_ID'], chebi_to_smiles_df['SMILES']))
brenda_enz_sub['SMILES'].fillna(brenda_enz_sub['molecule_ID'].map(dict_chebi2smiles),inplace=True)
brenda_enz_inh['SMILES'].fillna(brenda_enz_inh['molecule_ID'].map(dict_chebi2smiles),inplace=True)
print(data_report(brenda_enz_sub))
print(data_report(brenda_enz_inh))
##############################
# Map PubChem ID  to SMILES
##############################
# cid_to_smiles = pubchem2smiles(pubchem_ids)
# cid_to_smiles_df = pd.DataFrame(list(cid_to_smiles.items()), columns=['PubChem_ID', 'SMILES'])
# cid_to_smiles_df.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data", "brenda", "Brenda_cid2smiles.pkl"))

cid_to_smiles_df = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data", "brenda", "Brenda_cid2smiles.pkl"))
cid_to_smiles_dict = dict(zip(cid_to_smiles_df['PubChem_ID'], cid_to_smiles_df['SMILES']))
brenda_enz_sub['SMILES'].fillna(brenda_enz_sub['molecule_ID'].map(cid_to_smiles_dict),inplace=True)
brenda_enz_inh['SMILES'].fillna(brenda_enz_inh['molecule_ID'].map(cid_to_smiles_dict),inplace=True)
print(data_report(brenda_enz_sub))
print(data_report(brenda_enz_inh))
##################################
# Remove small molecules
##################################
# check additional_code folder to see how cofactors_list.txt has been created
with open(join(CURRENT_DIR, "..", "data", "processed_data","cofactors_list.txt"), "r") as f:
    remove_cofactor_energy_ids = [line.strip() for line in f.readlines()]
brenda_enz_inh["molecule_ID"] = brenda_enz_inh["molecule_ID"].astype(str)
brenda_enz_sub["molecule_ID"] = brenda_enz_sub["molecule_ID"].astype(str)
brenda_enz_inh = brenda_enz_inh.loc[~brenda_enz_inh["molecule_ID"].isin(remove_cofactor_energy_ids)]
brenda_enz_sub = brenda_enz_sub.loc[~brenda_enz_sub["molecule_ID"].isin(remove_cofactor_energy_ids)]
brenda_enz_inh.reset_index(drop=True, inplace=True)
brenda_enz_sub.reset_index(drop=True, inplace=True)
print(data_report(brenda_enz_sub))
print(data_report(brenda_enz_inh))

#######################################
# Get sequence
#######################################
Prot_Sub = brenda_enz_sub['Uni_SwissProt'].unique()
Prot_INH = brenda_enz_inh['Uni_SwissProt'].unique()
uniprot_IDs = np.unique(np.concatenate((Prot_Sub, Prot_INH)))
sequences = get_protein_sequences_with_retry(uniprot_IDs)
sequence_df = pd.DataFrame(list(sequences.items()), columns=['UniProt_ID', 'Protein_Sequence'])
sequence_df.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data", "brenda", "Brenda_proId2seq.pkl"))

sequence_df = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data", "brenda", "Brenda_proId2seq.pkl"))
brenda_enz_sub['Protein_Sequence'] = np.nan
brenda_enz_inh['Protein_Sequence'] = np.nan
uniprot_ids_no_seq = []
for _, row in sequence_df.iterrows():
    uniprot_id = row['UniProt_ID']
    sequence = row['Protein_Sequence']
    if pd.notna(sequence):
        brenda_enz_sub.loc[brenda_enz_sub['Uni_SwissProt'] == uniprot_id, 'Protein_Sequence'] = sequence
        brenda_enz_inh.loc[brenda_enz_inh['Uni_SwissProt'] == uniprot_id, 'Protein_Sequence'] = sequence
    else:
        uniprot_ids_no_seq.append(uniprot_id)
with open(join(CURRENT_DIR, "..", "data", "processed_data", "brenda", "uniprot_ids_no_seq.txt"), 'w') as file:
    for item in uniprot_ids_no_seq:
        file.write(f"{item}\n")


brenda_enz_sub = brenda_enz_sub.groupby(['Uni_SwissProt', 'molecule_ID'], as_index=False).apply(
    lambda x: x[x['standard_value'].notna()].head(1) if x['standard_value'].notna().any() else x.head(1)).reset_index(drop=True)
brenda_enz_sub['standard_type'] = None
brenda_enz_sub['standard_value'] = pd.to_numeric(brenda_enz_sub['standard_value'], errors='coerce')
brenda_enz_sub.loc[brenda_enz_sub['standard_value'].notna(), 'standard_type'] = "Km"
brenda_enz_inh['standard_type'] = None
brenda_enz_inh.loc[brenda_enz_inh['Ki_value'].notna(), 'assay_type'] = 'Ki'
brenda_enz_inh.loc[(brenda_enz_inh['IC50_value'].notna()) &(brenda_enz_inh['standard_type'].isna()),  'standard_type'] = 'IC50'

brenda_enz_inh["IC50_value"] = pd.to_numeric(brenda_enz_inh["IC50_value"], errors="coerce")
brenda_enz_inh["IC50_value"] = brenda_enz_inh["IC50_value"].apply(lambda x: x / 2 if pd.notna(x) else x)

brenda_enz_inh['standard_value'] = np.where(brenda_enz_inh['Ki_value'].notna(), brenda_enz_inh['Ki_value'],brenda_enz_inh['IC50_value'])
brenda_enz_inh.drop(['Ki_value', 'IC50_value'], axis=1, inplace=True)

brenda_enz_inh = brenda_enz_inh.groupby(['Uni_SwissProt', 'molecule_ID'], as_index=False).apply(
    lambda x: x[x['standard_value'].notna()].head(1) if x['standard_value'].notna().any() else x.head(1)).reset_index(drop=True)
brenda_enz_inh['standard_value'] = pd.to_numeric(brenda_enz_inh['standard_value'], errors='coerce')

brenda_enz_sub['Protein_Sequence'].replace('', pd.NA, inplace=True)
brenda_enz_inh['Protein_Sequence'].replace('', pd.NA, inplace=True)
brenda_enz_sub.dropna(subset=['Protein_Sequence', 'SMILES'], inplace=True)
brenda_enz_inh.dropna(subset=['Protein_Sequence', 'SMILES'], inplace=True)
brenda_enz_sub.reset_index(drop=True, inplace=True)
brenda_enz_inh.reset_index(drop=True, inplace=True)
print(data_report(brenda_enz_sub))
print(data_report(brenda_enz_inh))


brenda_enz_sub_no_km=brenda_enz_sub[brenda_enz_sub['standard_value'].isna()]
brenda_enz_sub_km=brenda_enz_sub[brenda_enz_sub['standard_value'].notna()]
# 2. Convert Km from mM to nM (1 mM = 1,000,000 nM)
brenda_enz_sub_km['standard_value'] = brenda_enz_sub_km['standard_value'] * 1000000
print(data_report(brenda_enz_sub_no_km))
print(data_report(brenda_enz_sub_km))

brenda_enz_inh_no_kiic=brenda_enz_inh[brenda_enz_inh['standard_value'].isna()]
brenda_enz_inh_kiic=brenda_enz_inh[brenda_enz_inh['standard_value'].notna()]
# 2. Convert ki and IC50 from mM to nM (1 mM = 1,000,000 nM)
brenda_enz_inh_kiic['standard_value'] = brenda_enz_inh_kiic['standard_value'] * 1000000
print(data_report(brenda_enz_inh_no_kiic))
print(data_report(brenda_enz_inh_kiic))

ec2go = pd.read_csv(join(CURRENT_DIR, "..", "data", "raw_data", "ec2go.csv"))
ec2go_dict = dict(zip(ec2go['EC_ID'], ec2go['GO_term']))
brenda_enz_sub_no_km['GO_ID'] = brenda_enz_sub_no_km['EC_ID'].map(ec2go_dict)
brenda_enz_inh_no_kiic['GO_ID'] = brenda_enz_inh_no_kiic['EC_ID'].map(ec2go_dict)
experimental_df = pd.read_pickle(join(CURRENT_DIR, "..", "data", "raw_data","GOA_data", "experimental_df_GO_UID.pkl"))
brenda_enz_sub_no_km = pd.merge(brenda_enz_sub_no_km, experimental_df, left_on=['Uni_SwissProt', "GO_ID"], right_on=['Uniprot ID',"GO Term"], how='inner')
brenda_enz_inh_no_kiic = pd.merge(brenda_enz_inh_no_kiic, experimental_df, left_on=['Uni_SwissProt', "GO_ID"], right_on=['Uniprot ID',"GO Term"], how='inner')

brenda_enz_sub_no_km.dropna(subset=['evidence'], inplace=True)
brenda_enz_inh_no_kiic.dropna(subset=['evidence'], inplace=True)
brenda_enz_sub_no_km = brenda_enz_sub_no_km.loc[brenda_enz_sub_no_km['evidence'] == 'exp']
brenda_enz_inh_no_kiic = brenda_enz_inh_no_kiic.loc[brenda_enz_inh_no_kiic['evidence'] == 'exp']
brenda_enz_sub_no_km.drop(['GO_ID','evidence','ECO_Evidence_code','Uniprot ID', 'GO Term'], axis=1, inplace=True)
brenda_enz_inh_no_kiic.drop(['GO_ID','evidence','ECO_Evidence_code','Uniprot ID', 'GO Term'], axis=1, inplace=True)
brenda_enz_inh_no_kiic["Evidence"]="EEC" # EEC stands for Experimental Evidence Code
brenda_enz_inh_no_kiic["activity_comment"]="CAI"
brenda_enz_inh_kiic["Evidence"]="DBA" # DBA stands for Direct Binding Assay
brenda_enz_inh_kiic["activity_comment"]=None
brenda_enz_sub_no_km["Evidence"]="EEC"
brenda_enz_sub_no_km["activity_comment"]="CAS"
brenda_enz_sub_km["Evidence"]="DBA"
brenda_enz_sub_km["activity_comment"]=None
print(data_report(brenda_enz_sub_no_km))
print(data_report(brenda_enz_inh_no_kiic))

final_enz_inh=pd.concat([brenda_enz_inh_no_kiic, brenda_enz_inh_kiic])
final_enz_sub=pd.concat([brenda_enz_sub_no_km,brenda_enz_sub_km])
final_enz_inh = final_enz_inh[['Uni_SwissProt', 'molecule_ID', 'Protein_Sequence','standard_type', 'standard_value', 'SMILES',"EC_ID", "Evidence","activity_comment"]]
final_enz_sub = final_enz_sub[['Uni_SwissProt', 'molecule_ID', 'Protein_Sequence','standard_type','standard_value', 'SMILES',"EC_ID", "Evidence","activity_comment"]]
final_enz_sub.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data", "brenda", "5-3-brenda_enz_sub.pkl"))
final_enz_inh.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data","brenda", "5-3-brenda_enz_inh.pkl"))
print(data_report(final_enz_inh))
print(data_report(final_enz_sub))

