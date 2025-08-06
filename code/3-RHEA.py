import pandas as pd
from os.path import join
import os
import sys
sys.path.append("/../utilities")
from utilities.helper_functions import *

CURRENT_DIR = os.getcwd()

df_RHEA = pd.DataFrame(columns=["RHEA ID", "Reactions", "Substrate_list", "EC_ID"])
# Download rhea-reactions.txt from here: https://ftp.expasy.org/databases/rhea/txt/
with open(join(CURRENT_DIR, "..", "data", "raw_data", "rhea-reactions.txt"), 'r') as file1:
    Lines = file1.readlines()
rows = []
while True:
    try:
        end = Lines.index('///\n')
        entry = Lines[:end]
        RHEA_ID = entry[0][len("ENTRY"):].strip().split(" ")[-1]
        CHEBI_IDs = entry[2][len("EQUATION"):].strip()
        CHEBI_IDs = CHEBI_IDs[CHEBI_IDs.index("CHEBI"):]
        enzyme = next((line[len("ENZYME"):].strip() for line in entry if line.startswith("ENZYME")), None)
        CHEBI_ID_list = get_substrate_IDs(IDs=CHEBI_IDs)
        rows.append({"RHEA ID": RHEA_ID,
                     "Reactions": CHEBI_IDs,
                     "Substrate_list": CHEBI_ID_list,
                     "EC_ID": enzyme})
        Lines = Lines[end + 1:]
    except ValueError:
        break
df_RHEA = pd.concat([df_RHEA, pd.DataFrame(rows)], ignore_index=True)
df_RHEA["RHEA ID"] = [int(ID.split(":")[-1]) for ID in df_RHEA["RHEA ID"]]
rhea2uniprot = pd.read_csv(join(CURRENT_DIR, "..", "data", "raw_data", "rhea2uniprot_sprot.tsv"), sep="\t")
rhea2go = pd.read_csv(join(CURRENT_DIR, "..", "data", "raw_data", "rhea2go.tsv"), sep="\t")
rhea2go.drop(columns=['DIRECTION', 'MASTER_ID'], inplace=True)
rhea2uniprot.drop(columns=['DIRECTION', 'MASTER_ID'], inplace=True)
df_RHEA_uniprot = pd.merge(df_RHEA, rhea2uniprot, left_on=['RHEA ID'], right_on=['RHEA_ID'], how='left')
df_RHEA_uniprot.drop(columns=['RHEA_ID'], inplace=True)
df_RHEA_uniprot.rename(columns={'ID': 'Uniprot_ID'}, inplace=True)
df_RHEA_uniprot_go = pd.merge(df_RHEA_uniprot, rhea2go, left_on=['RHEA ID'], right_on=['RHEA_ID'], how='left')
df_RHEA_uniprot_go.rename(columns={'ID': 'GO_ID'}, inplace=True)
df_RHEA_uniprot_go.dropna(subset=['Uniprot_ID'], inplace=True)
df_RHEA_uniprot_go.dropna(subset=['GO_ID'], inplace=True)
df_RHEA_uniprot_go.reset_index(drop=True, inplace=True)
print(data_report(df_RHEA_uniprot_go))
experimental_df = pd.read_pickle(join(CURRENT_DIR, "..", "data", "raw_data", "GOA_data", "experimental_df_GO_UID.pkl"))

df_RHEA_uniprot_go = pd.merge(df_RHEA_uniprot_go, experimental_df, left_on=['Uniprot_ID', "GO_ID"],
                              right_on=['Uniprot ID', "GO Term"], how='inner')
df_RHEA_uniprot_go.drop(columns=['Uniprot ID', 'RHEA ID'], inplace=True)
df_RHEA_uniprot_go = df_RHEA_uniprot_go.loc[df_RHEA_uniprot_go['evidence'] == 'exp']
print(data_report(df_RHEA_uniprot_go))
df_RHEA_uniprot_go['ChEBI_ID'] = df_RHEA_uniprot_go['Substrate_list'].apply(
    lambda x: x.split(',') if isinstance(x, str) else (x if isinstance(x, list) else [])
)
df_RHEA_exploded = df_RHEA_uniprot_go.explode('ChEBI_ID')
df_RHEA_exploded.reset_index(drop=True, inplace=True)
print(data_report(df_RHEA_exploded))
#######################################
# Get SMILES
#######################################
# CheBI_IDs = list(set(df_RHEA_exploded['ChEBI_ID']))
# dict_chebi2smiles = chebi2smiles(CheBI_IDs)
# chebi_to_smiles_df = pd.DataFrame(list(dict_chebi2smiles.items()), columns=['ChEBI_ID', 'SMILES'])
# chebi_to_smiles_df.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data","rhea", "Rhea_chebi2smiles.pkl"))

chebi_to_smiles_df = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data", "rhea", "Rhea_chebi2smiles.pkl"))
dict_chebi2smiles = dict(zip(chebi_to_smiles_df['ChEBI_ID'], chebi_to_smiles_df['SMILES']))
df_RHEA_exploded['SMILES'] = df_RHEA_exploded['ChEBI_ID'].map(dict_chebi2smiles)

#######################################
# Get sequence
#######################################
uniprot_id = df_RHEA_exploded['Uniprot_ID'].unique()
sequences = get_protein_sequences_with_retry(uniprot_id)
sequence_df = pd.DataFrame(list(sequences.items()), columns=['UniProt_ID', 'Protein_Sequence'])
sequence_df.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data", "rhea", "Rhea_proId2seq.pkl"))

sequence_df = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data", "rhea", "Rhea_proId2seq.pkl"))
df_RHEA_exploded['Protein_Sequence'] = np.nan
uniprot_ids_no_seq = []
for _, row in sequence_df.iterrows():
    uniprot_id = row['UniProt_ID']
    sequence = row['Protein_Sequence']
    if pd.notna(sequence):
        df_RHEA_exploded.loc[df_RHEA_exploded['Uniprot_ID'] == uniprot_id, 'Protein_Sequence'] = sequence
    else:
        uniprot_ids_no_seq.append(uniprot_id)
with open(join(CURRENT_DIR, "..", "data", "processed_data", "rhea", "Rhea_ids_no_seq.txt"), 'w') as file:
    for item in uniprot_ids_no_seq:
        file.write(f"{item}\n")

#######################################
# Map Chebi Id to Pubchem ID
#######################################
# Source mapping tables:
# https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/general-questions

chembl2chebi = pd.read_csv(
    join(CURRENT_DIR, "..", "data", "raw_data", "map_chembl_to_other_ids", "src1src7.txt"),
    sep='\t',
    header=None,
    names=["chembel_ID", "Chebi_ID"], skiprows=1
)
chembl2pub = pd.read_csv(
    join(CURRENT_DIR, "..", "data", "raw_data", "map_chembl_to_other_ids", "src1src22.txt"),
    sep='\t',
    header=None,
    names=["chembel_ID", "Pubchem_ID"], skiprows=1
)
chembl2chebi_dict = chembl2chebi.set_index("chembel_ID")["Chebi_ID"].to_dict()
chembl2pub["Chebi_ID"] = chembl2pub["chembel_ID"].map(chembl2chebi_dict)
chembl2pub.dropna(subset=["Chebi_ID"], inplace=True)
chembl2pub['Chebi_ID'] = chembl2pub['Chebi_ID'].apply(
    lambda x: f'CHEBI:{int(x)}' if pd.notna(x) and str(x).replace('.', '').isdigit() else x)
chebi2pub = chembl2pub[['Chebi_ID', 'Pubchem_ID']]
chebi2pub.reset_index(drop=True, inplace=True)
chebi2pub_dict = chebi2pub.set_index("Chebi_ID")["Pubchem_ID"].to_dict()
df_RHEA_exploded["Pubchem_ID"] = df_RHEA_exploded["ChEBI_ID"].map(chebi2pub_dict)
df_RHEA_exploded['Pubchem_ID'] = df_RHEA_exploded['Pubchem_ID'].round().astype('Int64')

df_RHEA_exploded['molecule_ID'] = [
    row['Pubchem_ID'] if pd.notnull(row['Pubchem_ID']) else
    row['ChEBI_ID']
    for index, row in df_RHEA_exploded.iterrows()
]

df_RHEA_exploded.rename(columns={'Uniprot_ID': 'Uni_SwissProt'}, inplace=True)
df_RHEA_exploded["Evidence"] = "EEC"  # EEC stand for Experimental Evidence Code
df_RHEA_exploded["activity_comment"] = "CAS"  # CAS stand for Classified As Substrate
df_RHEA_exploded = df_RHEA_exploded[
    ['Uni_SwissProt', 'molecule_ID', 'Protein_Sequence', 'SMILES', 'EC_ID', 'Evidence', 'activity_comment']]

# check additional_code folder to see how cofactors_list.txt has been created
with open(join(CURRENT_DIR, "..", "data", "processed_data", "cofactors_list.txt"), "r") as f:
    remove_cofactor_energy_ids = [line.strip() for line in f.readlines()]
df_RHEA_exploded["molecule_ID"] = df_RHEA_exploded["molecule_ID"].astype(str)
df_RHEA_exploded = df_RHEA_exploded.loc[~df_RHEA_exploded["molecule_ID"].isin(remove_cofactor_energy_ids)]
df_RHEA_exploded.reset_index(drop=True, inplace=True)
print(data_report(df_RHEA_exploded))
df_RHEA_exploded.dropna(subset=['SMILES'], inplace=True)
print(data_report(df_RHEA_exploded))
df_RHEA_exploded.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data", "rhea", "3-rhea_enz_sub.pkl"))
