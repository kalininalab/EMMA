import requests
import pandas as pd
import os
import io
import sys
import re
import pickle
from os.path import join, exists
import math
sys.path.append("./../utilities")
from utilities.helper_functions import *
CURRENT_DIR = os.getcwd()
print(CURRENT_DIR)



def get_sabio_data():
    ENTRYID_QUERY_URL = 'https://sabiork.h-its.org/sabioRestWebServices/searchKineticLaws/entryIDs'
    PARAM_QUERY_URL = 'https://sabiork.h-its.org/entry/exportToExcelCustomizable'

    # ask SABIO-RK for all EntryIDs matching a query
    query_dict = {"EntryID":"*"}
    query_string = ' AND '.join(['%s:%s' % (k,v) for k,v in query_dict.items()])
    query = {'format':'txt', 'q':query_string}

    # make GET request
    request = requests.get(ENTRYID_QUERY_URL, params=query)
    request.raise_for_status() # raise if 404 error
    entryIDs = [int(x) for x in request.text.strip().split('\n')]
    print('%d matching entries found.' % len(entryIDs))

    # encode next request, for parameter data given entry IDs
    data_field = {'entryIDs[]': entryIDs}
    query = {'format': 'tsv', 'fields[]': ['EntryID', 'Organism', 'UniprotID',
                                         'ECNumber','Parameter','EnzymeType', "Enzymename", "KeggReactionID",
                                         "PubMedID", "Substrate", "Inhibitor", "PubChemID"]}

    # make POST request
    request = requests.post(PARAM_QUERY_URL, params=query, data=data_field)
    request.raise_for_status()
    df = pd.read_csv(io.StringIO(request.text), sep='\t')
    df = df.dropna(subset=['UniprotID'])
    output_filename = join(CURRENT_DIR, "..", "data", "processed_data", "sabio", "10-sabio_rk_results.tsv")
    df.to_csv(output_filename, sep='\t', index=False, encoding='utf-8')
    print(f"Results saved to {output_filename}")
    print(f"Original entries: {len(entryIDs)}, Entries with UniprotID: {len(df)}")


# get_sabio_data()

sabio_data=pd.read_csv(join(CURRENT_DIR, "..", "data", "processed_data", "sabio", "10-sabio_rk_results.tsv" ), sep='\t')
print(data_report(sabio_data))


def extract_substrate(df):
    # only keep data points with Km values (that ar nonzero):
    data = df.copy()
    data = data.loc[data["UniprotID"] != ""]
    data = data.loc[~pd.isnull(data["UniprotID"])]
    sub_km = data.loc[data["parameter.type"].isin(["Km"]) ]
    sub_km = sub_km.loc[sub_km["parameter.startValue"] != 0]
    sub_km = sub_km.loc[sub_km["parameter.unit"] == "M"]
    sub_km["parameter.startValue"] = sub_km["parameter.startValue"] * 1e9
    # Process Substrate data - first split, then explode
    substrate = data.copy()
    substrate["Substrate"] = substrate["Substrate"].str.split(";")
    substrate_2 = substrate.explode("Substrate")

    # only keep necessary columns:
    enz_sub_km= pd.DataFrame(data={"Uni_SwissProt": sub_km["UniprotID"], "EC_ID": sub_km["ECNumber"],
                                   "standard_type":sub_km["parameter.type"],
                                   "standard_value": sub_km["parameter.startValue"],
                                   "Substrate": sub_km["parameter.associatedSpecies"]})

    enz_sub= pd.DataFrame(data={"Uni_SwissProt": substrate_2["UniprotID"], "EC_ID": substrate_2["ECNumber"],
                                "standard_type": None,
                                "standard_value": None,
                                "Substrate": substrate_2["Substrate"]})

    final_data=pd.concat([enz_sub, enz_sub_km])
    final_data = final_data.groupby(['Uni_SwissProt', 'Substrate'], as_index=False).apply(
        lambda x: x[x['standard_value'].notna()].head(1) if x['standard_value'].notna().any() else x.head(1)).reset_index(drop=True)

    final_data.reset_index(drop=True, inplace=True)

    return final_data

def extract_inhibitors(df):
    # only keep data points with Km values (that ar nonzero):
    data = df.copy()
    data = data.loc[data["UniprotID"] != ""]
    data = data.loc[~pd.isnull(data["UniprotID"])]
    inh_kic = data.loc[data["parameter.type"].isin(["Ki",'IC50',"Kis","Kii"])] #
    inh_kic = inh_kic.loc[inh_kic["parameter.startValue"] != 0]
    inh_kic = inh_kic.loc[inh_kic["parameter.unit"] == "M"]
    inh_kic["parameter.startValue"] = inh_kic["parameter.startValue"] * 1e9
    # Process Substrate data - first split, then explode
    inh = data.copy()
    inh["Inhibitor"] = inh["Inhibitor"].str.split(";")
    inh_2 = inh.explode("Inhibitor")
    inh_2.dropna(subset=["Inhibitor"], inplace=True)


    # only keep necessary columns:
    enz_inh_kiic= pd.DataFrame(data={"Uni_SwissProt": inh_kic["UniprotID"], "EC_ID": inh_kic["ECNumber"],
                                     "standard_type": inh_kic["parameter.type"],
                                     "standard_value": inh_kic["parameter.startValue"],
                                     "Inhibitor": inh_kic["parameter.associatedSpecies"]})

    enz_inh= pd.DataFrame(data={"Uni_SwissProt": inh_2["UniprotID"], "EC_ID": inh_2["ECNumber"],
                                "standard_type": None,
                                "standard_value": None,
                                "Inhibitor": inh_2["Inhibitor"]})
    enz_inh.reset_index(drop=True, inplace=True)

    final_data=pd.concat([enz_inh, enz_inh_kiic])
    final_data = final_data.groupby(['Uni_SwissProt', 'Inhibitor'], as_index=False).apply(
        lambda x: x[x['standard_value'].notna()].head(1) if x['standard_value'].notna().any() else x.head(1)).reset_index(drop=True)


    return final_data


substrate_df=extract_substrate(sabio_data)
inhibitors_df =extract_inhibitors(sabio_data)


#######################################
# Pubchem
#######################################
# mol_unique = list(set(substrate_df['Substrate'].unique().tolist() + inhibitors_df['Inhibitor'].unique().tolist()))
# pubID_dict = get_pubchem_ids(mol_unique)
# cid_to_name_df = pd.DataFrame(list(pubID_dict.items()), columns=['molecule_name', 'PubChem_ID'])
# cid_to_name_df.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data" ,"sabio", "sabio_cid2name.pkl"))

cid_to_name_df = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data","sabio", "sabio_cid2name.pkl"))
cid_to_name_dict = dict(zip(cid_to_name_df['molecule_name'], cid_to_name_df['PubChem_ID']))
substrate_df['molecule_ID'] = substrate_df['Substrate'].map(cid_to_name_dict)
inhibitors_df['molecule_ID'] = inhibitors_df['Inhibitor'].map(cid_to_name_dict)

substrate_df["Substrate"] = [name.lower() for name in substrate_df["Substrate"]]
inhibitors_df["Inhibitor"] = [name.lower() for name in inhibitors_df["Inhibitor"]]
mol2ids = pd.read_pickle(join(CURRENT_DIR, "..", "data", "raw_data", "molecule_name_to_ids", "mol2ids.pkl"))
mol2ids_dict = dict(zip(mol2ids['molecule_name'], mol2ids['molecule_ID']))
substrate_df['molecule_ID'].fillna(substrate_df['Substrate'].map(mol2ids_dict), inplace=True)
inhibitors_df['molecule_ID'].fillna(inhibitors_df['Inhibitor'].map(mol2ids_dict), inplace=True)
##################################
# Remove small molecules
##################################
# check additional_code folder to see how cofactors_list.txt has been created
with open(join(CURRENT_DIR, "..", "data", "processed_data", "cofactors_list.txt"), "r") as f:
    remove_cofactor_energy_ids = [line.strip() for line in f.readlines()]
inhibitors_df["molecule_ID"] = inhibitors_df["molecule_ID"].astype(str)
inhibitors_df = inhibitors_df.loc[~inhibitors_df["molecule_ID"].isin(remove_cofactor_energy_ids)]
inhibitors_df.dropna(subset=['molecule_ID'], inplace=True)
inhibitors_df.reset_index(drop=True, inplace=True)
substrate_df["molecule_ID"] = substrate_df["molecule_ID"].astype(str)
substrate_df = substrate_df.loc[~substrate_df["molecule_ID"].isin(remove_cofactor_energy_ids)]
substrate_df.dropna(subset=['molecule_ID'], inplace=True)
substrate_df.reset_index(drop=True, inplace=True)

#######################################
# Get sequence
#######################################
# uniprot_IDs = list(set(substrate_df['Uni_SwissProt'].unique().tolist() + inhibitors_df['Uni_SwissProt'].unique().tolist()))
# sequences = get_protein_sequences_with_retry(uniprot_IDs)
# sequence_df = pd.DataFrame(list(sequences.items()), columns=['Uniprot_ID', 'Protein_Sequence'])
# sequence_df.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data","sabio", "sabio_proId2seq.pkl"))

sequence_df = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data","sabio", "sabio_proId2seq.pkl"))
substrate_df['Protein_Sequence'] = np.nan
inhibitors_df['Protein_Sequence'] = np.nan
for _, row in sequence_df.iterrows():
    uniprot_id = row['Uniprot_ID']
    sequence = row['Protein_Sequence']
    if pd.notna(sequence):
        substrate_df.loc[substrate_df['Uni_SwissProt'] == uniprot_id, 'Protein_Sequence'] = sequence
        inhibitors_df.loc[inhibitors_df['Uni_SwissProt'] == uniprot_id, 'Protein_Sequence'] = sequence

substrate_df.dropna(subset=['Protein_Sequence'], inplace=True)
inhibitors_df.dropna(subset=['Protein_Sequence'], inplace=True)


def clean_molecule_ids(df):
    # Convert empty strings and 'nan' strings to actual NaN
    df['molecule_ID'] = df['molecule_ID'].replace(['', 'nan', 'None', None], np.nan)

    # Convert to string type if not already
    df['molecule_ID'] = df['molecule_ID'].astype(str)

    # Replace any remaining 'nan' strings that might have come from conversion
    df['molecule_ID'] = df['molecule_ID'].replace('nan', np.nan)

    # Now drop rows with NaN
    df.dropna(subset=['molecule_ID'], inplace=True)

    # Optional: Reset index if needed
    df.reset_index(drop=True, inplace=True)
    return df


# Apply to both dataframes
substrate_df = clean_molecule_ids(substrate_df)
inhibitors_df = clean_molecule_ids(inhibitors_df)

molecule_ids = list(set(substrate_df['molecule_ID'].unique().tolist() + inhibitors_df['molecule_ID'].unique().tolist()))

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
# kegg_to_smiles_df.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data","sabio", "sabio_kegg2smiles.pkl"))

kegg_to_smiles_df = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data","sabio", "sabio_kegg2smiles.pkl"))
dict_kegg2smiles = dict(zip(kegg_to_smiles_df['KEGG'], kegg_to_smiles_df['SMILES']))
substrate_df['SMILES'] = substrate_df['molecule_ID'].map(dict_kegg2smiles)
inhibitors_df['SMILES'] = inhibitors_df['molecule_ID'].map(dict_kegg2smiles)
##############################
# Map ChEBI ID  to SMILES
##############################
# dict_chebi2smiles = chebi2smiles(chebi_ids)
# chebi_to_smiles_df = pd.DataFrame(list(dict_chebi2smiles.items()), columns=['ChEBI_ID', 'SMILES'])
# chebi_to_smiles_df.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data","sabio", "sabio_chebi2smiles.pkl"))

chebi_to_smiles_df = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data","sabio", "sabio_chebi2smiles.pkl"))
dict_chebi2smiles = dict(zip(chebi_to_smiles_df['ChEBI_ID'], chebi_to_smiles_df['SMILES']))
substrate_df['SMILES'].fillna(substrate_df['molecule_ID'].map(dict_chebi2smiles),inplace=True)
inhibitors_df['SMILES'].fillna(inhibitors_df['molecule_ID'].map(dict_chebi2smiles),inplace=True)
##############################
# Map PubChem ID  to SMILES
##############################
# cid_to_smiles = pubchem2smiles(pubchem_ids)
# cid_to_smiles_df = pd.DataFrame(list(cid_to_smiles.items()), columns=['PubChem_ID', 'SMILES'])
# cid_to_smiles_df.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data","sabio", "sabio_cid2smiles.pkl"))

cid_to_smiles_df = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data","sabio", "sabio_cid2smiles.pkl"))
cid_to_smiles_dict = dict(zip(cid_to_smiles_df['PubChem_ID'], cid_to_smiles_df['SMILES']))
substrate_df['SMILES'].fillna(substrate_df['molecule_ID'].map(cid_to_smiles_dict),inplace=True)
inhibitors_df['SMILES'].fillna(inhibitors_df['molecule_ID'].map(cid_to_smiles_dict),inplace=True)
substrate_df["Evidence"]="DBA" # EEC stand for Experimental Evidence Code
substrate_df["activity_comment"] = None
inhibitors_df["Evidence"]="DBA" # EEC stand for Experimental Evidence Code
inhibitors_df["activity_comment"] = None
substrate_df = substrate_df[['Uni_SwissProt', 'molecule_ID', 'Protein_Sequence','standard_type', 'standard_value', 'SMILES',"EC_ID", "Evidence","activity_comment"]]
inhibitors_df = inhibitors_df[['Uni_SwissProt', 'molecule_ID', 'Protein_Sequence','standard_type', 'standard_value', 'SMILES',"EC_ID", "Evidence","activity_comment"]]
substrate_df.dropna(subset=['SMILES'], inplace=True)
substrate_df.dropna(subset=['standard_value'], inplace=True)
inhibitors_df.dropna(subset=['standard_value'], inplace=True)
substrate_df.reset_index(drop=True, inplace=True)
inhibitors_df.dropna(subset=['SMILES'], inplace=True)
inhibitors_df.reset_index(drop=True, inplace=True)
inhibitors_df.loc[inhibitors_df['standard_type'] == "IC50", 'standard_value'] *= 0.50
substrate_df.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data","sabio", "sabio_enz_sub.pkl"))
inhibitors_df.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data","sabio", "sabio_enz_inh.pkl"))
print(data_report(substrate_df))
print(data_report(inhibitors_df))