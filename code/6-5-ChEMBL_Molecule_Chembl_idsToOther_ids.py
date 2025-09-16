import pandas as pd
import requests
import time
import re
import sys
import warnings
import os
import math
from os.path import join

sys.path.append("./../utilities")
from utilities.helper_functions import *

CURRENT_DIR = os.getcwd()
warnings.filterwarnings("ignore")

chembl_binding_activities = pd.read_pickle(
    join(CURRENT_DIR, "..", "data", "processed_data", "chembl", "6-4-chembl_binding_activities.pkl"))

# Source mapping tables:
# https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/general-questions
chembl_chebi = pd.read_csv(join(CURRENT_DIR, "..", "data", "raw_data", "map_chembl_to_other_ids", "src1src7.txt"),
                           sep='\t')
chembl_chebi = chembl_chebi.rename(columns={"From src:'1'": "ChEMBL_ID", "To src:'7'": 'ChEBI_ID'})
chembl_chebi["ChEBI_ID"] = chembl_chebi["ChEBI_ID"].apply(lambda x: "CHEBI:" + str(x))
chembl_chebi_dict = dict(zip(chembl_chebi["ChEMBL_ID"], chembl_chebi["ChEBI_ID"]))

chembl_binding_activities["ChEBI_ID"] = chembl_binding_activities["molecule_chembl_id"].map(chembl_chebi_dict)

chembl_kegg = pd.read_csv(join(CURRENT_DIR, "..", "data", "raw_data", "map_chembl_to_other_ids", "src1src6.txt"),
                          sep='\t')
chembl_kegg = chembl_kegg.rename(columns={"From src:'1'": "ChEMBL_ID", "To src:'6'": 'KEGG_ID'})
chembl_kegg_dict = dict(zip(chembl_kegg["ChEMBL_ID"], chembl_kegg["KEGG_ID"]))

chembl_binding_activities["KEGG_ID"] = chembl_binding_activities["molecule_chembl_id"].map(chembl_kegg_dict)

chembl_pubchem = pd.read_csv(join(CURRENT_DIR, "..", "data", "raw_data", "map_chembl_to_other_ids", "src1src22.txt"),
                             sep='\t')
chembl_pubchem = chembl_pubchem.rename(columns={"From src:'1'": "ChEMBL_ID", "To src:'22'": 'PubChem_ID'})
chembl_pubchem_dict = dict(zip(chembl_pubchem["ChEMBL_ID"], chembl_pubchem["PubChem_ID"]))

chembl_binding_activities["PubChem_ID"] = chembl_binding_activities["molecule_chembl_id"].map(chembl_pubchem_dict)
chembl_binding_activities["PubChem_ID"] = pd.to_numeric(chembl_binding_activities["PubChem_ID"],
                                                        errors='coerce').astype('Int64')

chembl_binding_activities['molecule_ID'] = [
    row['ChEBI_ID'] if pd.notnull(row['ChEBI_ID']) else
    row['KEGG_ID'] if pd.notnull(row['KEGG_ID']) else
    row['PubChem_ID'] if pd.notnull(row['PubChem_ID']) else
    row['molecule_chembl_id']
    for index, row in chembl_binding_activities.iterrows()
]

chembl_binding_activities['molecule_IDs'] = chembl_binding_activities.apply(
    lambda row: [row[col] for col in ['ChEBI_ID', 'KEGG_ID', 'PubChem_ID', 'molecule_chembl_id'] if
                 pd.notnull(row[col])],
    axis=1
)

# Please check additional_code folder to see how cofactors_list.txt has been created
with open(join(CURRENT_DIR, "..", "data", "processed_data", "cofactors_list.txt"), "r") as f:
    remove_cofactor_energy_ids = [line.strip() for line in f.readlines()]

chembl_binding_activities = chembl_binding_activities.loc[
    ~chembl_binding_activities["molecule_ID"].isin(remove_cofactor_energy_ids)]

chembl_binding_activities.reset_index(drop=True, inplace=True)
chembl_binding_activities = chembl_binding_activities.rename(
    columns={'Uniprot_ID': 'Uni_SwissProt', 'canonical_smiles': 'SMILES'})
chembl_binding_activities.to_pickle(
    join(CURRENT_DIR, "..", "data", "processed_data", "chembl", "6-5-chembl_binding_activities.pkl"))
print(data_report(chembl_binding_activities))
