import os
from os.path import join
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import re
import concurrent.futures
import requests
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import sys

sys.path.append("./../utilities")
from utilities.helper_functions import *
from utilities.thresholds import *

# Define the current directory
CURRENT_DIR = os.getcwd()
print("Current Directory:", CURRENT_DIR)

# Define the columns to keep
# columns_to_keep = [
#     "Ligand SMILES",
#     "Ki (nM)",
#     "IC50 (nM)",
#     "PubChem CID",
#     "EC50 (nM)",
#     "Kd (nM)",
#     "UniProt (SwissProt) Primary ID of Target Chain",
# ]
#
# # Read the BindingDB file
# BindingDB = pd.read_csv(
#     join(CURRENT_DIR, "..", "data", "raw_data", "BindingDB_All.tsv"),
#     sep="\t",
#     usecols=columns_to_keep,
#     on_bad_lines="skip"
# )
# BindingDB.rename(columns={'UniProt (SwissProt) Primary ID of Target Chain': 'Uni_SwissProt',
#                           'Ligand SMILES': 'SMILES', "PubChem CID":"molecule_ID"}, inplace=True)
# # Drop rows where "UniProt (SwissProt) Primary ID of Target Chain" is NaN
# BindingDB = BindingDB.dropna(subset=["Uni_SwissProt"])
# # Drop rows where both "IC50 (nM)" and "Ki (nM)" are NaN
# BindingDB = BindingDB.dropna(subset=["IC50 (nM)", "Ki (nM)","EC50 (nM)","Kd (nM)"], how="all")
# # Function to clean and convert a column to numeric
# def clean_and_convert(column):
#     column = column.str.replace(r"^[><]", "", regex=True)
#     return pd.to_numeric(column, errors="coerce")
#
#
# BindingDB["IC50 (nM)"] = clean_and_convert(BindingDB["IC50 (nM)"])
# BindingDB["Ki (nM)"] = clean_and_convert(BindingDB["Ki (nM)"])
# BindingDB["Kd (nM)"] = clean_and_convert(BindingDB["Kd (nM)"])
# BindingDB["EC50 (nM)"] = clean_and_convert(BindingDB["EC50 (nM)"])
# # =========================== Remove cofactors and energy molecules ========================
# # check additional_code folder to see how cofactors_list.txt has been created
# with open(join(CURRENT_DIR, "..", "data", "processed_data","cofactors_list.txt"), "r") as f:
#     remove_cofactor_energy_ids = [line.strip() for line in f.readlines()]
# BindingDB.dropna(subset=["molecule_ID"], inplace=True)
# BindingDB["molecule_ID"] = BindingDB["molecule_ID"].astype(int)
# BindingDB = BindingDB.loc[~BindingDB["molecule_ID"].isin(remove_cofactor_energy_ids)]
# ###########################################################################################
#
#
# # Get unique UniProt IDs
# unique_uniprots = list(set(BindingDB["Uni_SwissProt"].tolist()))
#
# print(f"\nStarting processing of {len(unique_uniprots)} UniProt IDs...")
#
# # Process with parallel execution and progress bar
# with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
#     # Wrap with tqdm for progress bar
#     futures = [executor.submit(is_enzyme, uid) for uid in unique_uniprots]
#
#     # Initialize counters
#     success = 0
#     no_ec = 0
#     errors = 0
#
#     for future in tqdm(concurrent.futures.as_completed(futures),
#                        total=len(unique_uniprots),
#                        desc="Processing"):
#         result = future.result()
#         if result is None:
#             no_ec += 1
#         elif isinstance(result, str):
#             success += 1
#         else:
#             errors += 1
#
# print(f"\n\nSummary:")
# print(f"✅ Successfully processed: {success}")
# print(f"🟡 No EC numbers found: {no_ec}")
# print(f"🔴 Errors encountered: {errors}")
#
# # Create a dictionary of UniProt IDs to EC numbers
# uniprot_to_ec = {}
# for uid, future in zip(unique_uniprots, futures):
#     ec = future.result()
#     if ec:
#         uniprot_to_ec[uid] = ec
#
# # Add EC number information to the DataFrame
# BindingDB["EC_ID"] = BindingDB["Uni_SwissProt"].map(uniprot_to_ec)
# BindingDB.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data", "bindingDB", "BindingDB_curated_v1.pkl"))
# BindingDB = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data", "bindingDB", "BindingDB_curated_v1.pkl"))

# ###########################################################################################
# uniprot_enz_inh = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data","uniprot", "7-1-uniprot_enz-inh.pkl"))
# brenda_enz_inh = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data","brenda", "5-3-brenda_enz_inh.pkl"))
# uniprot_enz_sub = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data","uniprot", "2-1-uniprot_enz_sub.pkl"))
# brenda_enz_sub = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data","brenda", "5-3-brenda_enz_sub.pkl"))
# rhea_enz_sub = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data", "rhea","3-rhea_enz_sub.pkl"))
# gobo_enz_sub = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data", "GO","4-1-gobo_enz_sub.pkl"))
# pubchem=pd.read_csv(join(CURRENT_DIR, "..", "data", "processed_data","Pubchem", "8-1-pubchem_assays_combined.csv"))
# sabio_data=pd.read_csv(join(CURRENT_DIR, "..", "data", "processed_data", "sabio", "10-sabio_rk_results.tsv" ), sep='\t')
# luphar_enz_inh=pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data","luphar", "11-luphar_enz_inh.pkl"))
# chembl_binding_activities = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data","chembl", "6-2-chembl_binding_activities.pkl"))
#
# unique_uniprots= set(list(uniprot_enz_inh["ID"])+ list(brenda_enz_inh["Uni_SwissProt"]) +
#                      list(uniprot_enz_sub["ID"])+ list(brenda_enz_sub["Uni_SwissProt"]) +
#                      list(rhea_enz_sub["Uni_SwissProt"] )+ list(gobo_enz_sub["Uniprot_ID"]) +
#                      list(chembl_binding_activities["Uniprot_ID"]) +
#                      list(pubchem["UniProt ID"]) +
#                      list(sabio_data["UniprotID"]) +
#                      list(luphar_enz_inh["Uni_SwissProt"])
#                      )
#
# BindingDB = BindingDB[
#     BindingDB['Uni_SwissProt'].isin(unique_uniprots) |
#     BindingDB['EC_ID'].notna()
# ]
###########################################################################################
# uniprot_IDs = list(set(BindingDB["Uni_SwissProt"].unique().tolist()))
# print(len(uniprot_IDs))
# sequences = get_protein_sequences_with_retry(uniprot_IDs)
# sequence_df = pd.DataFrame(list(sequences.items()), columns=['Uni_SwissProt', 'Protein_Sequence'])
# sequence_df.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data","bindingDB", "BindingDB_ProId2seq.pkl"))

# sequence_df = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data", "bindingDB", "BindingDB_ProId2seq.pkl"))
# uni_ids2seq_dict = dict(zip(sequence_df['Uni_SwissProt'],sequence_df['Protein_Sequence']))
#
# BindingDB['Protein_Sequence'] = BindingDB['Uni_SwissProt'].map(uni_ids2seq_dict)
# BindingDB.dropna(subset=["Protein_Sequence"], inplace=True)
# BindingDB.dropna(subset=["Uni_SwissProt"], inplace=True)
# BindingDB.reset_index(drop=True, inplace=True)
# BindingDB.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data", "bindingDB", "BindingDB_curated_v2.pkl"))

BindingDB = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data", "bindingDB", "BindingDB_curated_v2.pkl"))
print(BindingDB.columns.tolist())

conditions = [
    BindingDB['Kd (nM)'].notna(),
    BindingDB['Ki (nM)'].notna(),
    BindingDB['IC50 (nM)'].notna(),
    BindingDB['EC50 (nM)'].notna()
]

choices_type = ['Kd', 'Ki', 'IC50', 'EC50']
choices_value = [
    BindingDB['Kd (nM)'],
    BindingDB['Ki (nM)'],
    BindingDB['IC50 (nM)'],
    BindingDB['EC50 (nM)']
]

BindingDB['standard_type'] = np.select(conditions, choices_type, default=None)
BindingDB['standard_value'] = np.select(conditions, choices_value, default=np.nan)
BindingDB.dropna(subset=['standard_type', 'standard_value'], inplace=True)
BindingDB.loc[BindingDB['standard_type'] == "IC50", 'standard_value'] *= 0.50
###########################################################################################

BindingDB_enz_inh = BindingDB[
    (BindingDB['standard_type'].isin(["IC50", "Ki"])) &
    (BindingDB['standard_value'] > 0) &
    (BindingDB['standard_value'] <= EnInh_KiIcMic_nM)
]
BindingDB_enz_ni = BindingDB[
    (BindingDB['standard_type'].isin(["Ki", 'Kd', "IC50", 'EC50'])) & (BindingDB['standard_value'] >= EnNi_KiECKdIC_nM)]
########################################################################################
BindingDB_enz_inh["Evidence"] = "BAA"
BindingDB_enz_inh["activity_comment"] = None

BindingDB_enz_ni["Evidence"] = "BAA"
BindingDB_enz_ni["activity_comment"] = None

BindingDB_enz_inh = BindingDB_enz_inh[
    ['Uni_SwissProt', 'molecule_ID', 'Protein_Sequence', 'standard_type', 'standard_value', 'SMILES', 'EC_ID',
     'Evidence', 'activity_comment']]
BindingDB_enz_ni = BindingDB_enz_ni[
    ['Uni_SwissProt', 'molecule_ID', 'Protein_Sequence', 'standard_type', 'standard_value', 'SMILES', 'EC_ID',
     'Evidence', 'activity_comment']]
BindingDB_enz_inh.reset_index(drop=True, inplace=True)
BindingDB_enz_ni.reset_index(drop=True, inplace=True)
BindingDB_enz_inh.to_pickle(
    join(CURRENT_DIR, "..", "data", "processed_data", "bindingDB", f"9-BindingDB_enz_inh.pkl"))

BindingDB_enz_ni.to_pickle(
    join(CURRENT_DIR, "..", "data", "processed_data", "bindingDB", f"9-BindingDB_enz_ni.pkl"))

print(data_report(BindingDB_enz_inh))
print(data_report(BindingDB_enz_ni))
print(BindingDB_enz_ni["standard_type"].unique())
print(BindingDB_enz_inh["standard_type"].unique())
