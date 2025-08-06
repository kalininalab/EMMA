import sys
import warnings
import pandas as pd
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from os.path import join

sys.path.append("./../utilities")
from helper_functions import *
from thresholds import *

warnings.filterwarnings("ignore")

CURRENT_DIR = os.getcwd()
base_DIR = join(CURRENT_DIR, "..", "data", "processed_data")

############################################################################
# Load and Process Data
####################
# Load all datasets
binding_activities = pd.read_pickle(join(base_DIR, "chembl", "6-5-chembl_binding_activities.pkl"))
uniprot_enz_inh = pd.read_pickle(join(base_DIR, "uniprot", "7-2-uniprot_enz-inh.pkl"))
brenda_enz_inh = pd.read_pickle(join(base_DIR, "brenda", "5-3-brenda_enz_inh.pkl"))
uniprot_enz_sub = pd.read_pickle(join(base_DIR, "uniprot", "2-2-uniprot_enz-sub.pkl"))
brenda_enz_sub = pd.read_pickle(join(base_DIR, "brenda", "5-3-brenda_enz_sub.pkl"))
rhea_enz_sub = pd.read_pickle(join(base_DIR, "rhea", "3-rhea_enz_sub.pkl"))
gobo_enz_sub = pd.read_pickle(join(base_DIR, "GO", "4-3-gobo_enz_sub.pkl"))
pubchem_enz_ni = pd.read_pickle(join(base_DIR, "pubchem", "8-2-pubchem_enz_ni.pkl"))
pubchem_enz_inh = pd.read_pickle(join(base_DIR, "pubchem", "8-2-pubchem_enz_inh.pkl"))
bindingDB_enz_inh = pd.read_pickle(join(base_DIR, "bindingDB", "9-BindingDB_enz_inh.pkl"))
bindingDB_enz_ni = pd.read_pickle(join(base_DIR, "bindingDB", "9-BindingDB_enz_ni.pkl"))
sabio_enz_sub = pd.read_pickle(join(base_DIR, "sabio", "10-sabio_enz_sub.pkl"))
sabio_enz_inh = pd.read_pickle(join(base_DIR, "sabio", "10-sabio_enz_inh.pkl"))
luphar_enz_ni = pd.read_pickle(join(base_DIR, "luphar", "11-luphar_enz_ni.pkl"))
luphar_enz_inh = pd.read_pickle(join(base_DIR, "luphar", "11-luphar_enz_inh.pkl"))

############################################################################
# Sabio Enzyme-Non-Substrate
####################
sabio_enz_ni_1 = sabio_enz_sub[
    (sabio_enz_sub['standard_value'].notna()) & (sabio_enz_sub['standard_value'] >= EnNi_KiECKdIC_nM)]
sabio_enz_sub_1 = sabio_enz_sub[
    (sabio_enz_sub['standard_value'].notna()) & (sabio_enz_sub['standard_value'].between(0.0001, EnSub_km_nM))]
sabio_enz_ni_2 = sabio_enz_inh[
    (sabio_enz_inh['standard_value'].notna()) & (sabio_enz_inh['standard_value'] >= EnNi_KiECKdIC_nM)]
sabio_enz_inh_1 = sabio_enz_inh[
    (sabio_enz_inh['standard_value'].notna()) & (sabio_enz_inh['standard_value'].between(0.0001, EnInh_KiECKdIC_nM))]
############################################################################
# Brenda Enzyme-Non-Substrate
####################
brenda_enz_sub_1 = brenda_enz_sub[
    (brenda_enz_sub['standard_value'].notna()) & (brenda_enz_sub['standard_value'].between(0.0001, EnSub_km_nM))]
brenda_enz_sub_2 = brenda_enz_sub[(brenda_enz_sub['standard_value'].isna()) & (brenda_enz_sub['Evidence'] == "EEC")]
brenda_enz_ni_1 = brenda_enz_sub[
    (brenda_enz_sub['standard_value'].notna()) & (brenda_enz_sub['standard_value'] >= EnNi_km_nM)]
brenda_enz_ni_2 = brenda_enz_inh[
    (brenda_enz_inh['standard_value'].notna()) & (brenda_enz_inh['standard_value'] >= EnNi_KiECKdIC_nM)]
brenda_enz_inh_1 = brenda_enz_inh[
    (brenda_enz_inh['standard_value'].notna()) & (brenda_enz_inh['standard_value'].between(0.0001, EnInh_KiECKdIC_nM))]
brenda_enz_inh_2 = brenda_enz_inh[(brenda_enz_inh['standard_value'].isna()) & (brenda_enz_inh['Evidence'] == "EEC")]

############################################################################
# Chembl Enzyme-Substrate
####################
chembl_enz_sub, chembl_enz_ni_1 = chembl_enzyme_substrate(binding_activities)
dict_chembl_enz_sub = chembl_enz_sub.groupby('standard_type')['standard_units'].apply(set).to_dict()
print(f"Selected assays for substrate: {dict_chembl_enz_sub}")
dict_chembl_enz_ni_1 = chembl_enz_ni_1.groupby('standard_type')['standard_units'].apply(set).to_dict()
print(f"Selected assays for enz_ni: {dict_chembl_enz_ni_1}")
chembl_enz_sub["Evidence"] = "BAA"
chembl_enz_ni_1["Evidence"] = "BAA"

chembl_enz_sub = chembl_enz_sub[
    ['Uni_SwissProt', 'molecule_ID', 'Protein_Sequence', 'standard_type', 'standard_value', 'SMILES', "EC_ID",
     "Evidence", "activity_comment"]]
############################################################################
# Chembl Enzyme-non-interacting Pair
####################
chembl_enz_ni_1 = chembl_enz_ni_1[
    ['Uni_SwissProt', 'molecule_ID', 'Protein_Sequence', 'standard_type', 'standard_value', 'SMILES', "EC_ID",
     "Evidence", "activity_comment"]]
enz_ni_commented, enz_ni_notcommented = chembl_low_affinity_pairs(binding_activities)

dict_enz_ni_commented = enz_ni_commented.groupby('standard_type')['standard_units'].apply(set).to_dict()
print(f"Selected assays for non_interacting_commented: {dict_enz_ni_commented}")
dict_enz_ni_notcommented = enz_ni_notcommented.groupby('standard_type')['standard_units'].apply(set).to_dict()
print(f"Selected assays for non_interacting_notcommented: {dict_enz_ni_notcommented}")
chembl_enz_ni_2 = pd.concat([enz_ni_commented, enz_ni_notcommented])
chembl_enz_ni_2["Evidence"] = "BAA"
chembl_enz_ni_2 = chembl_enz_ni_2[
    ['Uni_SwissProt', 'molecule_ID', 'Protein_Sequence', 'standard_type', 'standard_value', 'SMILES', "EC_ID",
     "Evidence", "activity_comment"]]
chembl_enz_ni_2.drop_duplicates(subset=['Uni_SwissProt', 'molecule_ID'], keep='first', inplace=True)
############################################################################
# Chembl Enzyme-Inhibitors
####################

enz_inh_commented, enz_inh_notcommented = chembl_enzyme_inhibitor(binding_activities)
grouped_dict_commented = enz_inh_commented.groupby('standard_type')['standard_units'].apply(set).to_dict()
print(f"Selected assays for enz_inh_commented: {grouped_dict_commented}")
grouped_dict_notcommented = enz_inh_notcommented.groupby('standard_type')['standard_units'].apply(set).to_dict()
print(f"Selected assays for enz_inh_notcommented: {grouped_dict_notcommented}")
chembl_enz_inh = pd.concat([enz_inh_commented, enz_inh_notcommented])
chembl_enz_inh["Evidence"] = "BAA"
chembl_enz_inh = chembl_enz_inh[
    ['Uni_SwissProt', 'molecule_ID', 'Protein_Sequence', 'standard_type', 'standard_value', 'SMILES', "EC_ID",
     "Evidence", "activity_comment"]]
chembl_enz_inh.drop_duplicates(subset=['Uni_SwissProt', 'molecule_ID'], keep='first', inplace=True)
############################################################################
# Enzyme-Inhibitor Final Dataset
####################
datasets_inh = [
    (brenda_enz_inh_1, 0, "Brenda"),
    (brenda_enz_inh_2, 0, "Brenda"),
    (chembl_enz_inh, 0, "ChEMBL"),
    (uniprot_enz_inh, 0, "Uniprot"),
    (bindingDB_enz_inh, 0, "BindingDB"),
    (sabio_enz_inh_1, 0, "Sabio"),
    (pubchem_enz_inh, 0, "Pubchem"),
    (luphar_enz_inh, 0, "Iuphar")
]
enz_inh_final = pd.concat([add_binding_source(df, binding_value, source) for df, binding_value, source in datasets_inh],
                          axis=0)
enz_inh_final = canonicalize_and_duplicate_and_len_constraint(enz_inh_final)
print(data_report(enz_inh_final))

############################################################################
# Enzyme-Non-Substrate Final Dataset
###################
datasets_enz_ni = [
    (chembl_enz_ni_1, 2, "ChEMBL"),
    (chembl_enz_ni_2, 2, "ChEMBL"),
    (bindingDB_enz_ni, 2, "BindingDB"),
    (brenda_enz_ni_1, 2, "Brenda"),
    (brenda_enz_ni_2, 2, "Brenda"),
    (sabio_enz_ni_1, 2, "Sabio"),
    (sabio_enz_ni_2, 2, "Sabio"),
    (pubchem_enz_ni, 2, "Pubchem")
    # (luphar_enz_ni, 2,"Iuphar")

]

enz_ni_final = pd.concat(
    [add_binding_source(df, binding_value, source) for df, binding_value, source in datasets_enz_ni], axis=0)
enz_ni_final = canonicalize_and_duplicate_and_len_constraint(enz_ni_final)
print(data_report(enz_ni_final))
############################################################################
# Enzyme-Substrate Final Dataset
####################
datasets = [
    (brenda_enz_sub_1, 1, "Brenda"),
    (gobo_enz_sub, 1, "Gobo"),
    (rhea_enz_sub, 1, "RHEA"),
    (uniprot_enz_sub, 1, "Uniprot"),
    (brenda_enz_sub_2, 1, "Brenda"),
    (sabio_enz_sub_1, 1, "Sabio"),
    (chembl_enz_sub, 1, "ChEMBL")
]

enz_sub_final = pd.concat([add_binding_source(df, binding_value, source) for df, binding_value, source in datasets],
                          axis=0)
enz_sub_final = canonicalize_and_duplicate_and_len_constraint(enz_sub_final)

# plot_top_keys_values(enz_sub_final, "Uni_SwissProt",
#                      "uniprot", "molecule ID",
#                      "distribution", color='blue',
#                      figsize=(12, 10), top_count=10000)

#
# unique_pro = set(enz_ni_final["Uni_SwissProt"].tolist() + enz_inh_final["Uni_SwissProt"].tolist())
# enz_sub_final = enz_sub_final[enz_sub_final['Uni_SwissProt'].isin(unique_pro)]
print(data_report(enz_sub_final))
############################################################################
# Similarity based Down sampling
####################
df_inh, df_enz_ni = precompute_fingerprints(enz_inh_final, enz_ni_final)

# Select molecules us'ing per-substrate similarity
print("\nDiversity-based down sampling...")

selected_inh, selected_enz_ni = down_sampling(
    df_inh,
    df_enz_ni,
    max_inh_per_enzyme=4,
    max_ni_per_enzyme=4,
)

enz_ni_final = enz_ni_final[enz_ni_final['molecule_ID'].isin(selected_enz_ni)]
enz_inh_final = enz_inh_final[enz_inh_final['molecule_ID'].isin(selected_inh)]
enz_inh_final.drop('fp', axis=1, inplace=True)
enz_ni_final.drop('fp', axis=1, inplace=True)
enz_ni_final.reset_index(drop=True, inplace=True)
enz_inh_final.reset_index(drop=True, inplace=True)
print(data_report(enz_inh_final))
print(data_report(enz_ni_final))
print(data_report(enz_sub_final))

molecule_distribution(df_inh, enz_inh_final, "Inhibitors")

molecule_distribution(df_enz_ni, enz_ni_final, "LAPs")
############################################################################
# Final Dataset
####################

final_dataset = pd.concat([enz_inh_final, enz_sub_final, enz_ni_final], axis=0)
final_dataset = filter_invalid_sequences(final_dataset)
final_dataset.dropna(subset=['SMILES', 'molecule_ID'], inplace=True)
# Save final dataset
final_dataset.drop_duplicates(inplace=True)
final_dataset["Mainclass"] = final_dataset["Binding"].apply(lambda x: 0 if x == 2 else 1)
final_dataset["Subclass"] = final_dataset["Binding"].apply(lambda x: -1 if x == 2 else x)

canonical_to_id = (final_dataset.drop_duplicates(subset=['SMILES']).set_index('SMILES')['molecule_ID'])
final_dataset['molecule_ID'] = final_dataset['SMILES'].map(canonical_to_id)

seq_to_id = (final_dataset.drop_duplicates(subset=['Protein_Sequence']).set_index('Protein_Sequence')['Uni_SwissProt'])
final_dataset['Uni_SwissProt'] = final_dataset['Protein_Sequence'].map(seq_to_id)

final_dataset.drop_duplicates(subset=['Protein_Sequence', 'SMILES'], keep='first', inplace=True)
final_dataset.drop_duplicates(subset=['Protein_Sequence', 'molecule_ID'], keep='first', inplace=True)
final_dataset.drop_duplicates(subset=['Uni_SwissProt', 'SMILES'], keep='first', inplace=True)
final_dataset.drop_duplicates(subset=['Uni_SwissProt', 'molecule_ID'], keep='first', inplace=True)
final_dataset.reset_index(drop=True, inplace=True)
enz_sub = final_dataset.loc[final_dataset['Binding'] == 1]
print(data_report(enz_sub))
print(enz_sub["standard_type"].unique().tolist())
enz_inh = final_dataset.loc[final_dataset['Binding'] == 0]
print(data_report(enz_inh))
print(enz_inh["standard_type"].unique().tolist())
enz_lap = final_dataset.loc[final_dataset['Binding'] == 2]
print(data_report(enz_lap))
print(enz_lap["standard_type"].unique().tolist())
final_dataset.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data", "Final_Dataset.pkl"))

############################################################################
# Data Report
####################
print(data_report(final_dataset))
unique_molecule_ids = final_dataset['molecule_ID'].dropna().unique()
# Initialize lists for each identifier type
chebi_ids = []
pubchem_ids = []
kegg_ids = []
chembel_ids = []

# Categorize each identifier
for molecule_id in unique_molecule_ids:
    molecule_id = str(molecule_id)
    if re.match(r'^CHEBI:', molecule_id):
        chebi_ids.append(molecule_id)
    elif re.match(r'^CHEMBL', molecule_id):
        chembel_ids.append(molecule_id)
    elif re.match(r'^\d+$', molecule_id):
        pubchem_ids.append(molecule_id)
    elif re.match(r'^C\d+', molecule_id):
        kegg_ids.append(molecule_id)
print(f"Number of substrates: {len(final_dataset.loc[final_dataset['Binding'] == 1])}")
print(f"Number of inhibitors: {len(final_dataset.loc[final_dataset['Binding'] == 0])}")
print(f"Number of non-substarte: {len(final_dataset.loc[final_dataset['Binding'] == 2])}")
print(
    f"Ratio of  of enz-inh to enz-sub : {round(len(final_dataset.loc[final_dataset['Binding'] == 0]) / len(final_dataset.loc[final_dataset['Binding'] == 1]), 2)}")
print(
    f"Ratio of  of enz-inh to enz-ni : {round(len(final_dataset.loc[final_dataset['Binding'] == 0]) / len(final_dataset.loc[final_dataset['Binding'] == 2]), 2)}")

print(
    f"Ratio of  of enz-sub  to non-sub : {round(len(final_dataset.loc[final_dataset['Binding'] == 1]) / len(final_dataset.loc[final_dataset['Binding'] == 2]), 2)}")

# Define label names mapping
label_names = {0: "enz-inh", 1: "enz-sub", 2: "enz-ni"}
for source in final_dataset['Source'].unique():
    counts = {label: len(final_dataset[(final_dataset['Binding'] == label) &
                                       (final_dataset['Source'] == source)])
              for label in label_names}
    print(f"Counts for {source}: {counts}")

print(f"Number of Pubchem IDs: {len(pubchem_ids)}")
print(f"Number of ChEBI IDs: {len(chebi_ids)}")
print(f"Number of Chembel IDs: {len(chembel_ids)}")
print(f"Number of KEGG IDs: {len(kegg_ids)}")

# Group by 'Uni_SwissProt' and count the number of unique labels for each group
label_counts = final_dataset.groupby('Uni_SwissProt')['Binding'].nunique()

# Count how many Uni_SwissProt have 1, 2, or 3 unique labels
count_1_label = (label_counts == 1).sum()
count_2_labels = (label_counts == 2).sum()
count_3_labels = (label_counts == 3).sum()

# Total number of unique Uni_SwissProt entries
total_uni = len(label_counts)

# Calculate percentages
percent_1_label = (count_1_label / total_uni) * 100
percent_2_labels = (count_2_labels / total_uni) * 100
percent_3_labels = (count_3_labels / total_uni) * 100

print(f"Percentage of Uni_SwissProt with 1 label: {percent_1_label:.2f}%")
print(f"Percentage of Uni_SwissProt with 2 labels: {percent_2_labels:.2f}%")
print(f"Percentage of Uni_SwissProt with 3 labels: {percent_3_labels:.2f}%")
