import sys
import warnings
import requests

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

enz_sub = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data", "brenda", "5-1-brenda_enz_sub.pkl"))
enz_sub_2 = enz_sub.copy()
enz_inh = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data", "brenda", "5-1-brenda_enz_inh.pkl"))
enz_inh_2 = enz_inh.copy()

#######################################
# Pubchem
#######################################
# sub_unique = enz_sub_2['Substrate'].unique().tolist()
# inh_unique = enz_inh_2['Inhibitor'].unique().tolist()
# combined_unique = list(set(sub_unique + inh_unique))
# pubID_dict = get_pubchem_ids(combined_unique)
# name_to_cid_df = pd.DataFrame(list(pubID_dict.items()), columns=['molecule_name', 'ID'])
# name_to_cid_df.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data","brenda", "Brenda_MolName2cid.pkl"))

name_to_cid_df = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data", "brenda", "Brenda_MolName2cid.pkl"))
pubID_dict = dict(zip(name_to_cid_df['molecule_name'], name_to_cid_df['ID']))
enz_sub_2['molecule_ID'] = enz_sub_2['Substrate'].map(pubID_dict)
enz_inh_2['molecule_ID'] = enz_inh_2['Inhibitor'].map(pubID_dict)
print(data_report(enz_sub_2))
print(data_report(enz_inh_2))

mol2ids = pd.read_pickle(join(CURRENT_DIR, "..", "data", "raw_data", "molecule_name_to_ids", "mol2ids.pkl"))
mol2ids.drop_duplicates(subset=['molecule_name'], keep='first', inplace=True)
enz_sub_2["Substrate"] = [name.lower() for name in enz_sub_2["Substrate"]]
enz_inh_2["Inhibitor"] = [name.lower() for name in enz_inh_2["Inhibitor"]]
mol2ids_dict = dict(zip(mol2ids['molecule_name'], mol2ids['molecule_ID']))
enz_sub_2['molecule_ID'].fillna(enz_sub_2['Substrate'].map(mol2ids_dict), inplace=True)
enz_inh_2['molecule_ID'].fillna(enz_inh_2['Inhibitor'].map(mol2ids_dict), inplace=True)
print(data_report(enz_sub_2))
print(data_report(enz_inh_2))
##########################################################
enz_sub_2.dropna(subset=['molecule_ID'], inplace=True)
enz_inh_2.dropna(subset=['molecule_ID'], inplace=True)
enz_sub_2.reset_index(drop=True, inplace=True)
enz_sub_2.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data", "brenda", "5-2-brenda_enz_sub.pkl"))
enz_inh_2.reset_index(drop=True, inplace=True)
enz_inh_2.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data", "brenda", "5-2-brenda_enz_inh.pkl"))
print(data_report(enz_sub_2))
print(data_report(enz_inh_2))
