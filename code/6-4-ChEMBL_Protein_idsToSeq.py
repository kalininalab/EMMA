import pandas as pd
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
    join(CURRENT_DIR, "..", "data", "processed_data", "chembl", "6-3-chembl_binding_activities.pkl"))

uniprot_IDs = chembl_binding_activities["Uniprot_ID"].unique().tolist()
print(len(uniprot_IDs))
sequences = get_protein_sequences_with_retry(uniprot_IDs)
chembl_sequence_df = pd.DataFrame(list(sequences.items()), columns=['Uniprot_ID', 'Protein_Sequence'])
chembl_sequence_df.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data", "chembl", "chembl_ProId2seq_v2.pkl"))

chembl_sequence_df = pd.read_pickle(
    join(CURRENT_DIR, "..", "data", "processed_data", "chembl", "chembl_ProId2seq_v2.pkl"))
uni_ids2seq_dict = dict(zip(chembl_sequence_df['Uniprot_ID'], chembl_sequence_df['Protein_Sequence']))

chembl_binding_activities['Protein_Sequence'] = chembl_binding_activities['Uniprot_ID'].map(uni_ids2seq_dict)
chembl_binding_activities.dropna(subset=["Uniprot_ID"], inplace=True)
chembl_binding_activities.reset_index(drop=True, inplace=True)
chembl_binding_activities.to_pickle(
    join(CURRENT_DIR, "..", "data", "processed_data", "chembl", "6-4-chembl_binding_activities.pkl"))
print(data_report(chembl_binding_activities))
