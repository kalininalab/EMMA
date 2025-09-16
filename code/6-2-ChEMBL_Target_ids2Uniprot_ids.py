from collections import defaultdict
import pandas as pd
from chembl_webresource_client.new_client import new_client
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys
import warnings
from os.path import join

sys.path.append("./../utilities")
from helper_functions import *

CURRENT_DIR = os.getcwd()
warnings.filterwarnings("ignore")

target_client = new_client.target
chembl_binding_activities = pd.read_pickle(
    join(CURRENT_DIR, "..", "data", "processed_data", "chembl", "6-1-chembl_binding_activities.pkl"))
unique_target_ids = chembl_binding_activities["target_chembl_id"].unique()


def map_target_ids_to_uniprot_and_go_ids(unique_target_ids):
    uniprot_id_map = defaultdict(list)
    total_target_ids = len(unique_target_ids)
    for target_id in unique_target_ids:
        target = target_client.get(target_id)
        if 'target_components' in target:
            for component in target['target_components']:
                if 'target_component_synonyms' in component:
                    for component_syn in component['target_component_synonyms']:
                        if component_syn.get('syn_type') == 'EC_NUMBER':
                            for xref in component['target_component_xrefs']:
                                if xref.get('xref_src_db') == 'GoFunction':
                                    uniprot_id_map[target_id].append(
                                        (component['accession'], component_syn['component_synonym'], xref['xref_id'],
                                         component['component_description'], xref['xref_name']))
                                    print(
                                        f"Target Chembl ID: {target_id} mapped to Uniprot ID: {component['accession']} and GO_ID: {xref['xref_id']}")
        total_target_ids -= 1
        print(f"Not found, remaining target ids: {total_target_ids}")

    uniprot_ec_data = []
    for key, value in uniprot_id_map.items():
        for item in value:
            uniprot_ec_data.append({'target_chembl_id': key, 'Uniprot_ID': item[0], 'EC_ID': item[1], "GO_ID": item[2],
                                    "component_description": item[3], "GO_name": item[4]})
    uniprot_go_df = pd.DataFrame(uniprot_ec_data)
    return uniprot_go_df


# uniprot_go= map_target_ids_to_uniprot_and_go_ids(unique_target_ids)
# uniprot_go.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data","chembl", "Chembl_target2uniprot.pkl"))


uniprot_go = pd.read_pickle(join(CURRENT_DIR, "..", "data", "processed_data", "chembl", "Chembl_target2uniprot.pkl"))
print(data_report(uniprot_go))
uniprot_id_map = dict(zip(uniprot_go['target_chembl_id'], uniprot_go['Uniprot_ID']))
chembl_binding_activities['Uniprot_ID'] = chembl_binding_activities['target_chembl_id'].map(uniprot_id_map)
ec_id_map = dict(zip(uniprot_go['target_chembl_id'], uniprot_go['EC_ID']))
chembl_binding_activities['EC_ID'] = chembl_binding_activities['target_chembl_id'].map(ec_id_map)
chembl_binding_activities.dropna(subset=['Uniprot_ID'], inplace=True)
chembl_binding_activities.reset_index(drop=True, inplace=True)
chembl_binding_activities.to_pickle(
    join(CURRENT_DIR, "..", "data", "processed_data", "chembl", "6-2-chembl_binding_activities.pkl"))
print(data_report(chembl_binding_activities))
