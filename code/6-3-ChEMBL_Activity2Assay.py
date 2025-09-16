import pandas as pd
from chembl_webresource_client.new_client import new_client
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from requests.exceptions import ConnectionError

assay_client = new_client.assay

chembl_binding_activities = pd.read_pickle(
    join(CURRENT_DIR, "..", "data", "processed_data", "chembl", "6-2-chembl_binding_activities.pkl"))
print(data_report(chembl_binding_activities))
chembl_binding_activities["relationship_type"] = None
chembl_binding_activities["confidence_score"] = None
chembl_binding_activities["src_id"] = None


def map_activity_to_assay(chembl):
    data = chembl
    assay_data = {
        'relationship_type': {},
        'confidence_score': {},
        'src_id': {}
    }
    tuples = set(zip(data['target_chembl_id'], data['assay_chembl_id']))
    total_data = len(tuples)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type(ConnectionError)
    )
    def fetch_assay_data(target_id, assay_id):
        return assay_client.filter(target_chembl_id=target_id, assay_chembl_id=assay_id)

    for target_id, assay_id in tuples:
        try:
            assays = fetch_assay_data(target_id, assay_id)
            if assays:
                assay = assays[0]
                if 'relationship_type' in assay:
                    assay_data['relationship_type'][(target_id, assay_id)] = assay['relationship_type']
                if 'confidence_score' in assay:
                    assay_data['confidence_score'][(target_id, assay_id)] = assay['confidence_score']
                if 'src_id' in assay:
                    assay_data['src_id'][(target_id, assay_id)] = assay['src_id']
                total_data -= 1
                print(f"Assay evidence was found for target:{(target_id, assay_id)}. Remaining target ids:{total_data}")
        except ConnectionError as e:
            print(f"Failed to fetch data for {target_id}, {assay_id} after retries: {e}")
    data['relationship_type'] = data.set_index(['target_chembl_id', 'assay_chembl_id']).index.map(
        assay_data['relationship_type']).values
    data['confidence_score'] = data.set_index(['target_chembl_id', 'assay_chembl_id']).index.map(
        assay_data['confidence_score']).values
    data['src_id'] = data.set_index(['target_chembl_id', 'assay_chembl_id']).index.map(assay_data['src_id']).values
    return data


chembl_binding_activities.dropna(subset=["standard_value", "activity_comment"], how='all', inplace=True)
chembl_binding_activities = map_activity_to_assay(chembl_binding_activities)

chembl_binding_activities = chembl_binding_activities[
    (chembl_binding_activities['relationship_type'] == 'D') &
    (chembl_binding_activities['confidence_score'] == 9)
    ].reset_index(drop=True)

chembl_binding_activities.to_pickle(
    join(CURRENT_DIR, "..", "data", "processed_data", "chembl", "6-3-chembl_binding_activities.pkl"))
print(data_report(chembl_binding_activities))
