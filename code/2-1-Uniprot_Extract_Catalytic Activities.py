import requests
import pandas as pd
import os
import sys
import re
from os.path import join
import gc
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append("./../utilities")
from utilities.helper_functions import *

CURRENT_DIR = os.getcwd()
print(CURRENT_DIR)

enzyme_data = []
search_url = 'https://rest.uniprot.org/uniprotkb/search'
ec_number_patterns = [f'{i}.*' for i in range(1, 8)]
processed_count = 0
session = requests.Session()
retry_strategy = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)


def annotation_score_filter(enzyme_id):
    """Fetches details for a specific entry to check annotationScore and extract information."""
    details_url = f"https://rest.uniprot.org/uniprotkb/{enzyme_id}"
    try:
        response = session.get(details_url, headers={"Accept": "application/json"})
        response.raise_for_status()
        detail_data = response.json()

        # Check for annotationScore of 5.0
        if detail_data.get('annotationScore') == 5.0:
            return detail_data
    except requests.exceptions.RequestException as e:
        print(f"Failed to retrieve details for {enzyme_id}: {e}")
    return None


for ec_pattern in ec_number_patterns:
    # Limit results to reviewed, active entries, and existence level 1
    next_page_url = f"{search_url}?query=ec:{ec_pattern}+AND+reviewed:true+AND+existence:1&fields=id,protein_name,sequence&format=json&size=500"
    while next_page_url:
        try:
            response = session.get(next_page_url)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"Failed to retrieve data for EC pattern {ec_pattern}: {e}")
            break

        if 'results' in data:
            entry_ids = [entry['primaryAccession'] for entry in data['results']]

            # Parallelize detail requests
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(annotation_score_filter, enzyme_id): enzyme_id for enzyme_id in entry_ids}
                for future in as_completed(futures):
                    detail_data = future.result()
                    if detail_data:
                        # Process the data only if annotationScore == 5.0
                        enzyme_id = detail_data['primaryAccession']
                        protein_description = detail_data.get('proteinDescription', {})
                        recommended_name = protein_description.get('recommendedName', {})
                        enzyme_name = recommended_name.get('fullName', {}).get('value', 'No name available')
                        sequence = detail_data.get('sequence', {}).get('value', '')
                        ec_number_str = ",".join(set([ec['value'] for ec in recommended_name.get('ecNumbers', [])]))
                        # Extract catalytic activity information
                        if 'comments' in detail_data:
                            for comment in detail_data['comments']:
                                if comment.get('commentType') == 'CATALYTIC ACTIVITY' and 'reaction' in comment:
                                    reaction = comment['reaction']
                                    if 'ecNumber' in reaction:
                                        EC_ID = reaction['ecNumber']
                                    else:
                                        EC_ID = ec_number_str
                                    reaction_name = reaction.get('name', '')
                                    right_side_list = []
                                    if '=' in reaction_name:
                                        right_side = reaction_name.split('=')[-1].strip()
                                        right_side_list = right_side.split(' + ')

                                    rhea_ids = [xref['id'] for xref in reaction.get('reactionCrossReferences', []) if
                                                xref['database'] == 'Rhea']
                                    evidence_codes = list(
                                        set([evidence['evidenceCode'] for evidence in reaction.get('evidences', [])]))
                                    for r in right_side_list:
                                        substrate_cleaned = re.sub(r'^\d+\s+', '', r)
                                        substrate_cleaned = re.sub(r'^(?:a|an)\s+', '',
                                                                   substrate_cleaned.replace('(in)', '').replace(
                                                                       '(out)', '').strip())
                                        if substrate_cleaned.startswith('oxidized') or substrate_cleaned.startswith(
                                                'reduced'):
                                            match = re.search(r'\[(.*?)\]', substrate_cleaned)
                                            if match:
                                                substrate_cleaned = match.group(1)
                                        enzyme_data.append({
                                            'ID': enzyme_id,
                                            'Name': enzyme_name,
                                            'Protein_Sequence': sequence,
                                            'EC Number': EC_ID if EC_ID else "No EC number",
                                            'Substrate': substrate_cleaned,
                                            'Rhea IDs': rhea_ids if rhea_ids else 'No Rhea IDs',
                                            'Evidence Codes': evidence_codes if evidence_codes else 'No evidence codes'
                                        })

        processed_count += len(data.get('results', []))
        print(f"Processed {processed_count} entities for EC {ec_pattern}")
        link_header = response.headers.get('Link', '')
        next_page_url = None
        if 'rel="next"' in link_header:
            match = re.search(r'<([^>]+)>;\s*rel="next"', link_header)

            if match:
                next_page_url = match.group(1)

    processed_count = 0

# Post-process and save data
df = pd.DataFrame(enzyme_data)
df = df[df['Evidence Codes'] != 'No evidence codes']
df = df[df['EC Number'] != 'No EC number']
df = df[df['Rhea IDs'] != 'No Rhea IDs']
substrates_to_filter = ['H2O', 'H2O2', 'H+', '2 H2O', 'H(+)', 'CO2', '2 H(+)', '2 CO2', 'O2', 'Ca2', 'Mn2', 'Na+',
                        'Zn2+', 'K+', 'Mg2+', 'Mg+', 'Cd2+', 'Cu+', 'phosphate', 'Cu2+']
df = df[df['Substrate'].apply(lambda x: not any(sub in x for sub in substrates_to_filter))]

experimental_df = pd.read_pickle(join(CURRENT_DIR, "..", "data", "raw_data", "GOA_data", "experimental_df_GO_UID.pkl"))
evidence_map = dict(zip(experimental_df['ECO_Evidence_code'], experimental_df['evidence']))


def map_evidence_codes(evidence_codes):
    is_exp = any(evidence_map.get(code, 'Not exp') == 'exp' for code in evidence_codes)
    return 'exp' if is_exp else 'Not exp'


df['Evidence'] = df['Evidence Codes'].apply(map_evidence_codes)
df = df[df['Evidence'] != 'Not exp']
df.reset_index(drop=True, inplace=True)
df.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data", "uniprot", "2-1-uniprot_enz_sub.pkl"))
print(data_report(df))
