import requests
import pandas as pd
import os
import sys
import re
from os.path import join
import json
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append("./../utilities")
from utilities.helper_functions import *

CURRENT_DIR = os.getcwd()
print(CURRENT_DIR)

enzyme_data = []
search_url = "https://rest.uniprot.org/uniprotkb/search"
ec_number_patterns = [f"{i}.*" for i in range(1, 8)]
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
        if detail_data.get("annotationScore") >= 4.0:
            return detail_data  # Return details if annotationScore is 5.0
    except requests.exceptions.RequestException as e:
        print(f"Failed to retrieve details for {enzyme_id}: {e}")
    return None


for ec_pattern in ec_number_patterns:
    # Limit results to reviewed, active entries, and existence level 1
    next_page_url = f"{search_url}?query=ec:{ec_pattern}+AND+existence:1&fields=id,protein_name,sequence&format=json&size=500"
    while next_page_url:
        try:
            response = session.get(next_page_url)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"Failed to retrieve data for EC pattern {ec_pattern}: {e}")
            break

        if "results" in data:
            entry_ids = [entry["primaryAccession"] for entry in data["results"]]

            # Parallelize detail requests
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(annotation_score_filter, enzyme_id): enzyme_id for enzyme_id in entry_ids}
                for future in as_completed(futures):
                    detail_data = future.result()
                    if detail_data:
                        # Process the data only if annotationScore == 5.0
                        enzyme_id = detail_data["primaryAccession"]
                        protein_description = detail_data.get("proteinDescription", {})
                        recommended_name = protein_description.get("recommendedName", {})
                        enzyme_name = recommended_name.get('fullName', {}).get("value", "No name available")
                        sequence = detail_data.get("sequence", {}).get("value", "")
                        ec_number_str = ",".join(set([ec["value"] for ec in recommended_name.get("ecNumbers", [])]))
                        if "comments" in detail_data:

                            for comment in detail_data["comments"]:
                                inhibitors = []
                                evidence_codes = None
                                if comment.get("commentType") in [
                                    "ACTIVITY REGULATION"] and "texts" in comment and "evidences" in comment["texts"][
                                    0]:
                                    text = comment["texts"][0]["value"]
                                    text = text.split(". ")
                                    evidences = comment["texts"][0]["evidences"]
                                    for sentence in text:
                                        if "not inhibit" in sentence.lower():
                                            continue
                                        elif "inhibit" in sentence.lower():
                                            inhibitors.append(sentence)
                                        else:
                                            continue
                                    evidences = comment["texts"][0]["evidences"]
                                    evidence_codes = list(set([evidences[0]["evidenceCode"] for evidence in evidences]))
                                    if len(inhibitors) != 0:
                                        enzyme_data.append({
                                            "ID": enzyme_id,
                                            "Name": enzyme_name,
                                            "Protein_Sequence": sequence,
                                            "Evidence Codes": evidence_codes if evidence_codes else "No evidence codes",
                                            "Ihibitors_description": inhibitors,
                                            'EC Number': ec_number_str
                                        })

        processed_count += len(data.get("results", []))
        print(f"Processed {processed_count} entities for EC {ec_pattern}")
        link_header = response.headers.get("Link", "")
        next_page_url = None
        if 'rel="next"' in link_header:
            match = re.search(r'<([^>]+)>;\s*rel="next"', link_header)

            if match:
                next_page_url = match.group(1)

    processed_count = 0

# Post-process and save data
df = pd.DataFrame(enzyme_data)
df = df[df["Evidence Codes"] != "No evidence codes"]
experimental_df = pd.read_pickle(join(CURRENT_DIR, "..", "data", "raw_data", "GOA_data", "experimental_df_GO_UID.pkl"))
evidence_map = dict(zip(experimental_df["ECO_Evidence_code"], experimental_df["evidence"]))


def map_evidence_codes(evidence_codes):
    is_exp = any(evidence_map.get(code, "Not exp") == "exp" for code in evidence_codes)
    return "exp" if is_exp else "Not exp"


df["Evidence"] = df["Evidence Codes"].apply(map_evidence_codes)
df = df[df["Evidence"] != "Not exp"]
print(data_report(df))
df_exploded = df.explode("Ihibitors_description")
df_exploded.reset_index(drop=True, inplace=True)
print(data_report(df_exploded))
df_exploded.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data", "uniprot", "7-1-uniprot_enz-inh.pkl"))
print(data_report(df_exploded))
