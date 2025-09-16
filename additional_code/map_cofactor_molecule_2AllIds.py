import requests
import time
import os
import sys
from os.path import join
sys.path.append("./../utilities")
from utilities.helper_functions import *

CURRENT_DIR = os.getcwd()
print(CURRENT_DIR)


def get_compound_info(identifier):
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    chebi_id = None
    pubchem_id = None
    kegg_id = None

    try:
        # If input is CHEBI ID → Fetch PubChem & KEGG only
        if identifier.startswith("CHEBI:"):
            chebi_id = identifier  # We already have it
            url = f"{base_url}/compound/sourceid/CHEBI/{identifier.split(':')[1]}/synonyms/JSON"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                synonyms = data.get("InformationList", {}).get("Information", [{}])[0].get("Synonym", [])
                for synonym in synonyms:
                    if synonym.startswith("CID") and not pubchem_id:
                        pubchem_id = synonym.split("CID")[1]
                    elif synonym.startswith("KEGG:") and not kegg_id:
                        kegg_id = synonym.split("KEGG:")[1]

        # If input is KEGG ID → Fetch CHEBI & PubChem only
        elif identifier.startswith("C") and identifier[1:].isdigit():
            kegg_id = identifier  # We already have it
            url = f"{base_url}/compound/sourceid/KEGG/{identifier}/synonyms/JSON"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                synonyms = data.get("InformationList", {}).get("Information", [{}])[0].get("Synonym", [])
                for synonym in synonyms:
                    if synonym.startswith("CHEBI:") and not chebi_id:
                        chebi_id = synonym
                    elif synonym.startswith("CID") and not pubchem_id:
                        pubchem_id = synonym.split("CID")[1]

        # If input is PubChem CID → Fetch CHEBI & KEGG only
        elif identifier.isdigit():
            pubchem_id = identifier  # We already have it
            url = f"{base_url}/compound/cid/{identifier}/synonyms/JSON"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                synonyms = data.get("InformationList", {}).get("Information", [{}])[0].get("Synonym", [])
                for synonym in synonyms:
                    if synonym.startswith("CHEBI:") and not chebi_id:
                        chebi_id = synonym
                    elif synonym.startswith("KEGG:") and not kegg_id:
                        kegg_id = synonym.split("KEGG:")[1]

    except Exception as e:
        print(f"Error fetching data for {identifier}: {e}")

    return (chebi_id, pubchem_id, kegg_id)


def map_all_ids(id_list):
    id_tuples = []
    for id_ in id_list:
        print(f"Processing {id_}...")
        chebi, pubchem, kegg = get_compound_info(id_)
        id_tuples.append((chebi, pubchem, kegg))
        time.sleep(0.2)  # Avoid rate limiting
    return id_tuples

# Example usage:
cofactor_ids = [
    "CHEBI:29101", "CHEBI:49552", "CHEBI:29035", "CHEBI:29103", "CHEBI:29105", "CHEBI:48775",
    "CHEBI:49807", "CHEBI:26710", "CHEBI:29317", "CHEBI:17996", "CHEBI:32588", "CHEBI:49847",
    "CHEBI:16793", "CHEBI:48828", "CHEBI:49786", "CHEBI:15076", "CHEBI:16382",
    "CHEBI:16042", "CHEBI:37136", "CHEBI:15858", "CHEBI:49544", "CHEBI:29036", "CHEBI:29250",
    "CHEBI:18420", "C00562", "C00211", "CHEBI:16526", "CHEBI:15379", "C00391", "CHEBI:15377",
    "CHEBI:17051", "CHEBI:28741", "CHEBI:6636", "C00046", "CHEBI:29108", "CHEBI:29034",
    "CHEBI:15378", "CHEBI:16541", "CHEBI:16991", "CHEBI:48607", "CHEBI:28938", "CHEBI:16412",
    "CHEBI:28868", "CHEBI:4705", "CHEBI:29033", "CHEBI:17499", "CHEBI:32030",
    "CHEBI:13193", "C01356", "CHEBI:35104", "CHEBI:49553", "CHEBI:31823", "CHEBI:30114",
    "CHEBI:49976", "CHEBI:78067", "CHEBI:35696", "CHEBI:49553", "CHEBI:33737", "CHEBI:33738",
    "CHEBI:15138", "CHEBI:29041", "CHEBI:17606", "CHEBI:49464","CHEBI:27638","CHEBI:3312","CHEMBL5437355","5238"
    "CHEBI:31206","CHEBI:26833","CHEBI:30517","CHEBI:34887","CHEBI:57287","CHEBI:29320","CHEBI:39099","104727"
    "813","104730","28179","3028194","30165","3028194","312","105153","259","104815","923","29936","26623","105130",
    "27668","934","104729","888","104810","32051","104798","31193","271","73212","87642","5886","6083","6031","5893",
    "27284","23964","27854","24085","24479","24014","24385","2124","5235","4873","433294","24012","4311764","25517",
    "29109","5460636","5232483","27099","5460626","107674","253877","104755","962","28486","104883","24293","24480",
    "5234","807","260","5462222","5362487","23990","23931","23994","23925","23978","5462224","23930","5360545","23973",
    "935","5357696","5359268","5352426","23938","24822","5355457","23991","105159","62762","62652","107649","105159",
    "24871","24459","5727","24458","171038574","62683","5460341","23954","5359327","813","28179","30165","1038",
    "CHEBI:33629","312","259","104727","24561","181095","5238", "CHEBI:16813", "CHEBI:58280", "CHEBI:61402",
    "CHEBI:61314", "C00016", "CHEBI:58069", "CHEBI:18072", "CHEBI:58307", "CHEBI:58938", "C00070", "CHEBI:18135",
    "CHEBI:58223", "CHEBI:58189", "CHEBI:57834", "CHEBI:57692", "C00040", "CHEBI:456216", "C00007",
    "CHEBI:58053", "C00080", "CHEBI:37563", "C00001", "C00024", "CHEBI:57288", "CHEBI:16304", "CHEBI:58103",
    "CHEBI:60052", "CHEBI:58207", "C01137", "CHEBI:57455", "CHEBI:597326"
]


energy_ids = [
    "CHEBI:16761", "CHEBI:16027", "CHEBI:18276", "CHEBI:15422", "CHEBI:30616", "CHEBI:16474", "CHEBI:16908",
    "CHEBI:15846", "CHEBI:18009", "CHEBI:58349", "CHEBI:17659", "CHEBI:57540",
    "CHEBI:17552", "CHEBI:15996", "CHEBI:37565", "CHEBI:57783", "CHEBI:13390",
    "C00008", "C00002", "C00004", "C00005", "C00003", "C00006", "C00044", "C00035",
    "5957", "6022", "439153", "5892", "5885", "5884", "135398633", "135398619", "CHEBI:44215",
    "CHEBI:44409", "CHEBI:78442", "962", "977", "CHEBI:33019", "CHEBI:57783", "CHEBI:57945", "6030",
    "CHEBI:456215", "CHEBI:46398", "783"
]

print("Mapping cofactor IDs...")
cofactor_tuples = map_all_ids(cofactor_ids)

print("\nMapping energy IDs...")
energy_tuples = map_all_ids(energy_ids)

# Combine, flatten, and remove None values
all_tuples = cofactor_tuples + energy_tuples
flattened_list = [item for tup in all_tuples for item in tup if item is not None]

# Save to a text file (one ID per line)
with open(join(CURRENT_DIR, "..", "data", "processed_data","cofactors_list.txt"), "w") as f:
    for id_ in flattened_list:
        f.write(f"{id_}\n")