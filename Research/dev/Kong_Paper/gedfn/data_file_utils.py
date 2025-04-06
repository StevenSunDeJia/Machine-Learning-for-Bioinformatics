import json
import xml.etree.ElementTree as ET

def load_metadata(file_path = "Kong_Paper/data/patient_genes/metadata.json"):

    # Load metadata file
    with open(file_path, "r") as file:
        metadata = json.load(file)
    return metadata

def seq_name_to_clinical_name(metadata):
    # Extract file IDs and patient barcodes
    data = {}
    for entry in metadata:
        full_id = entry['associated_entities'][0]['entity_submitter_id']
        # Split by '-' and join the first three parts to get the patient ID
        patient_id = '-'.join(full_id.split('-')[:3])
        clinical_file_name = f"nationwidechildrens.org_clinical.{patient_id}.xml"
        data.update({entry['file_name']: clinical_file_name})

    return data

def extract_er_status(file_name):

    try:
        tree = ET.parse(file_name)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file '{file_name}' was not found.")
    except ET.ParseError:
        raise ET.ParseError(f"Error: The file '{file_name}' could not be parsed as XML.")

    root = tree.getroot()

    namespace = {'brca_shared': 'http://tcga.nci/bcr/xml/clinical/brca/shared/2.7'}

    er_status = root.find('.//brca_shared:breast_carcinoma_estrogen_receptor_status', namespace)

    if er_status is not None:
        return er_status.text
    else:
        print("Estrogen Receptor Status not found.")