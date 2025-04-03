
config = {
    "azuregpt35-16k" : {
        "model_id":"gpt-35-turbo-16k", 
        "prompt_token_cost": 0.001,
        "response_token_cost": 0.002,
        "temperature": 0.7,
        "max_tokens": 2000,
        "stop": None,
        "api_version": "",
        "api_base": "",
        "api_key": "",
        "embedding_id":"text-embedding-ada-002"
    },
    "memgraph" : {
        "host": "",
        "port": 7687
    },
    "weaviate" : {
        "api_key": "",
        "url": "",
        "db": "",
        "limit": 200
    }
}

from escargot import Escargot
escargot = Escargot(config, node_types = "BiologicalProcess, BodyPart, CellularComponent, Datatype, Disease, Drug, DrugClass, Gene, MolecularFunction, Pathway, Symptom", relationship_types = """CHEMICALBINDSGENE
CHEMICALDECREASESEXPRESSION
CHEMICALINCREASESEXPRESSION
DRUGINCLASS
DRUGCAUSESEFFECT
DRUGTREATSDISEASE
GENEPARTICIPATESINBIOLOGICALPROCESS
GENEINPATHWAY
GENEINTERACTSWITHGENE
GENEHASMOLECULARFUNCTION
GENEASSOCIATEDWITHCELLULARCOMPONENT
GENEASSOCIATESWITHDISEASE
SYMPTOMMANIFESTATIONOFDISEASE
BODYPARTUNDEREXPRESSESGENE
BODYPARTOVEREXPRESSESGENE
DISEASELOCALIZESTOANATOMY
DISEASEASSOCIATESWITHDISEASET""", 
model_name="azuregpt35-16k")
escargot.graph_client.schema = """Node properties are the following:
Node name: 'BiologicalProcess', Node properties: ['commonName']
Node name: 'BodyPart', Node properties: ['commonName']
Node name: 'CellularComponent', Node properties: ['commonName']
Node name: 'Disease', Node properties: ['commonName']
Node name: 'Drug', Node properties: ['commonName']
Node name: 'DrugClass', Node properties: ['commonName']
Node name: 'Gene', Node properties: ['commonName', 'geneSymbol', 'typeOfGene']
Node name: 'MolecularFunction', Node properties: ['commonName']
Node name: 'Pathway', Node properties: ['commonName']
Node name: 'Symptom', Node properties: ['commonName']
Relationship properties are the following:
The relationships are the following:
(:Drug)-[:CHEMICALBINDSGENE]-(:Gene)
(:Drug)-[:CHEMICALDECREASESEXPRESSION]-(:Gene)
(:Drug)-[:CHEMICALINCREASESEXPRESSION]-(:Gene)
(:Drug)-[:DRUGINCLASS]-(:DrugClass)
(:Drug)-[:DRUGCAUSESEFFECT]-(:Disease)
(:Drug)-[:DRUGTREATSDISEASE]-(:Disease)
(:Gene)-[:GENEPARTICIPATESINBIOLOGICALPROCESS]-(:BiologicalProcess)
(:Gene)-[:GENEINPATHWAY]-(:Pathway)
(:Gene)-[:GENEINTERACTSWITHGENE]-(:Gene)
(:Gene)-[:GENEHASMOLECULARFUNCTION]-(:MolecularFunction)
(:Gene)-[:GENEASSOCIATEDWITHCELLULARCOMPONENT]-(:CellularComponent)
(:Gene)-[:GENEASSOCIATESWITHDISEASE]-(:Disease)
(:Symptom)-[:SYMPTOMMANIFESTATIONOFDISEASE]-(:Disease)
(:BodyPart)-[:BODYPARTUNDEREXPRESSESGENE]-(:Gene)
(:BodyPart)-[:BODYPARTOVEREXPRESSESGENE]-(:Gene)
(:Disease)-[:DISEASELOCALIZESTOANATOMY]-(:BodyPart)
(:Disease)-[:DISEASEASSOCIATESWITHDISEASET]-(:Disease)"""


import json
import dill
json_files =  ['MCQ_1hop.json', 'MCQ_2hop.json', 'OpenEnded_1hop.json', 'OpenEnded_2hop.json', 'True_or_False_1hop.json', 'True_or_False_2hop.json']
responses = {}
for json_file in json_files:
    print(json_file)
    with open("../dataset/"+json_file) as f:
        data = json.load(f)
    responses[json_file] = {}
    for question in data:
        response = ''
        tries = 0
        while response == '' and tries < 3:
            escargot.graph_client.cache = {}
            try:
                response = escargot.ask(question['question'], answer_type= "array",debug_level = 0)
            except Exception as e:
                response = ''
            tries += 1
        
        print('question:', question['question'], 'answer:', question['answer'], 'response:', response)
        print("------------------------------------------------------------------------------------------------------------------------------\n")
        responses[json_file][question['question']] = str(response)
dill.dump(responses, open('Escargot_esponses.pkl', 'wb'))