<p align="center">
<img src="https://github.com/EpistasisLab/escargot/blob/main/escargot_logo.jpeg" width=400 />
</p>

# ESCARGOT
## Overview

LLMs like GPT-4, despite their advancements, often produce hallucinations and struggle with integrating
external knowledge effectively. While Retrieval-Augmented Generation (RAG) attempts to address this by incorporating
external information, it faces significant challenges such as context length limitations and imprecise vector similarity
search. ESCARGOT aims to overcome these issues by combining LLMs with a dynamic Graph of Thoughts and knowledge
graphs, improving output reliability and reducing hallucinations.
ESCARGOT significantly outperforms industry-standard RAG methods, particularly in open-ended questions
that demand high precision. ESCARGOT also offers greater transparency in its reasoning process, allowing for the vetting
of both code and knowledge requests, in contrast to the black-box nature of LLM-only or RAG-based approaches.

ESCARGOT significantly outperforms industry-standard RAG methods, particularly in open-ended questions that demand high precision. ESCARGOT also offers greater transparency in its reasoning process, allowing for the vetting of both code and knowledge requests, in contrast to the black-box nature of LLM-only or RAG-based approaches.

## Results

<div style="text-align: center;">

| **Dataset**                                      | **GPT 3.5 Turbo/GPT 4o-mini** | **Standard RAG<br>(GPT3.5 Turbo/GPT4o-mini)**  | **KRAGEN (GPT3.5 Turbo)** | **ESCARGOT<br>(GPT3.5/GPT4o-mini)**       |
|--------------------------------------------------|-------------------|----------|-------------|--------------------|
| Openended 1-hop (508 questions)                  | 3.3%   / 4.2%           | 50.2% / 55.5%   | 53.1%    | 81.0% / **88.4%**          |
| Openended 2-hop (450 questions)                  | 3.5%   / 4.9%           | 12.8% / 23.1%   | 19.9%    | **91.8%** / 85.8%          |
| True/False 1-hop (560 questions)                 | 55.9%  /  60.5%          | 73.0% / 85.2%  | 80.2%    | 80.7% / **90.9%**          |
| True/False 2-hop (540 questions)                 | 26.7%  /  59.4%         | 64.4% / 75.6%  | 62.4%    | **77.6%** / 74.1%          |
| Multiple Choice 1-hop (498 questions)            | 42.6% /   58.2%           | 77.7% / 88.8%   | 65.1%    | **94.6%** / 93.4%           |
| Multiple Choice 2-hop (419 questions)            | 49.9% /   61.3%          | 81.9% / 86.4%   | 62.2%    | **94.2%** / 89.5%           |

</div>


## Key Features

1. **Dynamic GoT Generation**: ESCARGOT dynamically generates a Python-executable Graph of Thoughts (GoT) that integrates with knowledge graphs. This dynamic approach ensures improved accuracy and contextual relevance compared to static GoT frameworks.

2. **Strategic Planning and Execution**: ESCARGOT's workflow includes strategy generation, assessment, code generation, XML conversion, and execution. This multi-step process ensures that each strategy is thoroughly evaluated and executed efficiently.

3. **Advanced Knowledge Retrieval**:
   - **Cypher Queries**: ESCARGOT utilizes Cypher queries to extract structured and precise information from knowledge graphs like Memgraph, offering superior accuracy in data retrieval.
   - **Vector Database Requests**: As a backup, ESCARGOT leverages vector databases for similarity searches, enhancing the system’s ability to handle diverse queries even if Cypher queries fail.

4. **Direct Python Execution**: The system supports direct Python execution for tasks requiring high precision. By converting knowledge into executable Python code, ESCARGOT ensures reliable and efficient computation, reducing the risk of errors and hallucinations.

5. **Self-Debugging and Adaptability**: ESCARGOT includes built-in self-debugging capabilities. It can autonomously analyze and revise code if errors occur during execution, ensuring resilience and reducing the need for manual intervention.

6. **Enhanced Accuracy and Reduced Hallucinations**: By integrating structured knowledge retrieval and direct code execution, ESCARGOT minimizes the risk of hallucinations and improves the overall accuracy of reasoning and computation.


---

## Quick Install Through Pip
`pip install escargot`

## Example Usage

Here's how to configure and use the `Escargot` library with your chosen models and databases:

### Configuration

First, set up your configuration with the necessary parameters:

```python
config = {
    "azuregpt35-16k" : {
        "model_id":"gpt-35-turbo-16k", 
        "prompt_token_cost": 0.001,
        "response_token_cost": 0.002,
        "temperature": 0.7,
        "max_tokens": 2000,
        "stop": None,
        "api_version": "API_VERSION",
        "api_base": "API_BASE_HERE",
        "api_key": "API_KEY_HERE",
        "embedding_id":"text-embedding-ada-002"
    },
    "memgraph" : {
        "host": "MEMGRAPH_URL",
        "port": 7687
    },
    "neo4j" : {
        "host": "NEO4J_URL",
        "port": 7687
    },
    "weaviate" : {
        "api_key": "WEAVIATE_API_KEY",
        "url": "WEAVIATE_URL",
        "db": "WEAVIATE_DB",
        "limit": 200
    }
}
```
### Initializing Escargot
Initialize the Escargot instance with your configuration. Escargot will automatically connect to the Memgraph database and retrieve all the node types and relationships.

```python
from escargot import Escargot

escargot = Escargot(
    config, model_name="azuregpt35-16k"
)
```
### Ask a question

To query the `Escargot` library, use the `ask` function. You can specify optional parameters such as `debug_level` and `answer_type` to control the verbosity of logging and the format of the response.

#### Example of extracting information from the knowledge graph

```python
# Basic question
response = escargot.ask("What is the function of the gene APOE?")
```
```
The gene APOE has several functions, including low-density lipoprotein particle receptor binding, protein homodimerization activity, steroid binding, heparin binding, tau protein binding, amide binding, lipoprotein particle receptor binding, and protein-lipid complex assembly.
```
```python
# Advanced usage with additional parameters
response = escargot.ask(
    "What is the function of the gene APOE?",
    debug_level=0,       # Set the level of logging (0 to 3). 0 is default
    answer_type="natural"  # Specify the response format ("natural" or "array") "natural" is default.
)
```
#### Example of extracting an array of genes from the knowledge graph
```python
escargot.ask("List the genes that are associated with the Alzheimer's disease", answer_type="array")
```
```
{'genes_list': ['GSK3B',
  'CASP3',
  'CHRNB2',
  'IGF2',
  'IQCK',
  'MS4A4A',
  'IL1B',
  'ACE',
  'VEGFA',
  'WWOX',
  'DPYSL2',
  'MIR4467',
...
  'INSR',
  'ABCA7',
  'SORL1',
  'HLA-DRB5',
  'ACHE']}
```


---
### Manually Configure Knowledge Graph Schema (bypasses automated database schema extraction)
Retrieving the schema does take time, so it is useful to extract the schema once, manually configure it and insert it into the instantiation of Escargot.
If you need to manually configure the node_types and relationships, you will skip the above automated knowledge graph schema retrieval and use the manually inputted configuration:
```python
from escargot import Escargot

escargot = Escargot(
    config,
    node_types="BiologicalProcess, BodyPart, CellularComponent, Datatype, Disease, Drug, DrugClass, Gene, MolecularFunction, Pathway, Symptom",
    relationship_types="""CHEMICALBINDSGENE
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
    model_name="azuregpt35-16k"
)
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

```
