
import escargot.language_models as language_models
import logging
import io

#vectorized knowledge
from escargot.vector_db.weaviate import WeaviateClient

#memgraph Cypher
import escargot.cypher.memgraph as memgraph

from escargot.parser import ESCARGOTParser
from escargot.coder import Coder

from escargot import Escargot
from escargot import operations
import escargot.controller as controller
import escargot.memory as memory

from abc import ABC, abstractmethod
from typing import Dict, List
import re
import logging
from utils import strip_answer_helper, strip_answer_helper_all, parse_xml, parse_xml_code

class CypherESCARGOTParser(ESCARGOTParser):
     def parse_generate_answer(self, state: Dict, texts: List[str]) -> List[Dict]:
        """
        Parse the response from the language model for a generate prompt.

        In GOT, the generate prompt is used for planning, plan assessment, xml conversion, knowledge extraction and array function 

        :param state: The thought state used to generate the prompt.
        :type state: Dict
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The new thought states after parsing the respones from the language model.
        :rtype: List[Dict]
        """

        if type(texts) == str:
            texts = [texts]
        for text in texts:
            self.logger.debug(f"Got response: {text}")
            if state["method"] == "got":
                try:
                    if state["phase"] == "querying":
                        new_state = state.copy()
                        new_state["input"] = text
                        new_state["previous_phase"] = "querying"
                        new_state["phase"] = "steps"
                        new_state["num_branches_response"] = 1
                        # new_state["generate_successors"] = 1
                        new_state["full_code"] = text
                        new_state['instructions'] = [{
                            "StepID": '1',
                            "Instruction": state["question"],
                            "Code": ["result = knowledge_extract(\"\"\""+text.strip()+"\"\"\")"]
                        }]
                        new_state['edges'] = []
                    # elif state["phase"] == "xml_conversion":
                    #     new_state = state.copy()
                    #     new_state["input"] = text
                    #     new_state["previous_phase"] = "xml_conversion"
                    #     new_state["phase"] = "steps"
                    #     instructions, edges = parse_xml_code(text, self.logger)
                    #     new_state["instructions"] = instructions
                    #     new_state["edges"] = edges
                    #     self.logger.info("Got instructions: \n%s",instructions)
                    #     self.logger.info("Got edges: \n%s",edges)
                    # elif state["phase"] == "xml_cleanup":
                    #     new_state = state.copy()
                    #     new_state["input"] = text
                    #     new_state["previous_phase"] = "xml_cleanup"
                    #     new_state["phase"] = "steps"
                    #     instructions, edges = parse_xml(text, self.logger)
                    #     new_state["instructions"] = instructions
                    #     new_state["edges"] = edges
                    #     self.logger.info(f"Got instructions: {instructions}")
                    #     self.logger.info(f"Got edges: {edges}")
                    elif state["phase"] == "steps":
                        new_state["previous_phase"] = "steps"
                        new_state = state.copy()
                        new_state["input"] = text
                    elif state["phase"] == "output":
                        new_state = state.copy()
                        new_state["input"] = text
                except Exception as e:
                    self.logger.error(
                        f"Could not parse step answer: {text}. Encountered exception: {e}"
                    )
            
        return new_state

class CypherESCARGOTPrompter:
    """
    ALZKBPrompter provides the generation of prompts specific to the
    ALZKB example for the language models.

    Inherits from the Prompter class and implements its abstract methods.
    """
    
    memgraph_prompt_1= """You are an expert Memgraph Cypher translator who understands the knowledge graph request and will convert it to Cypher strictly based on the Neo4j Schema provided and following the instructions below:
1. Generate Cypher query compatible ONLY for Memgraph 2.17.0
2. Do not use EXISTS, SIZE, CONTAINS ANY keywords in the cypher. Use alias when using the WITH keyword
3. Please do not use same variable names for different nodes and relationships in the query.
4. Use only Nodes and relationships mentioned in the schema
5. Always enclose the Cypher output inside 3 backticks
6. Always do a case-insensitive and fuzzy search for any properties related search. Eg: to search for a Company name use toLower(c.name) contains 'neo4j'
7. Always use aliases to refer the node in the query
8. 'Answer' is NOT a Cypher keyword. Answer should never be used in a query.
9. Please generate only one Cypher query per question. 
10. Cypher is NOT SQL. So, do not mix and match the syntaxes.
11. Every Cypher query always starts with a MATCH keyword.
12. Always use IN keyword instead of CONTAINS ANY
13. If there is a word surrounded by !, it means it is a specific node and not a node type. For instance, if the word is !Alzheimer's Disease!, it means it is a specific Disease node and not a Disease node type.
14. For the return, return only one property.
15. If a node is a Gene, please make sure you use the geneSymbol property, NOT the commonName.
16. You will receive a request for a specific node or relationship as well as a natural language instruction. Use only the specific node or relationship from the request and only if the request is unclear, use the natural language instruction as.
17. Don't worry about directionality of relationships. Assume all relationships are bidirectional and use a single dash (-) to represent relationships.
18. Return only a single property in the return statement. Do not return multiple properties.
19. When extracting information about genes, always return the geneSymbol property, not the commonName property.
20. If possible, try to keep the query to only one relationship, even if the request seems to require more than one relationship. Use the most direct relationship possible by using the instructions as a guide.
21. The natural language instruction should have what the query should return. Overwrite the return portion of the query based on the natural language instruction.

Schema:
{schema}
"""

    memgraph_prompt_2 = """
Examples:
Instruction: Identify the gene METTL5
Answer: MATCH (g:Gene) WHERE toLower(g.geneSymbol) = toLower("METTL5") RETURN g.geneSymbol

Instruction: Identify the drug that treats Alzheimer's Disease
Answer: MATCH (dr:Drug)-[:DRUGTREATSDISEASE]-(d:Disease) WHERE toLower(d.commonName) = toLower("Alzheimer's Disease") RETURN dr.commonName

Instruction: Find the genes that interact with the gene MCM4.
Answer: MATCH (g1:Gene {geneSymbol: "MCM4"})-[:GENEINTERACTSWITHGENE]-(g2:Gene) RETURN g2.geneSymbol

Instruction: Identify Genes Expressed in Brain. Return the genes in a list.
Memgraph request: MATCH (bp:BodyPart)-[r]-(g:Gene) WHERE toLower(bp.commonName) = toLower("Brain") RETURN g.geneSymbol

Instruction: Find the gene(s) that are subject to decreased expression by the drug Yohimbine.
Memgraph request: MATCH (d:Drug {commonName: "Yohimbine"})-[:CHEMICALDECREASESEXPRESSION]-(g:Gene) RETURN g.geneSymbol

Instruction: Determine the gene symbol for "cytochrome c oxidase assembly factor 7"
Memgraph request: MATCH (g:Gene) WHERE toLower(g.commonName) = toLower("cytochrome c oxidase assembly factor 7") RETURN g.geneSymbol

Instruction: Find the drugs that increase the expression of the gene TMAC and return their relationship score
Memgraph request: MATCH (g:Gene { geneSymbol: "TMAC" } )-[r:CHEMICALINCREASESEXPRESSION]-(d:Drug) RETURN g.geneSymbol, r.z_score

Instruction: Find genes that are associated with Alzheimer's Disease and return their relationship score
Memgraph request: MATCH (g:Gene)-[r:GENEASSOCIATESWITHDISEASE]-(d:Disease {commonName: "Alzheimer's Disease"}) RETURN g.geneSymbol, r.score

Instruction: Find the gene(s) that interact with the transcription factor AHR
Memgraph request: MATCH (tf:TranscriptionFactor {TF: "AHR"})-[:TRANSCRIPTIONFACTORINTERACTSWITHGENE]-(g:Gene) RETURN g.geneSymbol

Instruction: Find the transcription factors(s) that interact with the gene APP
Memgraph request: MATCH (g:Gene {commonName: "APP"})-[:TRANSCRIPTIONFACTORINTERACTSWITHGENE]-(tf:TranscriptionFactor) RETURN tf.TF

Instruction: Find the number of drugs connected to the gene APOE and categorize by relationship type 
Memgraph request: MATCH (g:Gene {commonName: "APOE"})-[r]-(d:Drug) RETURN type(r) AS relationshipType, count(d) AS drugCount

Instruction: Identify drugs associated with Alzheimer's disease and filter genes linked to both these drugs and Alzheimer's disease. Then, exclude the drugs already associated with Alzheimer's disease, and rank the remaining drugs based on the number of connections they have with the identified genes.
Memgraph request: MATCH (d1:Disease{commonName:"Alzheimer's Disease"})--(g:Gene)
MATCH (d1)-[:DRUGTREATSDISEASE]-(:Drug)--(g)--(d2:Drug)
WHERE not exists ((d1)--(d2))
return distinct d2.commonName as drug_name, count(distinct g) as gene_count order by gene_count desc

Instruction: Identify body parts associated with Alzheimer's disease and filter genes connected to these body parts, Alzheimer's disease and TranscriptionFactor. Then, find drugs linked to these genes and also in DrugClass, rank drug based on the number of unique drug gene connections.
Memgraph request: MATCH (d1:Disease)--(b:BodyPart)
WHERE d1.commonName = "Alzheimer's Disease"
WITH DISTINCT b
MATCH (d1:Disease)--(g:Gene)--(b:BodyPart)
MATCH (d:Drug)--(g)
MATCH (g)--(:TranscriptionFactor)
WHERE d1.commonName = "Alzheimer's Disease"
WITH d, count(DISTINCT g) AS gene_count
MATCH (d)-[:DRUGINCLASS]->(:DrugClass)
RETURN DISTINCT d.commonName AS drug_name, gene_count
ORDER BY gene_count DESC

Instruction: Identify body parts associated with Alzheimer's disease via genes and filter genes connected to both these body parts and Alzheimer's disease. Then, find and rank the drugs linked to these genes based on the number of connections.
Memgraph request: MATCH (d:Disease {commonName: "Alzheimer's Disease"})-[:DISEASELOCALIZESTOANATOMY]-(bp:BodyPart) WITH DISTINCT bp MATCH (bp)-[:BODYPARTUNDEREXPRESSESGENE|:BODYPARTOVEREXPRESSESGENE]-(g:Gene) MATCH (g)-[:GENEASSOCIATESWITHDISEASE]-(d) WITH DISTINCT g MATCH (d2:Drug)-[:CHEMICALBINDSGENE|:CHEMICALDECREASESEXPRESSION|:CHEMICALINCREASESEXPRESSION]-(g) RETURN d2.commonName AS drug_name, count(DISTINCT g) AS gene_count ORDER BY gene_count DESC

"""
    memgraph_prompt_3 = """Instruction: {instruction}
Return the Cypher query only."""

    memgraph_adjustment_prompt = """You will be given a potential request extracting information from a knowledge graph, and you must convert the request into a specific format with the following rules:
1. There are specific node names, generic node types, and relationships that must be used in the query.
2. The format of the query should be in the form of Node Name-Relationship-Node Name, where the node names can be specific nodes or generic node types.
3. If possible, use specific node names in the query, not generic node types."""

    clean_up_vector_db_prompt = """Use the following information only:
{knowledge}

Answer the question: {instruction}
Return only the answer in an array of values. Example: "[A,B,...]"""

    code_adjustment_prompt = """Use the following local variables and context to adjust the python code if you see if it gives more accurate results based on intent:
{context}

Here is the intent and the code. Do not reassign local variables and assume the local variables are in context. Adjust so that it gives accurate results:
Instruction: {instruction}
Code to adjust: {code}

Return the adjusted code only."""

    debug_code_prompt = """You will be given a Python code snippet that failed to execute. You must debug the code snippet and provide the corrected code snippet.
The instruction is as follows:
{instruction}

The code snippet is as follows:
{code}

The error message is as follows:
{error}

You must correct the code snippet and provide the corrected code snippet. Return only the code."""

    array_output_prompt = """Given the following question and instructions, you will be provided with a set of steps, associated Python code, and output for the code. You must provide the final variable that answers the question.
Question: {question}

Here are the steps, code, and output:
{steps}

Return the final variable that answers the question."""

    determine_variable_name_prompt = """Given the following Python code, you will be asked to determine the variable name for the knowledge request. A knowledge request should look like: variable_a = knowledge_request("GENE OVEREXPRESSED IN BODYPART-Brain")

Code: {code}
    
Return only the variable name."""

    determine_datastructure_prompt = """Given the following variable and Python code, you will be asked to determine the data structure of the variable.
Variable: {variable}

Code: {code}

Determine the data structure of the variable. If it is a list, return 'list'. If it is a dictionary, return 'dictionary' along with details about the key names. """

    convert_datastructure_prompt = """Given the following variable name, an incomplete preview of its contents (which can be either a dictionary or list), and the original code where the variable is already set but may fail to compile, please return code that modifies the data structure of the variable if needed. The code should convert the variable into a format that will allow the original code to compile. The conversion should happen after the variable is instantiated.
Variable name:
{variable}
                       
Preview of its contents:
{knowledge_array}
                       
Original code:
{code}

Take note of the data structure of the variable and also the data structure of it in the original code after the variable is instantiated. Think step by step and write your thoughts about the data structures you see. Once you think in steps, return only the Python code snippet that converts the data structure in between ``` tags, and if conversion is not needed, do not add any code in your response. Do not return the original code."""

    output_prompt = """Question:
{question}

Train of thoughts:
{steps}

Describe the thought process above, and then answer the question. Do not include any additional information or reasoning beyond what is provided.

Question:
{question}"""
    def __init__(self,vector_db = None, lm = None, graph_client = None, node_types = "", relationship_types = "", relationship_scores = "", logger: logging.Logger = None):
        self.vector_db = vector_db
        self.lm = lm
        self.graph_client = graph_client
        self.node_types = node_types
        self.relationship_types = relationship_types
        self.relationship_scores = relationship_scores
        self.logger = logger
        pass
     
    def generate_prompt(
        self,        
        question: str,
        method: str,
        input: str,
        **kwargs,
    ) -> str:
        """
        Generate a generate prompt for the language model.

        :param num_branches: The number of responses the prompt should ask the LM to generate.
        :type num_branches: int
        :param question: The question to be answered.
        :type question: str
        :param question_type: The type of the question.
        :type question_type: str
        :param method: The method used to generate the prompt.
        :type method: str
        :param input: The intermediate solution.
        :type input: str
        :param kwargs: Additional keyword arguments.
        :return: The generate prompt.
        :rtype: str
        :raise AssertionError: If method is not implemented yet.
        """
        assert question is not None, "Question should not be None."
        if method == "got":
            if (input is None or input == "") and kwargs["phase"] == "querying":
                return self.memgraph_prompt_1.format(schema=self.graph_client.schema) + str(self.memgraph_prompt_2) + str(self.memgraph_prompt_3.format(instruction= question))
            elif kwargs["phase"] == "plan_assessment":
                return self.plan_assessment_prompt.format(question=question, approach_1=input[0], approach_2=input[1], approach_3=input[2], node_types=self.node_types, relationship_types=self.relationship_types, relationship_scores=self.relationship_scores)
            elif kwargs["phase"] == "python_conversion":
                return self.python_conversion_prompt.format(instructions=input)
            elif kwargs["phase"] == "code_assessment":
                return self.python_assessment_prompt.format(question=question, approach=kwargs["full_plan"], approach_1=input[0], approach_2=input[1], approach_3=input[2])
            elif kwargs["phase"] == "xml_conversion":
                return self.xml_conversion_prompt.format(instructions=input)
            elif kwargs["phase"] == "xml_cleanup":
                return self.xml_cleanup_prompt.format(xml=input, node_types=self.node_types, relationship_types=self.relationship_types)
            elif kwargs["phase"] == "output":
                steps = ""
                for step in kwargs["instructions"]:
                    steps += "Step " + step["StepID"] + ": " + step["Instruction"] + "\n"
                    steps += "Code: " + step["Code"][0] + "\n"
                    output = input[step["StepID"]]
                    if len(str(output)) > 256:
                        output = str(output)[:256] + "..."
                    steps += "Output: " + str(output) + "\n\n"
                if kwargs["answer_type"] == "array":
                    return self.array_output_prompt.format(question=question, steps=steps)
                else:
                    return self.output_prompt.format(question=question, steps=steps)
        else:
            raise AssertionError(f"Method {method} is not implemented yet.")
    def get_knowledge(self,knowledge_request,instruction, code = "", full_code = ""):
        statement_to_embed = knowledge_request
        if statement_to_embed == "" or statement_to_embed is None:
            return []
        if self.graph_client is not None:
            knowledge_array = list(self.graph_client.client.execute_and_fetch(statement_to_embed))
            # knowledge_array = self.graph_client.execute(self.lm, self.memgraph_prompt_1.format(schema=self.graph_client.schema) + str(self.memgraph_prompt_2) + str(self.memgraph_prompt_3.format(instruction=str(instruction),cypher=str(statement_to_embed))),str(statement_to_embed))
            #Identify genes associated with Alzheimer’s disease and filter bodypart linked to both these genes 
# and Alzheimer’s disease. Then, exclude the genes already associated with Alzheimer’s disease, and rank the
# remaining genes based on the number of connections they have with the identified bodypart.
            self.logger.error(f"```escargot|SHOW```Executed Cypher: {statement_to_embed}")
            
            return knowledge_array
        return "knowledge"
    def generate_debug_code_prompt(self, code, instruction, e):
        prompt = self.debug_code_prompt.format(instruction=instruction, code=code, error=e)
        return prompt
    
    def adjust_code(self, code, instruction, context):
        prompt = self.code_adjustment_prompt.format(instruction=instruction, code=code, context=context)
        code = self.lm.get_response_texts(
            self.lm.query(prompt, num_responses=1)
        )[0]
        self.logger.debug(f"Adjusting code prompt: {prompt}")
        self.logger.debug(f"Adjusted code: {code}")
        return code
    
    def restructure_data_structure(self, data, context):
        prompt = self.determine_datastructure_prompt.format(data=data, context=context)
        data_format = self.lm.get_response_texts(
           self.lm.query(prompt, num_responses=1)
        )[0]

        prompt = self.restructure_datastructure_code_prompt.format(data=data, context=context)
        data = self.lm.get_response_texts(
            self.lm.query(prompt, num_responses=1)
        )[0]

        return data

class CypherEscargot(Escargot):
    def __init__(self, config: str, node_types:str = "", relationship_types:str = "", relationship_scores:str ="", model_name: str = "azuregpt35-16k"):
        logger = logging.getLogger(__name__)
        self.logger = logger
        self.log = ""
        self.lm = language_models.AzureGPT(config, model_name=model_name, logger=logger)
        self.vdb = WeaviateClient(config, self.logger)
        self.memory = None
        self.node_types = ""
        self.relationship_types = ""
        self.question = ""
        self.controller = None
        self.operations_graph = None
        if self.vdb.client is None:
            self.vdb = None
        self.graph_client = memgraph.MemgraphClient(config, logger)
        if self.graph_client.client is None:
            self.graph_client = None
        else:
            if node_types == "" or relationship_types == "":
                self.node_types, self.relationship_types = self.graph_client.get_schema()
            else:
                self.node_types = node_types
                self.relationship_types = relationship_types
        self.relationship_scores = relationship_scores

    def ask(self, question, answer_type = 'natural', num_strategies=1, debug_level = 0, memory_name = "escargot_memory", max_run_tries = 3):
        """
        Ask a question and get an answer.

        :param question: The question to ask.
        :type question: str
        :param answer_type: The type of answer to expect. Defaults to 'natural'. Options are 'natural', 'array'.
        :type answer_type: str
        :param num_strategies: The number of strategies to generate. Defaults to 3.
        :type num_strategies: int
        :return: The answer to the question.
        :rtype: str
        """

        #setup logger
        log_stream, c_handler, f_handler = self.setup_logger(debug_level)
        def got() -> operations.GraphOfOperations:
            operations_graph = operations.GraphOfOperations()

            instruction_node = operations.Generate(1, 1)
            operations_graph.append_operation(instruction_node)
            
            return operations_graph
        
        # Create the Controller
        got = got()
        try:
            self.memory = memory.Memory(self.lm, collection_name = memory_name)
            self.controller = controller.Controller(
                self.lm, 
                got, 
                CypherESCARGOTPrompter(graph_client = self.graph_client,vector_db = self.vdb, lm=self.lm,node_types=self.node_types,relationship_types=self.relationship_types, relationship_scores=self.relationship_scores,logger = self.logger),
                CypherESCARGOTParser(self.logger),
                self.logger,
                Coder(),
                {
                    "question": question,
                    "input": "",
                    "phase": "querying",
                    "method" : "got",
                    "num_branches_response": num_strategies,
                    "answer_type": answer_type,
                }
            )
            self.controller.max_run_tries = max_run_tries
            self.controller.run()
        except Exception as e:
            self.logger.error("Error executing controller: %s", e)

        self.operations_graph = self.controller.graph.operations
        output = ""
        if self.controller.final_thought is not None:
            if answer_type == 'natural':
                output = self.controller.final_thought.state['input']
            elif answer_type == 'array':
                # output = list(list(self.controller.coder.step_output.values())[-1].values())[-1]
                output = list(self.controller.coder.step_output.values())[-1]

        self.logger.warning(f"Output: {output}")

        #remove logger
        self.finalize_logger(log_stream, c_handler, f_handler)

        return output
    