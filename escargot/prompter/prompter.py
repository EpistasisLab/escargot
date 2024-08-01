from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List
import re
import logging


class ESCARGOTPrompter:
    """
    ALZKBPrompter provides the generation of prompts specific to the
    ALZKB example for the language models.

    Inherits from the Prompter class and implements its abstract methods.
    """
    planning_prompt = """You are a brilliant strategic thinker with access to a knowledge base. You will receive a question that will require the knowledge base to answer. The knowledge base is built from a knowledge graph, but not the knowledge graph itself. You will break down the question into steps. If knowledge needs to be pulled from the knowledge base, you will provide what specific relationships or node information is necessary. You do not need to try to answer the question but simply plan out the steps.

The knowledge graph contains the following generic node types: {node_types}.
The knowledge graph contains the following relationships: {relationship_types}.

Only show the steps you will take and a small description for each step. If you can determine the knowledge graph relationship that can provide insight in the step, provide the relationship in it and if possible the specific node name, not the node type. If a question require a specific relationship between two specific nodes, provide the specific nodes in the relationship.
Each step should also try to be clear on what the knowledge request output should be. If the output is a list of genes, then the step should be clear on what genes should be returned.

Examples:
Question: List the genes which bind to the drug Cyclothiazide
Step 1: Find the genes that bind to the drug Cyclothiazide. (CHEMICALBINDSGENE)
Step 2: List the genes that were found.

Question: List the genes which are commonly under-expressed in spinal cord and thyroid gland
Step 1: Identify Genes Under-Expressed in Spinal Cord. Return the genes in a list.
Relationship: BODYPARTUNDEREXPRESSESGENE
Node: Spinal Cord
Step 2: Identify Genes Under-Expressed in Thyroid Gland. Return the genes in a list.
Relationship: BODYPARTUNDEREXPRESSESGENE
Node: Thyroid Gland
Step 3: Find Common Genes
Action: Compare the lists of genes from steps 1 and 2 to identify common genes under-expressed in both the spinal cord and thyroid gland.

Question: True or False Question: Posaconazole binds to the gene CYP3A4
Step 1: Find the drugs that bind to the gene CYP3A4. List the drugs.
Step 2: Check if Posaconazole is in the list of drugs that bind to the gene CYP3A4.

Here is your question:
{question}

Let's think step by step, and be very succinct, clear, and efficient in the number of steps by avoiding redundant knowledge extractions."""

    plan_assessment_prompt = """You are a brilliant strategic thinker with access to a biomedical knowledge graph. You will receive a query that will require the knowledge graph to answer and a few approaches that will try to resolve that query. 

Here is your question:
{question}

Here is ApproachNumber 1:
"{approach_1}"

Here is ApproachNumber 2:
"{approach_2}"

Here is ApproachNumber 3:
"{approach_3}"

If knowledge needs to be pulled from the knowledge graph, they will try to provide what specific relationships or node information is necessary.
The score should reflect on how clear and efficient each step is. It is not about how many steps are taken, but the quality of the steps. The highest score approach would be specific on what knowledge to extract, especially if it includes specific nodes. For instance, an approach with step 'Find the body parts or anatomy that over-express METTL5. (BODYPART OVEREXPRESSES GENE)' scores very high since it is specific on the node METTL5 and performs a 1 hop knowledge request.
An approach where there are steps that contain a specific node with a relationship scores higher than an approach where there any steps that contain only arbitrary node types such as when a step that is a generic node type like 'Determine the relationship between genes and body parts that represents over-expression'
Do not allow steps that simply ask for a generic node, such as "List all drugs" or "Find all diseases". These steps should be be penalized in the score.
If the question is a multiple choice question, the approaches that include the specific multiple choice options in at least one of the steps will score higher.

The knowledge graph contains node types: {node_types}.
The knowledge graph contains relationships: {relationship_types}.

Return a XML formatted list with all the approaches in Approach tags. Each approach should be within <Approach> tags and will have an incremental <ApproachID> value within it. The score should be an integer between 1-10 within <Score> tags.
An example is as follows:
<Approaches>
  <Approach>
    <ApproachID>1</ApproachID>
    <Score>...</Score>
  </Approach>
  ...
</Approaches>

Only return the XML.
"""

    python_conversion_prompt = """You will be given a set of instructions and you must convert the instructions into Python code with the following rules:

{instructions}

Provide Python code for each step, and put the instructions for that step in comments before the code for that step. Do not define new functions. Only use default Python packages and numpy. Do not include print statements. If the code within a step requires a request from the knowledge graph, you must use the knowledge_extract(x) function where there is only one variable in x, which is the specific knowledge you are requesting. This function will return a list. For instance, if you need to find the genes over-expressed in the brain, you would use knowledge_extract("GENE OVEREXPRESSED IN BODYPART-Brain")."""

    xml_conversion_prompt = """You will be given instructions and python code for each step. The instructions assume you have connection to a knowledge graph. The must convert it into XML with the following rules:
Format your response in XML format, where the steps will be within <Instructions> tags. Each step will be within <Step> tags and will have an incremental <StepID> value within it. The full description of the step will be put in the <Instruction> tags within the <Step>. Following the <Instruction>, you must put in the code corresponding to the step within <Code> tags.
If the code within a step requires a request from the knowledge graph, you must use the knowledge_extract(x) function where x is the specific keyword you are requesting. For instance, if you need to find the genes over-expressed in the brain, you would use knowledge_extract("GENE OVEREXPRESSED IN BODYPART-Brain"). The code will likely have placeholders for the knowledge request that you must fill in, such as ['gene1', 'gene2', 'gene3'].
Outside of the <Instructions> tag, add an edge list in <EdgeList>, where information from one step to another will be listed. Each edge will be within <Edge> tags, and the edge would be in the format StepID1-StepID2 which describes that StepID1 directs to StepID2.
Do not include any other tags other than the ones mentioned above.

Here is an example XML:
<Instructions>
    <Step>
        <StepID>1</StepID>
        <Instruction>
            Find Body Parts Over-Expressing Gene METTL5
        </Instruction>
        <Code>
            genes_overexpressed_in_nipple = knowledge_extract("BODYPART OVEREXPRESSES GENE-METTL5")
        </Code>
    </Step>
    <Step>
        <StepID>2</StepID>
        <Instruction>
            Find Body Parts Over-Expressing Gene STYXL2
        </Instruction>
        <Code>
            genes_overexpressed_in_nipple = knowledge_extract("BODYPART OVEREXPRESSES GENE-STYXL2")
        </Code>
    </Step>
    <Step>
        <StepID>3</StepID>
        <Instruction>
            List the intersect of body parts
        </Instruction>
        <Code>
            intersect = set(genes_overexpressed_in_nipple) & set(genes_overexpressed_in_brain)
        </Code>
    </Step>
</Instructions>
<EdgeList>
    <Edge>1-3</Edge>
    <Edge>2-3</Edge>
</EdgeList>

Here are the instructions you must convert:
{instructions}"""

    knowledge_request_adjustment_prompt = """You will be given a potential request extracting information from a knowledge graph, and you must convert the request into a specific format with the following rules:
1. There are specific node names, generic node types, and relationships that must be used in the query. 
2. The format of the query should be in the form of Node Name-Relationship-Node Name, where the node names can be specific nodes or generic node types.
3. If possible, use specific node names in the query, not generic node types.
4. If a specific node name is provided, surround the node name with a !. For example, if the node name is "METTL5", the query should be !METTL5!.
5. Do not surround a generic node type with a !. For example, if the node type is "Gene", the query should be Gene.
6. You must only use the the relationships and node types provided in the schema.
7. If the instruction is unclear, you must use the natural language instruction as a guide.
8. If you notice that the request does not contain a specific node name, you must use the natural language instruction as a guide.
9. Body parts should be lower case and not capitalized.
10. If the instruction has a reference to a specific step, such as "Return the genes from Step 3", you must use the step identifier in the query.
11. If there are no changes needed to the request, return the request as is.
The knowledge graph contains the following generic node types: {node_types}.
The knowledge graph contains the following relationships: {relationship_types}.

Examples:
Request: METTL5
Answer: !METTL5!

Request: Alzheimer's Disease-DRUG TREATS DISEASE-Drug
Answer: !Alzheimer's Disease!-DRUG TREATS DISEASE-Drug

Request: GENE-GENE INTERACTS WITH GENE-MCM4
Answer: Gene-GENE INTERACTS WITH GENE-!MCM4!

Request: GENE-BODYPART UNDER EXPRESSES GENE-Brain
Answer: Gene-BODYPART UNDER EXPRESSES GENE-!brain!

Request: Drug-CHEMICALDECREASESEXPRESSION-Gene
Answer: !Yohimbine!-CHEMICALDECREASESEXPRESSION-Gene

Request: Datatype-GENE-NAME-cytochrome c oxidase assembly factor 7
Answer: Gene-!cytochrome c oxidase assembly factor 7!

Here is the instruction you must convert:
Request: {statement_to_embed}

Return only the answer."""

    function_prompt="""{function}"""

    output_prompt = """Use the following knowledge to answer the question:
{knowledge}

With the above knowledge, follow this step:
{instruction}
and answer this question: {question}"""

    memgraph_prompt_1= """You are an expert memgraph Cypher translator who understands the knowledge graph request and will convert it to Cypher strictly based on the Neo4j Schema provided and following the instructions below:
1. Generate Cypher query compatible ONLY for memgraph 2.17.0
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
Request: !METTL5!
Answer: MATCH (g:Gene) WHERE toLower(g.geneSymbol) = toLower("METTL5") RETURN g.geneSymbol

Instruction: Identify the drug that treats Alzheimer's Disease
Request: !Alzheimer's Disease!-DRUG TREATS DISEASE-Drug
Answer: MATCH (dr:Drug)-[:DRUGTREATSDISEASE]-(d:Disease) WHERE toLower(d.commonName) = toLower("Alzheimer's Disease") RETURN dr.commonName

Instruction: Find the genes that interact with the gene MCM4.
Request: GENE-GENE INTERACTS WITH GENE-!MCM4!
Answer: MATCH (g1:Gene {geneSymbol: "MCM4"})-[:GENEINTERACTSWITHGENE]-(g2:Gene) RETURN g2.geneSymbol

Instruction: Identify Genes Under-Expressed in Brain. Return the genes in a list.
Request: GENE-BODYPART UNDER EXPRESSES GENE-!Brain!'
Memgraph request: MATCH (bp:BodyPart)-[:BODYPARTUNDEREXPRESSESGENE]-(g:Gene) WHERE toLower(bp.commonName) = toLower("Brain") RETURN g.geneSymbol

Instruction: Find the gene(s) that are subject to decreased expression by the drug Yohimbine.
Request: Drug-CHEMICALDECREASESEXPRESSION-Gene
Memgraph request: MATCH (d:Drug {commonName: "Yohimbine"})-[:CHEMICALDECREASESEXPRESSION]-(g:Gene) RETURN g.geneSymbol

Instruction: Determine the gene symbol for "cytochrome c oxidase assembly factor 7"
Request: Datatype-GENE-NAME-!cytochrome c oxidase assembly factor 7!
Memgraph request: MATCH (g:Gene) WHERE toLower(g.commonName) = toLower("cytochrome c oxidase assembly factor 7") RETURN g.geneSymbol

"""
    memgraph_prompt_3 = """Instruction: {instruction}
Do not get confused with the above examples and return only the Cypher query for the following question: {cypher}"""

    debug_code_prompt = """You will be given a Python code snippet that failed to execute. You must debug the code snippet and provide the corrected code snippet.
The instruction is as follows:
{instruction}

The code snippet is as follows:
{code}

The error message is as follows:
{error}

You must correct the code snippet and provide the corrected code snippet. Return only the code."""


    output_prompt = """Question:
{question}

Train of thoughts:
{steps}

Describe the thought process above, and then answer the question. Do not include any additional information or reasoning beyond what is provided.

Question:
{question}"""
    def __init__(self,vector_db = None, lm = None, memgraph_client = None, node_types = "", relationship_types = "", logger: logging.Logger = None):
        self.vector_db = vector_db
        self.lm = lm
        self.memgraph_client = memgraph_client
        self.node_types = node_types
        self.relationship_types = relationship_types
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
            if (input is None or input == "") and kwargs["phase"] == "planning":
                return self.planning_prompt.format(question=question, node_types=self.node_types, relationship_types=self.relationship_types)
            elif kwargs["phase"] == "plan_assessment":
                return self.plan_assessment_prompt.format(question=question, approach_1=input[0], approach_2=input[1], approach_3=input[2], node_types=self.node_types, relationship_types=self.relationship_types)
            elif kwargs["phase"] == "python_conversion":
                return self.python_conversion_prompt.format(instructions=input)
            elif kwargs["phase"] == "xml_conversion":
                return self.xml_conversion_prompt.format(instructions=input)
            elif kwargs["phase"] == "xml_cleanup":
                return self.xml_cleanup_prompt.format(xml=input, node_types=self.node_types, relationship_types=self.relationship_types)
            # elif kwargs["phase"] == "steps":
            #     #check for all steps in got_steps for predecessors. If no predecessors, assign the self as predecessor
            #     if "StepID" not in kwargs:
            #         return None
            #     else:
            #         current_instruction = kwargs["instruction"]
            #         self.logger.info(f"Current instruction: {current_instruction}")
            #         if current_instruction['InstructionType'] == "KnowledgeRequest":
            #             knowledge_requests = current_instruction["KnowledgeRequests"]
            #         elif current_instruction['InstructionType'] == "Function":
            #             return self.function_prompt.format(function=current_instruction["Function"], node_types=self.node_types, relationship_types=self.relationship_types)
            elif kwargs["phase"] == "output":
                steps = ""
                for step in kwargs["instructions"]:
                    steps += "Step " + step["StepID"] + ": " + step["Instruction"] + "\n"
                    steps += "Code: " + step["Code"][0] + "\n"
                    output = input[step["StepID"]]
                    if len(str(output)) > 256:
                        output = str(output)[:256] + "..."
                    steps += "Output: " + str(output) + "\n\n"
                return self.output_prompt.format(question=question, steps=steps)
        else:
            raise AssertionError(f"Method {method} is not implemented yet.")
    def get_knowledge(self,knowledge_request,instruction):
        statement_to_embed = knowledge_request
        if statement_to_embed == "" or statement_to_embed is None:
            return []
        # statement_to_embed = self.lm.get_response_texts(
        #     self.lm.query(self.knowledge_request_adjustment_prompt.format(node_types=self.node_types,relationship_types=self.relationship_types, statement_to_embed=statement_to_embed), num_responses=1)
        # )
        # statement_to_embed = statement_to_embed[0]
        statement_to_embed_cleaned = statement_to_embed.replace("!","")
        # If it's a cypher query, then execute the query and return the results directly
        if self.memgraph_client is not None:
            knowledge_array = self.memgraph_client.execute(self.lm, self.memgraph_prompt_1.format(schema=self.memgraph_client.schema) + str(self.memgraph_prompt_2) + str(self.memgraph_prompt_3.format(instruction=str(instruction),cypher=str(statement_to_embed))),str(statement_to_embed))
            self.logger.info(f"Cypher knowledge for {statement_to_embed}: {str(knowledge_array)}")
            #backup if the memgraph client fails or doesn't provide any knowledge
            if knowledge_array == [] and self.vector_db is not None:
                if statement_to_embed.count("!") >= 2:
                    #if there is only once specific node and nothing else, then return the knowledge
                    embedded_question = self.lm.get_embedding(statement_to_embed_cleaned)
                    node_filters = re.findall(r'!(.*?)!', statement_to_embed)
                    knowledge_arrays = []
                    for node_filter in node_filters:
                        knowledge_array,distances = self.vector_db.get_knowledge(embedded_question, keyword_filter = node_filter)
                        if len(node_filters) > 1:
                            for node_filter in node_filters:
                                knowledge_array = [knowledge for knowledge in knowledge_array if node_filter in knowledge]
                        knowledge_arrays.append(knowledge_array)
                    knowledge_array = list(set().union(*knowledge_arrays))
                else:
                    embedded_question = self.lm.get_embedding(statement_to_embed)
                    knowledge_array,distances = self.vector_db.get_knowledge(embedded_question)
                self.logger.info(f"Vector DB knowledge for {statement_to_embed}: {knowledge_array}")
            return knowledge_array
        return "knowledge"
    def generate_debug_code_prompt(self, code, instruction, e):
        prompt = self.debug_code_prompt.format(instruction=instruction, code=code, error=e)
        return prompt
