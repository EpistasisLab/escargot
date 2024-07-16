from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List
import re


class ESCARGOTPrompter:
    """
    ALZKBPrompter provides the generation of prompts specific to the
    ALZKB example for the language models.

    Inherits from the Prompter class and implements its abstract methods.
    """
    planning_prompt = """You are a brilliant strategic thinker with access to a knowledge base. You will receive a question that will require the knowledge base to answer. The knowledge base is built from a knowledge graph, but not the knowledge graph itself. You will break down the question into steps. If knowledge needs to be pulled from the knowledge base, you will provide what specific relationships or node information is necessary. You do not need to try to answer the question but simply plan out the steps.

The knowledge graph contains node types: {node_types}.
The knowledge graph contains relationships: {relationship_types}.

Only show the steps you will take and a small description for each step. If you can determine the knowledge graph relationship that can provide insight in the step, provide the relationship in it and if possible the specific node name, not the node type. If a question require a specific relationship between two specific nodes, provide the specific nodes in the relationship.

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

    xml_conversion_prompt = """You will be given a set of instructions and you must convert the instructions into XML with the following rules:

The knowledge graph contains node types: {node_types}.
The knowledge graph contains relationships: {relationship_types}.

Format your response in XML format, where the steps will be within <Instructions> tags. Each step will be within <Step> tags and will have an incremental <StepID> value within it. The full description of the step will be put in the <Instruction> tags within the <Step>. Following the <Instruction>, there should only be one type of instruction within a <Step>: <KnowledgeRequest> or <Function>. Every step should have filled at least one of these instructions.

<KnowledgeRequest>:
 If a request to the knowledge graph needs to be made within a step, you must include the knowledge requests as simple single node to Relationship to node format. Each <KnowledgeRequest> should have an identifier in <KnowledgeID> tags such as Knowledge_1 and Knowledge_2.  Any knowledge request should have the format of <Knowledge> tag where it labeled with: Node Name-Relationship-Node Name. There must be at least one specific node that is requested such as a specific gene or disease. If there are any specific nodes and not a node type, put a ! before and after the word. For instance, Alzheimer's is a specific Disease node, so it should be labeled "!Alzheimer's Disease!". Another example are of gene symbols, which are specific genes, so they should be labeled "!APOE!". 
Each <KnowledgeRequest> should contain a specific node and should not be between two node types. For instance, "Drug-DRUG TREATS DISEASE-!Alzheimer's Disease!" is correct, but "Drug-DRUG TREATS DISEASE-Disease" is incorrect.        
If you detect two specific keywords in the query, you can use both of them in a single <Request> tag. For instance, "!APOE!-GENE ASSOCIATES WITH DISEASE-!Alzheimer's Disease!" is correct, instead of having two separate <Request> tags. Most requests should have one specific keyword. If a step requests for only a single node, omit that step altogether. You must be sure that the relationship is from the above list as well as the node types.

Here is an example <KnowledgeRequest> requesting for all body parts connected to the gene STYXL2: 
<KnowledgeRequest>
    <KnowledgeID>Knowledge_2</KnowledgeID>
    <Node>BODYPART-GENE EXPRESSES-!STYXL2!</Node>
</KnowledgeRequest>

<Function>:
You will have functionality of running array based functions that the machine will execute:
  UNION(x,y): This function returns a distinct union of elements that are in set x and set y
  INTERSECT(x,y): This function returns all elements that are found in both set x and set y
  DIFFERENCE(x, y): This function returns the elements that are in set x but not in set y. It's useful for finding the elements unique to one set compared to another. 
Any reference to arrays determined by previous steps should be by either the <StepID> identifier or <KnowledgeID> identifier. There should be no Knowledge Requests within a Function, only the identifiers. If an request is needed, the Knolwedge Request should be done within the same step.

If a function needs to be run such as UNION, INTERSECT, or DIFFERENCE using the knowledge, that should be within <Function> tags with nothing else other than the function and its variables.

Here is an example of a <Function> requesting the Intersect of two arrays from two knowledge requests:
<Function>
    INTERSECT(Knowledge_1, Knowledge_2)
</Function>

Here is an example XML:
<Instructions>
    <Step>
        <StepID>1</StepID>
        <Instruction>
            Find Body Parts Over-Expressing Gene METTL5
        </Instruction>
        <KnowledgeRequest>
                <KnowledgeID>Knowledge_1</KnowledgeID>
                <Node>BODYPART-BODYPART OVER EXPRESSES GENE-!METTL5!</Node>
            </KnowledgeRequest>
    </Step>
    <Step>
        <StepID>2</StepID>
        <Instruction>
            Find Body Parts Over-Expressing Gene STYXL2
        </Instruction>
        <KnowledgeRequest>
            <KnowledgeID>Knowledge_2</KnowledgeID>
            <Node>BODYPART-BODYPART OVER EXPRESSES GENE-!STYXL2!</Node>
        </KnowledgeRequest>
    </Step>
    <Step>
        <StepID>3</StepID>
        <Instruction>
            List the intersect of body parts
        </Instruction>
        <Function>
            INTERSECT(Knowledge_1, Knowledge_2)
        </Function>
    </Step>
</Instructions>

Outside of the <Instructions> tag, add an edge list in <EdgeList>, where information from one step to another will be listed. Each edge will be within <Edge> tags, and the edge would be in the format StepID1-StepID2 which describes that StepID1 directs to StepID2.
Do not include any other tags other than the ones mentioned above.

Here are the instructions you must convert:
{instructions}"""

    xml_conversion_prompt_1 = """You will be given a set of instructions and you must convert the instructions into XML with the following rules:

The knowledge graph contains node types: {node_types}.         
The knowledge graph contains relationships: {relationship_types}.

Format your response in XML format, where the steps will be within <Instructions> tags. Each step will be within <Step> tags and will have an incremental <StepID> value within it. The full description of the step will be put in the <Instruction> tags within the <Step>. Following the <Instruction>, there should only be one type of instruction within a <Step>: <KnowledgeRequest>, <For>, or <Function>

<KnowledgeRequest>:
 If a request to the knowledge graph needs to be made within a step, you must include the knowledge requests as simple single node to Relationship to node format. Each <KnowledgeRequest> should have an identifier in <KnowledgeID> tags such as Knowledge_1 and Knowledge_2.  Any knowledge request should have the format of <Knowledge> tag where it labeled with: Node Name-Relationship-Node Name. There must be at least one specific node that is requested such as a specific gene or disease. If there are any specific nodes and not a node type, put a ! before and after the word. For instance, Alzheimer's is a specific Disease node, so it should be labeled "!Alzheimer's Disease!". Another example are of gene symbols, which are specific genes, so they should be labeled "!APOE!". 
Each <KnowledgeRequest> should contain a specific node and should not be between two node types. For instance, "Drug-DRUG TREATS DISEASE-!Alzheimer's Disease!" is correct, but "Drug-DRUG TREATS DISEASE-Disease" is incorrect.        
If you detect two specific keywords in the query, you can use both of them in a single <Request> tag. For instance, "!APOE!-GENE ASSOCIATES WITH DISEASE-!Alzheimer's Disease!" is correct, instead of having two separate <Request> tags. Most requests should have one specific keyword. If a step requests for only a single node, omit that step altogether. You must be sure that the relationship is from the above list as well as the node types.

Here is an example <KnowledgeRequest> requesting for all body parts connected to the gene STYXL2: 
<KnowledgeRequest>
    <KnowledgeID>Knowledge_2</KnowledgeID>
    <Node>BODYPART-GENE EXPRESSES-!STYXL2!</Node>
</KnowledgeRequest>

<Function>:
You will have functionality of running array based functions that the machine will execute:
  UNION(x,y): This function returns a distinct union of elements that are in set x and set y
  INTERSECT(x,y): This function returns all elements that are found in both set x and set y
  DIFFERENCE(x, y): This function returns the elements that are in set x but not in set y. It's useful for finding the elements unique to one set compared to another. 
Any reference to arrays determined by previous steps should be by either the <StepID> identifier or <KnowledgeID> identifier.

If a function needs to be run such as UNION, INTERSECT, or DIFFERENCE using the knowledge, that should be within <Function> tags with nothing else other than the function and its variables.

Here is an example of a <Function> requesting the Intersect of two arrays from two knowledge requests:
<Function>
    INTERSECT(Knowledge_1, Knowledge_2)
</Function>

<For>:
A For loop maybe necessary for certain steps. The For loop will be determined by entering a <For> tag within the <Step>. The <For> tag must include a <ForVariable> which is a reference to a variable such as Knowledge_1 or StepID_1 or a distinct array (ie. an array of genes). The <For> tag must also include a <ForFunction>, which will include a <KnowledgeRequest> or a <Function> using the above format that will be used to execute for each element in the <ForVariable>. The element is labeled as "ForElement".

Here is an example of a <For> loop that gets the results from Step 1 and executes a knowledge request for each element in it, specifically if the gene list from Step 1 binds to a drug/chemical: 
<For>
    <ForVariable>StepID_1</ForVariable>
    <ForFunction>
        <KnowledgeRequest><Node>DRUG-CHEMICAL BINDS GENE-ForElement</Node></KnowledgeRequest>
    </ForFunction>
</For>

Here is an example XML:
<Instructions>
    <Step>
        <StepID>1</StepID>
        <Instruction>
            Find Body Parts Over-Expressing METTL5
        </Instruction>
        <KnowledgeRequest>
                <KnowledgeID>Knowledge_1</KnowledgeID>
                <Node>BODYPART-BODYPART OVER EXPRESSES GENE-!METTL5!</Node>
            </KnowledgeRequest>
    </Step>
    <Step>
        <StepID>2</StepID>
        <Instruction>
            Find Body Parts Over-Expressing STYXL2
        </Instruction>
        <KnowledgeRequest>
            <KnowledgeID>Knowledge_2</KnowledgeID>
            <Node>BODYPART-BODYPART OVER EXPRESSES GENE-!STYXL2!</Node>
        </KnowledgeRequest>
    </Step>
    <Step>
        <StepID>3</StepID>
        <Instruction>
            Intersect Body Parts
        </Instruction>
        <Function>
            INTERSECT(Knowledge_1, Knowledge_2)
        </Function>
    </Step>
</Instructions>

Outside of the <Instructions> tag, add an edge list in <EdgeList>, where information from one step to another will be listed. Each edge will be within <Edge> tags, and the edge would be in the format StepID1-StepID2 which describes that StepID1 directs to StepID2.
Do not include any other tags other than the ones mentioned above.

Here are the instructions you must convert:
{instructions}"""

    xml_cleanup_prompt = """Given the following XML:
{xml}

If you notice a Knowledge Request where the Node element refers to specific nodes and not a class of nodes (current node types: {node_types}), put a ! before and after the word. For instance, gene APOE would be !APOE! and Alzheimer's Disease would be !Alzheimer's Disease!
If a Node simply tries to retrieve a single node, the Node should just refer to the specific node without anything else. For instance if the node has Drug-BENZATROPINE, it should instead be !BENZATROPINE!
Respond in the same format as the above and nothing else."""

    knowledge_extraction_prompt = """Use the following cypher results for getting {statement_to_embed_cleaned}:
{knowledge}

You will be given a question to answer and MUST ONLY use the above knowledge statements. Assume that the knowledge statements come from a knowledge graph with nodes describing specific things. If the question is to identify a node, simply return that node if you see it in the knowledge statements.
Your answer must be in an array format within single brackets. Please answer the following question and if you cannot answer it, return an empty array: {instruction}"""

    function_prompt="""{function}"""

    output_prompt = """Use the following knowledge to answer the question:
{knowledge}

With the above knowledge, follow this step:
{instruction}
and answer this question: {question}"""

    memgraph_prompt_1= """You are an expert memgraph Cypher translator who understands the question in english and convert to Cypher strictly based on the Neo4j Schema provided and following the instructions below:
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
14. If a node is a Gene, please make sure you use the geneSymbol property, not the commonName.
15. For the return, return only one property, either the commonName or the geneSymbol property. Do not return both properties.
16. The request may not clear and you do your best to assume the proper relationship to use based on the original question.

Original question: {question}

Schema:
"""

    memgraph_prompt_2 = """
Samples:
Question: Provide the cypher for !METTL5!
Answer: MATCH (g:Gene {geneSymbol: "METTL5"}) RETURN g.geneSymbol

Question: !Alzheimer's Disease!-DRUG TREATS DISEASE-Drug
Answer: MATCH (d:Drug)-[:DRUGTREATSDISEASE]->(:Disease {commonName: "Alzheimer\'s Disease"}) RETURN d.commonName

Question: Provide the Cypher for Gene IN PATHWAY-!STYXL2!
Answer: MATCH (g:Gene {geneSymbol: "STYXL2"})-[:GENEINPATHWAY]->(p:Pathway) RETURN p.commonName

"""

    output_prompt = """Question:
{question}

Answer:
{input}
Format the answer."""
    def __init__(self,vector_db = None, lm = None, memgraph_client = None, node_types = "", relationship_types = "") -> None:
        self.vector_db = vector_db
        self.lm = lm
        self.memgraph_client = memgraph_client
        self.node_types = node_types
        self.relationship_types = relationship_types
        pass
     
    def generate_prompt(
        self,
        knowledge_list : Dict,
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
            elif kwargs["phase"] == "xml_conversion":
                # print("strategy",input)
                return self.xml_conversion_prompt.format(instructions=input, node_types=self.node_types, relationship_types=self.relationship_types)
            elif kwargs["phase"] == "xml_cleanup":
                # print("strategy",input)
                return self.xml_cleanup_prompt.format(xml=input, node_types=self.node_types, relationship_types=self.relationship_types)
            elif kwargs["phase"] == "steps":
                #check for all steps in got_steps for predecessors. If no predecessors, assign the self as predecessor
                if "StepID" not in kwargs:
                    return None
                else:
                    current_instruction = kwargs["instruction"]
                    if kwargs['debug_level'] > 1:
                        print("current_instruction:", current_instruction)
                    if current_instruction['InstructionType'] == "KnowledgeRequest":
                        knowledge_requests = current_instruction["KnowledgeRequests"]
                        for knowledge_request in knowledge_requests:
                            statement_to_embed = knowledge_request["Node"]
                            statement_to_embed_cleaned = statement_to_embed.replace("!","")
                            for i in range(statement_to_embed.count("Knowledge_")):
                                #get the number that follows "Knowledge_" and fill it in from the knowledge_list
                                knowledge_number = re.search(r'Knowledge_(\d+)', statement_to_embed).group(1)
                                statement_to_embed = statement_to_embed.replace(f"Knowledge_{knowledge_number}", "!" + str((",").join(knowledge_list[f"Knowledge_{knowledge_number}"])) + "!")
                            
                            if self.vector_db is not None:
                                if statement_to_embed.count("!") >= 2:
                                    #if there is only once specific node and nothing else, then return the knowledge
                                    if statement_to_embed.count("!") == 2 and statement_to_embed[0] == "!" and statement_to_embed[-1] == "!":
                                        return self.knowledge_extraction_prompt.format(question=question, knowledge=statement_to_embed_cleaned, instruction=current_instruction["Instruction"])
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
                                    knowledge = "\n".join(knowledge_array)
                                else:
                                    embedded_question = self.lm.get_embedding(statement_to_embed)
                                    knowledge_array,distances = self.vector_db.get_knowledge(embedded_question)
                                    knowledge = "\n".join(knowledge_array)
                            # If it's a cypher query, then execute the query and return the results directly
                            elif self.memgraph_client is not None:
                                knowledge_array = self.memgraph_client.execute(self.lm, str(self.memgraph_prompt_1.format(question=question)) +self.memgraph_client.schema + str(self.memgraph_prompt_2) + "Return only the Cypher query for: " + str(statement_to_embed),str(statement_to_embed),debug_level=kwargs['debug_level'])
                                knowledge = ",".join(knowledge_array)
                                return knowledge
                            if knowledge == "":
                                knowledge = "No knowledge"
                        return self.knowledge_extraction_prompt.format(question=question,statement_to_embed_cleaned=statement_to_embed_cleaned, knowledge=knowledge, instruction=current_instruction["Instruction"], node_types=self.node_types, relationship_types=self.relationship_types)
                    elif current_instruction['InstructionType'] == "Function":
                        return self.function_prompt.format(function=current_instruction["Function"], node_types=self.node_types, relationship_types=self.relationship_types)
            elif kwargs["phase"] == "output":
                return self.output_prompt.format(question=question, input=input)
        else:
            raise AssertionError(f"Method {method} is not implemented yet.")
