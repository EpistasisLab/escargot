
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

class DataExplorerESCARGOTParser(ESCARGOTParser):
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
                    if state["phase"] == "programming":
                        new_state = state.copy()
                        new_state["input"] = text
                        new_state["previous_phase"] = "programming"
                        new_state["phase"] = "steps"
                        new_state["num_branches_response"] = 1
                        # new_state["generate_successors"] = 1
                        new_state["full_code"] = text
                        new_state['instructions'] = [{
                            "StepID": '1',
                            "Instruction": state["question"],
                            "Code": [text.strip()]
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

class DataExplorerESCARGOTPrompter:
    """
    ALZKBPrompter provides the generation of prompts specific to the
    ALZKB example for the language models.

    Inherits from the Prompter class and implements its abstract methods.
    """
    programming_prompt_with_dataset = """You are a brilliant data scientist and Python programmer who will generate a python script to answer a request.

Here are some notes about your environment:
Only use default Python packages, pandas, numpy, scikit-learn, seaborn, dill, and matplotlib.
If you need to access the dataset, it is located at ../dataset.feather and you can load it using the pandas library with the following code: df = pd.read_feather('../dataset.feather')
The dataset has the following columns: {columns}

Any assets created in previous steps can be retrieved using the following pickle files associated with the step. For example, if you created an asset in step 1, you can load it using the following code: asset_1 = dill.load(open('asset_1.pkl', 'rb'))
Be careful with the pickle file dictionary structure when accessing it. When you load the pickle file, assume it is already in the correct format and you can access the data directly without using eval or exec.
Pickle files:
{assets}

Do NOT assume that you will get the exact columns when you are going to create a subset of the original dataset with columns provided by the pickle files. Use a ".isin()" function to make sure the columns are exact.
When creating a new dataset, if possible, make your best judgement on trying keeping the target or outcome variable for the dataset.
Do NOT use "eval" or "exec" operations for the code generation.
If you generate any plots or any visualizations, generate code that will save them in the working directory. Do NOT show the plots or display them in the notebook.

You are performing the last step in the following:
{steps}

Respond with Python script only."""

    programming_prompt_without_dataset = """You are a brilliant data scientist and Python programmer who will generate a python script to answer a request.

Here are some notes about your environment:
Only use default Python packages, pandas, numpy, scikit-learn, seaborn, dill, and matplotlib.

Any assets created in previous steps can be retrieved using the following pickle files associated with the step. For example, if you created an asset in step 1, you can load it using the following code: asset_1 = dill.load(open('asset_1.pkl', 'rb'))
Be careful with the pickle file dictionary structure when accessing it. When you load the pickle file, assume it is already in the correct format and you can access the data directly without using eval or exec.
Pickle files:
{assets}

Do NOT assume that you will get the exact columns when you are going to create a subset of the original dataset with columns provided by the pickle files. Use a ".isin()" function to make sure the columns are exact.
When creating a new dataset, if possible, make your best judgement on trying keeping the target or outcome variable for the dataset.
Do NOT use "eval" or "exec" operations for the code generation.
If you generate any plots or any visualizations, generate code that will save them in the working directory. Do NOT show the plots or display them in the notebook.
If you are saving a file that can be stored in a csv file, save it as a csv in the working directory.

You are performing the last step in the following:
{steps}

Respond with Python script only."""

    python_assessment_prompt = """You are a Python expert who will assess the Python code generated from the instructions. You will receive a question that requires a knowledge graph to answer and three approaches that will try to resolve that question. You will assess the quality of the Python code generated from the approaches.
You will assess the Python code based on the following criteria:
1. The code should be clear and concise.
2. The code should follow the instructions provided.
3. The code should not define any new functions.
4. The code should only use default Python packages and numpy.
5. The code should not contain placeholder variables, such as ['gene1', 'gene2', 'gene3']. Instead, the code should contain the actual variables from the instructions.
6. The code should have comments for each step that describe the instructions for that step.
7. Any code that requires the knowledge graph should use the knowledge_extract(x) function, where x is the specific knowledge you are requesting.

Return an XML formatted list with all the code snippets in Code tags. Each code snippet should be within <Code> tags and will have an incremental <CodeID> value within it. The score should be an integer between 1-10 within <Score> tags.
An example is as follows:
<Codes>
  <Code>
    <CodeID>1</CodeID>
    <Score>...</Score>
  </Code>
  ...
</Codes>

Only return the XML.

Here is the problem:
{question}
Here is your instructions:
{approach}

Here is code snippet 1:
{approach_1}

Here is code snippet 2:
{approach_2}

Here is code snippet 3:
{approach_3}
"""
    
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

    function_prompt="""{function}"""

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

    output_prompt = """Given the following code, describe what the code does and provide all the new files created in the working directory.
{steps}"""
    def __init__(self,lm = None, logger: logging.Logger = None):
        self.lm = lm
        self.logger = logger
     
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
        if (input is None or input == "") and kwargs["phase"] == "programming":
            if kwargs["columns"] != []:
                return self.programming_prompt_with_dataset.format(columns=kwargs["columns"], assets=kwargs["assets"], steps=kwargs["steps"])
            else:
                return self.programming_prompt_without_dataset.format(assets=kwargs["assets"], steps=kwargs["steps"])
        elif kwargs["phase"] == "plan_assessment":
                return self.plan_assessment_prompt.format(question=question, approach_1=input[0], approach_2=input[1], approach_3=input[2], agents=self.agents)
        elif kwargs["phase"] == "aggregate_agents":
                return self.aggregate_agents_prompt.format(agents=self.agents, steps=input)
        elif kwargs["phase"] == "python_conversion":
                return self.python_conversion_prompt.format(instructions=question)
        elif kwargs["phase"] == "code_assessment":
                return self.python_assessment_prompt.format(question=question, approach=input, approach_1=input[0], approach_2=input[1], approach_3=input[2])
        elif kwargs["phase"] == "xml_conversion":
                return self.xml_conversion_prompt.format(instructions=input)
        elif kwargs["phase"] == "output":
                steps = ""
                for step in kwargs["instructions"]:
                    steps += "Step " + step["StepID"] + ": " + step["Instruction"] + "\n"
                    steps += "Code: " + step["Code"][0] + "\n"
                    output = input[step["StepID"]]
                    # if len(str(output)) > 256:
                    #     output = str(output)[:256] + "..."
                    steps += "Output: " + str(output) + "\n\n"
                if kwargs["answer_type"] == "array":
                    return self.array_output_prompt.format(question=question, steps=steps)
                else:
                    return self.output_prompt.format(question=question, steps=steps)
        else:
            raise AssertionError(f"Method {method} is not implemented yet.")
    
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

class DataExplorerEscargot(Escargot):
    def __init__(self, config: str, dataframe_columns: list = [], file_descriptions: str = "", plans: str = "", model_name: str = "azuregpt35-16k"):
        logger = logging.getLogger(__name__)
        self.logger = logger
        self.log = ""
        self.lm = language_models.AzureGPT(config, model_name=model_name, logger=logger)
        self.memory = None
        self.question = ""
        self.controller = None
        self.operations_graph = None
        self.vdb = None
        self.graph_client = None
        self.dataframe_columns = dataframe_columns
        self.file_descriptions = file_descriptions
        self.plans = plans

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
                DataExplorerESCARGOTPrompter(lm=self.lm, logger = self.logger),
                DataExplorerESCARGOTParser(self.logger),
                self.logger,
                Coder(file_descriptions = self.file_descriptions),
                {
                    "question": question,
                    "input": "",
                    "phase": "programming",
                    "method" : "got",
                    "num_branches_response": num_strategies,
                    "answer_type": answer_type,
                    "columns": self.dataframe_columns,
                    "assets": self.file_descriptions,
                    "steps": self.plans
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
    
    def initialize_controller(self, question, answer_type = 'natural', num_strategies=3, debug_level = 0, memory_name = "escargot_memory", max_run_tries = 3):
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
        if self.controller is not None:
            del self.controller
            self.controller = None

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
                DataExplorerESCARGOTPrompter(lm=self.lm, logger = self.logger),
                DataExplorerESCARGOTParser(self.logger),
                self.logger,
                Coder(),
                {
                    "question": question,
                    "input": "",
                    "phase": "planning",
                    "method" : "got",
                    "num_branches_response": num_strategies,
                    "answer_type": answer_type
                }
            )
            self.controller.max_run_tries = max_run_tries
        except Exception as e:
            self.logger.error("Error executing controller: %s", e)
        #remove logger
        self.finalize_logger(log_stream, c_handler, f_handler)

    def generate_plan(self, question, num_strategies=3, debug_level = 0, memory_name = "escargot_memory", max_run_tries = 3):
        """
        Generate a plan to answer a question.

        :param question: The question to ask.
        :type question: str
        :param num_strategies: The number of strategies to generate. Defaults to 3.
        :type num_strategies: int
        :return: The answer to the question.
        :rtype: str
        """
        # print(f"Generating plan for question: {question}")

        self.initialize_controller(question, answer_type = 'natural', num_strategies=num_strategies, debug_level = debug_level, memory_name = memory_name, max_run_tries = max_run_tries)
        # two steps to generate the plan from prompting to assessing
        self.step()
        self.step()
        self.step()
        self.step()
        # print(f"Final thought: {self.controller.final_thought.state}")
        output = ""
        if self.controller.final_thought is not None:
            output = self.controller.final_thought.state['input']
        return output
    