import os
import escargot.language_models as language_models
import logging
import io
from escargot import operations
import escargot.controller as controller
import escargot.memory as memory

#vectorized knowledge
from escargot.vector_db.weaviate import WeaviateClient

from escargot.parser import ESCARGOTParser
from escargot.prompter import ESCARGOTPrompter
from escargot.coder import Coder

#memgraph Cypher
import escargot.cypher.memgraph as memgraph

import dill as pickle

class Escargot:

    def __init__(self, config: str, node_types:str = "", relationship_types:str = "", model_name: str = "azuregpt35-16k"):
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
        self.memgraph_client = memgraph.MemgraphClient(config, logger)
        if self.memgraph_client.memgraph is None:
            self.memgraph_client = None
        else:
            if node_types == "" or relationship_types == "":
                self.node_types, self.relationship_types = self.memgraph_client.get_schema()
            else:
                self.node_types = node_types
                self.relationship_types = relationship_types
    
    def setup_logger(self, debug_level):
        log_stream = io.StringIO()
        f_handler = logging.StreamHandler(log_stream)
        c_handler = logging.StreamHandler()
        if debug_level == 0:
            self.logger.setLevel(logging.ERROR)
            c_handler.setLevel(logging.ERROR)
            f_handler.setLevel(logging.ERROR)
        elif debug_level == 1:
            self.logger.setLevel(logging.WARNING)
            c_handler.setLevel(logging.WARNING)
            f_handler.setLevel(logging.WARNING)
        elif debug_level == 2:
            self.logger.setLevel(logging.INFO)
            c_handler.setLevel(logging.INFO)
            f_handler.setLevel(logging.INFO)
        elif debug_level == 3:
            self.logger.setLevel(logging.DEBUG)
            c_handler.setLevel(logging.DEBUG)
            f_handler.setLevel(logging.DEBUG)
        c_format = logging.Formatter('%(asctime)s - %(filename)s - %(funcName)s(%(lineno)d) - %(message)s')
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(c_format)
        self.logger.addHandler(f_handler)
        self.logger.addHandler(c_handler)

        return log_stream, c_handler, f_handler

    def finalize_logger(self,log_stream, c_handler, f_handler):
        self.log += log_stream.getvalue()
        #reset logger
        self.logger.removeHandler(c_handler)
        c_handler.close()
        self.logger.removeHandler(f_handler)
        f_handler.close()

    #debug_level: 0, 1, 2, 3
    #0: no debug, only output
    #1: output, instructions, and exceptions
    #2: output, instructions, exceptions, and debug info
    #3: output, instructions, exceptions, debug info, and LLM output
    def ask(self, question, answer_type = 'natural', num_strategies=3, debug_level = 0, memory_name = "escargot_memory", max_run_tries = 3):
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
        
        self.memory = memory.Memory(self.lm, collection_name = memory_name)
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
            self.controller = controller.Controller(
                self.lm, 
                got, 
                ESCARGOTPrompter(memgraph_client = self.memgraph_client,vector_db = self.vdb, lm=self.lm,node_types=self.node_types,relationship_types=self.relationship_types, logger = self.logger),
                ESCARGOTParser(self.logger),
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
        if self.controller is not None:
            del self.controller
            self.controller = None

        self.memory = memory.Memory(self.lm, collection_name = memory_name)

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
            self.controller = controller.Controller(
                self.lm, 
                got, 
                ESCARGOTPrompter(memgraph_client = self.memgraph_client,vector_db = self.vdb, lm=self.lm,node_types=self.node_types,relationship_types=self.relationship_types, logger = self.logger),
                ESCARGOTParser(self.logger),
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
        except Exception as e:
            self.logger.error("Error initializing controller: %s", e)
        self.finalize_logger(log_stream, c_handler, f_handler)

    def step(self):
        #setup logger
        log_stream, c_handler, f_handler = self.setup_logger(self.logger.level)
        try:
            current_thought = self.controller.execute_step()
        except Exception as e:
            self.logger.error("Error executing controller: %s", e)

        output = ""
        if self.controller.final_thought is not None:
            self.operations_graph = self.controller.graph.operations
            output = self.controller.final_thought.state['input']

        self.logger.warning(f"Output: {output}")

        #remove logger
        self.finalize_logger(log_stream, c_handler, f_handler)
        return current_thought
    
    def quick_chat(self,chat, num_responses=1):
        if self.lm is not None:
            response = self.lm.get_response_texts(
                    self.lm.query(chat, num_responses=num_responses)
                )
            return response[0]
        else:
            return "No language model available."
        
    def go_to_phase(self, phase):
        if self.controller is not None:
            self.controller.go_to_phase(phase)
        
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
        self.initialize_controller(question, answer_type = 'natural', num_strategies=num_strategies, debug_level = debug_level, memory_name = memory_name, max_run_tries = max_run_tries)
        # two steps to generate the plan from prompting to assessing
        self.step()
        self.step()
        output = ""
        if self.controller.final_thought is not None:
            output = self.controller.final_thought.state['input']
        return output
    
    def generate_code_from_plans(self):
        """
        Generate code from plans.

        :return: The code generated from the plans.
        :rtype: str
        """
        if self.controller is None:
            return ""
        if self.controller.final_thought.state['previous_phase'] != "plan_assessment":
            return ""
        # two steps to generate the plan from prompting to assessing
        self.step()
        self.step()
        output = ""
        if self.controller.final_thought is not None:
            output = self.controller.final_thought.state['input']
        return output
    
    def generate_xml_from_code(self):
        """
        Generate XML from code.

        :param code: The code to convert to XML.
        :type code: str
        :param instructions: The instructions to convert to XML.
        :type instructions: str
        :return: The XML generated from the code.
        :rtype: str
        """
        if self.controller is None:
            return ""
        if self.controller.final_thought.state['previous_phase'] != "code_assessment":
            return ""
        # two steps to generate the plan from prompting to assessing
        self.step()
        output = ""
        if self.controller.final_thought is not None:
            output = self.controller.final_thought.state['input']
        return output
    
    def save_controller(self, path = "controller_state.pkl"):
        if self.controller is not None:
            self.controller.save_controller_state(path)
    
    def load_controller(self, path = "controller_state.pkl"):
        if self.controller is not None and os.path.exists(path):
            self.controller = self.controller.load_state(path, self.controller.logger, self.controller.lm, self.controller.prompter, self.controller.parser, self.controller.coder)
            
