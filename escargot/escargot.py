import os
import escargot.language_models as language_models
import logging
import io
import threading
import time
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
import escargot.cypher.neo4j as neo4j

import dill as pickle
from typing import Optional

class Escargot:

    def __init__(self, config: str, node_types:str = "", relationship_types:str = "", model_name: str = "azuregpt35-16k"):
        logger = logging.getLogger(__name__)
        self.logger = logger
        self.log = ""
        if 'ollama' in config:
            self.lm = language_models.Ollama(config, model_name=model_name, logger=logger)
        if 'azuregpt' in config:
            self.lm = language_models.AzureGPT(config, model_name=model_name, logger=logger)
        self.vdb = WeaviateClient(config, self.logger)
        self.memory = memory.Memory(self.lm)
        self.node_types = ""
        self.relationship_types = ""
        self.question = ""
        self.controller = None
        self.operations_graph = None
        if self.vdb.client is None:
            self.vdb = None
        if "memgraph" in config: 
            self.graph_client = memgraph.MemgraphClient(config, logger)
        if "neo4j" in config:
            self.graph_client = neo4j.Neo4jClient(config, logger)
        if self.graph_client.client is None:
            self.graph_client = None
        else:
            if node_types == "" or relationship_types == "":
                self.node_types, self.relationship_types = self.graph_client.get_schema()
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

    def _ask_worker(self, question, answer_type, num_strategies, memory_name, max_run_tries, result_container, exception_container, cancellation_event, max_tokens):
        # Logger is managed by the main 'ask' function thread
        try:
            # Create the Controller
            def create_initial_graph() -> operations.GraphOfOperations:
                operations_graph = operations.GraphOfOperations()
                instruction_node = operations.Generate(1, 1)
                operations_graph.append_operation(instruction_node)
                return operations_graph

            initial_graph = create_initial_graph()

            self.controller = controller.Controller(
                 self.lm,
                 initial_graph,
                 ESCARGOTPrompter(graph_client = self.graph_client,vector_db = self.vdb, lm=self.lm,node_types=self.node_types,relationship_types=self.relationship_types, logger = self.logger),
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
                 },
                 cancellation_event=cancellation_event, # Pass the event
                 max_tokens=max_tokens # Pass max_tokens limit
            )
            self.controller.max_run_tries = max_run_tries
            self.controller.run() # This is the potentially long-running part

            self.operations_graph = self.controller.graph.operations
            output = ""
            if self.controller.final_thought is not None:
                if answer_type == 'natural':
                    output = self.controller.final_thought.state['input']
                elif answer_type == 'array':
                    # output = list(list(self.controller.coder.step_output.values())[-1].values())[-1] # Original line commented out for safety, assuming the next line is the intended one
                    output = list(self.controller.coder.step_output.values())[-1]


            self.logger.warning(f"Output: {output}")

            # Generate and store summary in memory if output was successful
            if output:
                try:
                    summary_prompt = f"Summarize the key information derived from answering the question: '{question}' with the answer: '{str(output)[:500]}...'. If the answer seems like complex data (list, dict), describe what the data represents rather than listing it."
                    summary = self.quick_chat(summary_prompt) # Use quick_chat for summarization

                    if answer_type == 'natural':
                        self.memory.store_memory(text=summary)
                        self.logger.error(f"Stored natural language summary in memory for collection '{self.memory.collection_name}'.")
                    else: # Handles 'array' and potentially other non-natural types
                        self.memory.store_memory(text=summary, data=output)
                        self.logger.error(f"Stored summary with pickled data in memory for collection '{self.memory.collection_name}'.")
                except Exception as e:
                    self.logger.error(f"Failed to generate or store summary in memory: {e}")

            result_container.append(output) # Store result

        except Exception as e:
            self.logger.error("Error executing controller in worker thread: %s", e, exc_info=True) # Log traceback
            exception_container.append(e) # Store exception
        finally:
            # Logger finalization happens in the main ask function
            pass

    #debug_level: 0, 1, 2, 3
    #0: no debug, only output
    #1: output, instructions, and exceptions
    #2: output, instructions, exceptions, and debug info
    #3: output, instructions, exceptions, debug info, and LLM output
    def ask(self, question, answer_type = 'natural', num_strategies=3, debug_level = 0, memory_name = "default", max_run_tries = 3, timeout: Optional[int] = 120, max_tokens: Optional[int] = None):
        """
        Ask a question and get an answer with an optional timeout.

        :param question: The question to ask.
        :type question: str
        :param answer_type: The type of answer to expect. Defaults to 'natural'. Options are 'natural', 'array'.
        :type answer_type: str
        :param num_strategies: The number of strategies to generate. Defaults to 3.
        :type num_strategies: int
        :param debug_level: Debug level (0=Error, 1=Warning, 2=Info, 3=Debug). Defaults to 0.
        :type debug_level: int
        :param memory_name: Name of the memory collection. Defaults to "default".
        :type memory_name: str
        :param max_run_tries: Maximum attempts for controller runs. Defaults to 3.
        :type max_run_tries: int
        :param timeout: Maximum time in seconds to wait for an answer. Defaults to 120. If <= 0 or None, no timeout.
        :type timeout: Optional[int]
        :param max_tokens: Maximum total tokens (prompt + completion) allowed. Defaults to None (no limit).
        :type max_tokens: Optional[int]
        :return: The answer, or a message indicating timeout, token limit exceeded, or error.
        :rtype: str or list (depending on answer_type) or str (status/error message)
        """

        self.memory = memory.Memory(self.lm, collection_name = memory_name)
        #setup logger
        log_stream, c_handler, f_handler = self.setup_logger(debug_level)

        final_output = f"Timeout occurred after {timeout} seconds." # Default timeout message
        result_container = []
        exception_container = []
        cancellation_event = threading.Event() # Create the event

        worker_thread = threading.Thread(
            target=self._ask_worker,
            args=(question, answer_type, num_strategies, memory_name, max_run_tries, result_container, exception_container, cancellation_event, max_tokens), # Pass event and max_tokens to worker
            daemon=True # Set thread as daemon so it doesn't block program exit if main thread finishes
        )

        start_time = time.time()
        worker_thread.start()

        # Only apply timeout if timeout is positive
        effective_timeout = timeout if timeout is not None and timeout > 0 else None
        worker_thread.join(timeout=effective_timeout)
        end_time = time.time()

        timed_out = worker_thread.is_alive()

        if timed_out:
            self.logger.error(f"Timeout reached after {timeout} seconds for question: '{question}'")
            # Note: Without a cooperative cancellation mechanism in Controller,
            # the thread might continue running in the background even after timeout is reported.
            # The daemon=True setting helps ensure it doesn't prevent program exit.
            cancellation_event.set() # Signal the worker thread to stop
            final_output = f"Timeout occurred after {timeout} seconds."
        else:
            # Thread finished
            final_thought_phase = self.controller.final_thought.state.get("phase") if self.controller and self.controller.final_thought else None

            if exception_container:
                 # Log the exception caught in the thread
                 exc = exception_container[0]
                 self.logger.error(f"Exception occurred in worker thread: {exc}", exc_info=True)
                 # Return an error message or raise the exception
                 final_output = f"An error occurred during execution: {exc}"
            elif final_thought_phase == "token_limit_exceeded":
                 self.logger.error(f"Token limit ({max_tokens}) exceeded.")
                 final_output = f"Operation cancelled: Token limit ({max_tokens}) exceeded."
            elif result_container:
                final_output = result_container[0]
            else:
                 # Should not happen if thread finished without exception and without result
                 self.logger.error("Worker thread finished but no result or exception was captured.")
                 final_output = "An unexpected error occurred after thread completion."

        # Finalize logger regardless of outcome
        self.finalize_logger(log_stream, c_handler, f_handler)

        return final_output
    
    def initialize_controller(self, question, answer_type = 'natural', num_strategies=3, debug_level = 0, memory_name = "default", max_run_tries = 3):
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
                ESCARGOTPrompter(graph_client = self.graph_client,vector_db = self.vdb, lm=self.lm,node_types=self.node_types,relationship_types=self.relationship_types, logger = self.logger),
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
        
    def query_memory(self, query: str, max_results: int = 10, metadata: dict = None):
        """
        Query the persistent memory collection.

        :param query: The query string to search for in the memory.
        :type query: str
        :param max_results: The maximum number of results to return. Defaults to 10.
        :type max_results: int
        :param metadata: Optional metadata dictionary to filter results. Defaults to None.
        :type metadata: dict
        :return: The query results from the memory collection.
        :rtype: dict
        """
        if self.memory is None:
            self.logger.error("Memory not initialized. Call 'ask' or 'initialize_controller' first.")
            return None # Or return an empty dict: {"ids": [], "distances": []}
        
        try:
            results = self.memory.query_collection(query, max_results=max_results, metadata=metadata)
            self.logger.info(f"Queried memory collection '{self.memory.collection_name}' with '{query}'. Found {len(results.get('ids', [[]])[0])} results.")
            return results
        except Exception as e:
            self.logger.error(f"Failed to query memory collection '{self.memory.collection_name}': {e}")
            return None

    def go_to_phase(self, phase):
        if self.controller is not None:
            self.controller.go_to_phase(phase)
        
    def generate_plan(self, question, num_strategies=3, debug_level = 0, memory_name = "default", max_run_tries = 3):
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
            
