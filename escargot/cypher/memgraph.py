from gqlalchemy import Memgraph
#from langchain_community.graphs import MemgraphGraph
import os
from typing import Dict
import json
import logging

class MemgraphClient:
    def __init__(self, config_path):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config: Dict = None
        if type(config_path) == dict:
            self.config = config_path
        else:
            self.load_config(config_path)
        if "memgraph" not in self.config:
            self.memgraph = None
        else:
            self.config: Dict = self.config["memgraph"]
            self.host = self.config["host"]
            self.port = self.config["port"]
            self.memgraph = Memgraph(host=self.host, port=self.port)
            self.num_responses = 3
            self.cache = {}
            self.schema = None

    def get_schema(self):
        SCHEMA_QUERY = """
        CALL llm_util.schema("raw")
        YIELD *
        RETURN *
        """
        tries = 3
        while tries > 0:
            try:
                db_structured_schema = list(self.memgraph.execute_and_fetch(SCHEMA_QUERY))
                assert db_structured_schema is not None
                structured_schema = db_structured_schema[0]['schema']
                break
            except Exception as e:
                tries -= 1
                self.logger.error(f"Error in memgraph: {e}, trying again {tries}")
        
        formatted_node_props = []
        node_names = []

        for node_name, properties in structured_schema["node_props"].items():
            node_names.append(node_name)
            formatted_node_props.append(
                f"Node name: '{node_name}', Node properties: {properties}"
            )

        formatted_rels = []
        relationship_names = []
        for rel in structured_schema["relationships"]:
            relationship_names.append(rel['type'])
            formatted_rels.append(
                f"(:{rel['start']})-[:{rel['type']}]->(:{rel['end']})"
            )

        self.schema = "\n".join(
            [
                "Node properties are the following:",
                *formatted_node_props,
                "The relationships are the following:",
                *formatted_rels,
            ]
        )
        return (", ").join(node_names), ("\n").join(relationship_names)

    def load_config(self, path: str) -> None:
        """
        Load configuration from a specified path.

        :param path: Path to the config file. If an empty path provided,
                     default is `config.json` in the current directory.
        :type path: str
        """
        if path == "":
            current_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(current_dir, "config.json")

        with open(path, "r") as f:
            self.config = json.load(f)

    def execute(self, lm, query, statement, debug_level=0):
        if statement in self.cache:
            results = self.cache[statement]
        else:
            num_responses = self.num_responses
            response = ''
            if num_responses == 1:
                response = self.chat([{"role": "system", "content": query}], num_responses)
            else:
                next_try = num_responses
                total_num_attempts = num_responses
                memgraph_results = []
                iter = 0
                while memgraph_results == [] and iter < 3:
                    iter += 1
                    while num_responses > 0 and total_num_attempts > 0:
                        try:
                            assert next_try > 0
                            response = lm.get_response_texts(
                                    lm.query(query, num_responses=1)
                                )
                            response = response[0]
                            # Remove "Answer:" from the response
                            if response.startswith("Answer:"):
                                response= response[8:].strip()
                            
                            #remove ```cypher from the response
                            response = response.replace("```cypher", "")

                            #remove ``` from anywhere in the response
                            response = response.replace("```", "")

                            #remove \n from the response
                            response = response.replace("\n", "")
                            
                            if debug_level > 1:
                                print("Memgraph request:",response)

                            memgraph_results = self.memgraph.execute_and_fetch(response)
                            memgraph_results = list(memgraph_results)
                            num_responses -= next_try
                            next_try = min(num_responses, next_try)
                        except Exception as e:
                            next_try = (next_try + 1) // 2
                            if debug_level > 0:
                                print(f"Error in memgraph: {e}, trying again {next_try}")
                            total_num_attempts -= 1

            results = []
            # get the value in the dictionary x in memgraph_results
            for value in memgraph_results:
                for key, val in value.items():
                    results.append(val)
            self.cache[statement] = [response,results]

        return results