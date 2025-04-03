from gqlalchemy import Neo4j
import os
from typing import Dict
import json
import logging

class Neo4jClient:
    def __init__(self, config_path, logger):
        self.logger = logger
        self.config: Dict = None
        if type(config_path) == dict:
            self.config = config_path
        else:
            self.load_config(config_path)
        if "neo4j" not in self.config:
            self.client = None
        else:
            self.config: Dict = self.config["neo4j"]
            self.host = self.config["host"]
            self.port = self.config["port"]
            self.client = Neo4j(host=self.host, port=self.port)
            self.num_responses = 7
            self.cache = {}
            self.schema = None

    def get_schema(self):
        SCHEMA_QUERY = """
        CALL db.schema.visualization()
        """
        tries = 3
        while tries > 0:
            try:
                db_structured_schema = list(self.client.execute_and_fetch(SCHEMA_QUERY))
                assert db_structured_schema is not None
                structured_schema = db_structured_schema[0]
                break
            except Exception as e:
                tries -= 1
                self.logger.error(f"Error in graph client: {e}, trying again {tries}")
        
        formatted_node_props = []
        node_names = []

        for node in structured_schema["nodes"]:
            node_name = list(node.labels)[0]
            node_names.append(node_name)
            formatted_node_props.append(
                f"Node name: '{node_name}', Node properties: {dict(node)['indexes']}"
            )

        formatted_rels = []
        relationship_names = []
        for rel in structured_schema["relationships"]:
            relationship_names.append(rel.type)
            formatted_rels.append(
                f"(:{list(rel.nodes[0].labels)[0]})-[:{rel.type}]->(:{list(rel.nodes[1].labels)[0]})"
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

    def clear_cache(self) -> None:
        """
        Clear the response cache.
        """
        self.cache = {}

    def execute(self, lm, query, statement):
        # if statement in self.cache:
        #     results = self.cache[statement][1]
        # else:
            num_responses = self.num_responses
            response = ''
            responses_hash = set()
            client_results = []
            iter = 0
            responses = lm.get_response_texts(
                lm.query(query, num_responses=num_responses)
            )
            empty_responses = True
            while iter < num_responses and empty_responses:
                try:
                    response = responses[iter]
                    iter += 1
                    # Check if the response is in the hash set
                    if response in responses_hash:
                        continue
                    responses_hash.add(response)
                    # Remove "Answer:" from the response
                    if response.startswith("Answer:"):
                        response= response[8:].strip()
                    
                    if response == "" or response == None:
                        continue
                    #remove ```cypher from the response
                    response = response.replace("```cypher", "")

                    #remove ``` from anywhere in the response
                    response = response.replace("```", "")

                    #remove \n from the response
                    response = response.replace("\n", " ")

                    #TODO: tweak if necessary. Get rid of directionality in the response
                    response = response.replace("<-", "-")
                    response = response.replace("->", "-")
                    
                    self.logger.info(f"Executing client for statement: {statement}, response: {response}")
                    client_results = self.client.execute_and_fetch(response)
                    client_results = list(client_results)
                    if client_results != []:
                        empty_responses = False
                        
                except Exception as e:
                    self.logger.error(f"Error in client_results: {e}, trying again {iter}")
            self.logger.info(f"Graph Client results: {client_results}")
            # results = []
            # # get the value in the dictionary x in memgraph_results
            # for value in memgraph_results:
            #     for key, val in value.items():
            #         if val != [] and val != None and val != "":
            #             if type(val) == list:
            #                 for v in val:
            #                     results.append(v)
            #             else:
            #                 results.append(val)
            # #get unique values
            # results = list(set(results))
            # self.cache[statement] = [response,results]

            return client_results, response