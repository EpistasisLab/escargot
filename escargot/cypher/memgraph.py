from gqlalchemy import Memgraph
import os
from typing import Dict
import json

class MemgraphClient:
    def __init__(self, config_path):
        self.config: Dict = None
        self.load_config(config_path)
        self.config: Dict = self.config["memgraph"]
        self.host = self.config["host"]
        self.port = self.config["port"]
        self.memgraph = Memgraph(host=self.host, port=self.port)

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

    def execute(self, lm, query):
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
        
        memgraph_results = self.memgraph.execute_and_fetch(response)
        memgraph_results = list(memgraph_results)[0]
        
        results = []
        # get the value for each key in the memgraph_results
        for key in memgraph_results.keys():
            results.append(memgraph_results[key])

        return results