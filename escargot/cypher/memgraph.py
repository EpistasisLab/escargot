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
        results = self.memgraph.execute_and_fetch(response)
        return list(results)