from gqlalchemy import Memgraph
import os
from typing import Dict
import json
import logging

class MemgraphClient:
    def __init__(self, config_path):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config: Dict = None
        self.load_config(config_path)
        self.config: Dict = self.config["memgraph"]
        self.host = self.config["host"]
        self.port = self.config["port"]
        self.memgraph = Memgraph(host=self.host, port=self.port)
        self.num_responses = 3
        self.cache = {}

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
        if query in self.cache:
            results = self.cache[query]
        else:
            num_responses = self.num_responses
            if num_responses == 1:
                response = self.chat([{"role": "system", "content": query}], num_responses)
            else:
                next_try = num_responses
                total_num_attempts = num_responses
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
                        
                        print("Memgraph request:",response)

                        memgraph_results = self.memgraph.execute_and_fetch(response)
                        memgraph_results = list(memgraph_results)
                        num_responses -= next_try
                        next_try = min(num_responses, next_try)
                    except Exception as e:
                        next_try = (next_try + 1) // 2
                        self.logger.warning(
                            f"Error in memgraph: {e}, trying again {next_try}"
                        )
                        total_num_attempts -= 1

            results = []
            # get the value in the dictionary x in memgraph_results
            for value in memgraph_results:
                for key, val in value.items():
                    results.append(val)
            self.cache[query] = results

        return results