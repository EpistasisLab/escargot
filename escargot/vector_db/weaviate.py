import weaviate
import os
from openai import AzureOpenAI
import numpy as np
from typing import Dict
import json
import logging

weaviate_client = None

class WeaviateClient:
    def __init__(self, config_path, logger):
        self.config: Dict = None
        self.logger = logger
        if type(config_path) == dict:
            self.config = config_path
        else:
            self.load_config(config_path)
        if "weaviate" not in self.config:
            self.client = None
        else:
            self.config: Dict = self.config["weaviate"]
            self.url = self.config["url"]
            self.api_key = self.config["api_key"]
            self.db = self.config["db"]
            self.limit = self.config["limit"]
            self.client = self.get_client()
        
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

        
    def get_client(self):
        auth_config = weaviate.AuthApiKey(api_key=self.api_key)
        global weaviate_client
        if weaviate_client is None:
            weaviate_client = weaviate.Client(
                url=self.url,
                auth_client_secret=auth_config
            )
        return weaviate_client
    
    def query_bm25(self, properties=["knowledge"], query_string="test", additional="score", limit=3):
        return ((self.client.query
            .get(self.db, properties))
            .with_bm25(
                query=query_string,
                properties=properties  # this does not need to be the same as columns
            )
            .with_additional(additional)
            .with_limit(limit)
            .do()
        )
    
    
    def query_near_text(self, properties=["knowledge"], near_text=["gene"], additional="score", limit=3):
        return ((self.client.query
            .get(self.db, properties))
            .with_near_text({
                "concepts": near_text
            })
            .with_limit(limit)
            .with_additional(additional)
            .do()
        )
    
    
    def query_my_near_text(self, prompt, properties=["knowledge"], additional="score", limit=3):
        vector = {
            "vector": prompt
        }
        return self.query_near_vector(self.db, properties, near_vector=vector, additional=additional, limit=limit)
    
    def query_near_vector(self, properties=["knowledge"], near_vector={}, additional="score", limit=3):
        return ((self.client.query
            .get(self.db, properties))
            .with_near_vector({'vector':near_vector})
            .with_limit(limit)
            .with_additional(additional)
            .do()
        )
    
    def query_with_hybrid(self, properties=["knowledge"], near_vector=[],  near_text='', additional="score", autocut=5):
        return ((self.client.query
            .get(self.db, properties))
            .with_hybrid(query = near_text, vector = near_vector, alpha = 0.8)
            # .with_limit(limit)
            .with_autocut(autocut)
            .with_additional(additional)
            .do()
        )
        
    def object_count(self):
        return ((self.client.query
            .aggregate(self.db)
            .with_meta_count()
            .do()))
        
    
    def get_knowledge(self,embedded_question, max_tokens=4000, max_distance = 0.3, min_score = 0.003, keyword_filter=''):
        if keyword_filter != '':
            knowledge_array = self.query_with_hybrid(near_vector=embedded_question, near_text=keyword_filter, additional=["score"])
        else:
            knowledge_array = self.query_near_vector(near_vector=embedded_question, additional=["distance"], limit=self.limit)
        # knowledge_array = self.query_near_vector(near_vector=embedded_question, additional=["distance"], limit=self.limit)
        knowledge_array = knowledge_array["data"]["Get"][self.config["db"]]
        in_context = []
        distances = []
        cur_tokens = 0
        for knowledge in knowledge_array:
            if keyword_filter != '' and keyword_filter.lower() not in knowledge["knowledge"].lower():
                continue

            if keyword_filter == '' :
                if knowledge['_additional']["distance"] > max_distance:
                    break
            else:
                if float(knowledge['_additional']["score"]) < min_score:
                    break
            
            cur_tokens += len(knowledge["knowledge"].split(" "))
            if cur_tokens > max_tokens:
                break
            in_context.append(knowledge["knowledge"])

            if keyword_filter == '' :
                distances.append(knowledge['_additional']["distance"])
            else:  
                distances.append(float(knowledge['_additional']["score"]))
            
        return in_context, distances
