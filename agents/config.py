config = {
    "azuregpt-4o-mini" : {
        "model_id":"gpt-4o-2024-05-13", 
        "prompt_token_cost": 0.001,
        "response_token_cost": 0.002,
        "temperature": 0.7,
        "max_tokens": 4096,
        "stop": None,
        "api_version": "2023-03-15-preview",
        "api_base": "",
        "api_key": "",
        "embedding_id":"text-embedding-3-small"
    },
    # "ollama" : {
    #     "model_id":"qwen2.5:32b", 
    #     "prompt_token_cost": 0.0,
    #     "response_token_cost": 0.0,
    #     "temperature": 0.7,
    #     "max_tokens": 8000,
    #     "stop": None,
    # },
    "memgraph" : {
        "host": "alzkb.ai",
        "port": 7687
    }
    # "neo4j":{
    #     'host': 'neo4j.het.io/',
    #     'port': 7687
    # }
}