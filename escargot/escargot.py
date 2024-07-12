#TODO:
#semi-auto, full-auto modes:
    #step function to step into the next on the execution queue
    #chat log function for outputting later
#just like got_steps variable, we need an entire edge list and accessibility to any step
#accessing and rerunning steps (what are the existing values that need to be updated, new variables? [reran, attempted, previous_executions])


import escargot.language_models as language_models
import escargot.controller as controller
from escargot.parser import ESCARGOTParser
from escargot.prompter import ESCARGOTPrompter
from escargot import operations

#vectorized knowledge
from escargot.vector_db.weaviate import WeaviateClient

#memgraph Cypher
import escargot.cypher.memgraph as memgraph

class Escargot:

    def __init__(self, config: str, model_name: str = "azuregpt35-16k"):
        self.lm = language_models.AzureGPT(config, model_name=model_name)
        self.vdb = WeaviateClient(config)
        if self.vdb.client is None:
            self.vdb = None
        self.memgraph_client = memgraph.MemgraphClient(config)
        if self.memgraph_client.memgraph is None:
            self.memgraph_client = None
        self.node_types = ""
        self.relationship_types = ""
        self.question = ""
        self.controller = None
        self.operations_graph = None
    
    def ask(self,question,num_strategies=3):

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
                ESCARGOTPrompter(memgraph_client = self.memgraph_client,vector_db = self.vdb, lm=self.lm,node_types=self.node_types,relationship_types=self.relationship_types),
                ESCARGOTParser(),
                {
                    "question": question,
                    "input": "",
                    "phase": "planning",
                    "method" : "got",
                    "num_branches_response": num_strategies,
                }
            )
            self.controller.run()
        except Exception as e:
            print("exception:",e)

        self.controller.logger.handlers = []
        self.controller.logger = None

        self.operations_graph = self.controller.graph.operations
        del self.controller



