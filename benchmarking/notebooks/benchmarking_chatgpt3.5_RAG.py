
# Please make sure to set the env file in the VectorEmbeddingSearchingNick_v1



# NICK 
# BASE_GENERATION_TEMPLATE = """
# Answer the following question and return only the answer. If it's multiple choice, return the answer in the format "1", "2", "3", "4", etc. If it's a free text answer, return the answer as a string.
# Question: {question}
# """


# me
# for multiple choice questions
# BASE_GENERATION_TEMPLATE = """
# {question} Please respond in the format 'Answer:<number>'. For instance, if the correct answer to the question falls within choices 1 to 5 and the correct answer is 1, simply respond with 'Answer:1'
# """

# me
# for open ended questions
# BASE_GENERATION_TEMPLATE = """{question} """










import json
import dill
import os
import sys

# Assuming you're running this in a Jupyter notebook or an interactive session
notebook_dir = os.getcwd()  # Get the current working directory
sys.path.append(os.path.abspath(os.path.join(notebook_dir, "../escargot/benchmarking/notebooks")))

dataset_dir = os.path.abspath(os.path.join(notebook_dir, "../escargot/benchmarking/dataset"))
results_dir = os.path.abspath(os.path.join(notebook_dir, "../escargot/benchmarking/results"))

# from VectorEmbeddingSearchingNick_v1 import VectorEmbeddingSearching as vesNick
from VectorEmbeddingSearchingNick_v1 import VectorEmbeddingSearching as vesNick



max_tokens_weaviate = 10000

quantile_weaviate = 0.1

max_distance_weaviate = 0.25

show_distances_weaviate = True

model = "gpt-3.5-turbo-16k"

# temperature_azure_gpt = 0
temperature_azure_gpt = 0.7

max_tokens_azure_gpt = 1000


json_files =  ['MCQ_1hop.json', 'MCQ_2hop.json', 'OpenEnded_1hop.json', 'OpenEnded_2hop.json', 'True_or_False_1hop.json', 'True_or_False_2hop.json']
# json_files =  ['MCQ_1hop.json']
responses = {}
for json_file in json_files:
    # print(json_file)

    with open(dataset_dir+'/'+json_file) as f:
        data = json.load(f)
    
    if "True_or_False" in json_file:
        # Me
        # for multiple choice questions
        BASE_GENERATION_TEMPLATE = """
        {question} Please reply to the question with 'Answer: True' if the statement is correct, or 'Answer: False' if the statement is incorrect. Ensure you choose only one of these options.'
        """

    elif "MCQ" in json_file:
        # Me
        # for multiple choice questions
        BASE_GENERATION_TEMPLATE = """
        {question} Please respond in the format 'Answer:<number>'. For instance, if the correct answer to the question falls within choices 1 to 5 and the correct answer is 1, simply respond with 'Answer:1'
        """
    else:
        # Me
        # for open ended questions
        BASE_GENERATION_TEMPLATE = """{question}"""

    responses[json_file] = {}
    for question in data:
        response = ''
        formatted_question = BASE_GENERATION_TEMPLATE.format(question = question['question'])
        try:
            
            # Get the knowledge array
            knowledge_array,distances = vesNick.get_knowledge(formatted_question, max_tokens= max_tokens_weaviate, quantile = quantile_weaviate, max_distance=max_distance_weaviate, show_distances = show_distances_weaviate)
            # get the answer using the knowledge array
            response = vesNick.get_answer(model, formatted_question, knowledge_array, temperature = temperature_azure_gpt, max_tokens=max_tokens_azure_gpt)
            
            # The reason why i commented out the following is because of this kind of response (Answer:1)
            # For response= response[8:].strip(), the response is '', which is not what we want.
            # Remove "Answer:" from the response
            # if response.startswith("Answer:"):
            #     response= response[8:].strip()
            
            #remove ```cypher from the response
            response = response.replace("```cypher", "")

            #remove ``` from anywhere in the response
            response = response.replace("```", "")

            #remove \n from the response
            response = response.replace("\n", "")
            
            # print("Memgraph request:",response)
            print("question:",question['question'])
            print("request:",response)
        except Exception as e:
            # print("Memgraph request failed",e)
            response = ''
            print("error:", e)
        
        responses[json_file][question['question']] = str(response)
        # break
    # break
    # save the responses[json_file] after each json file as json with {json_file}_result name
    # result_file_path = os.path.join(results_dir, f"{json_file}_result.json")
    # with open(result_file_path, 'w') as result_file:
    #     json.dump(responses[json_file], result_file, indent=4)


# os.chdir("../")
dill.dump(responses, open(results_dir+'/'+'results/RAG_responses_mine.pkl', 'wb'))
    



