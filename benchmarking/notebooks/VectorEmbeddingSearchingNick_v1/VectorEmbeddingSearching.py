
import weaviate
# from embed import get_embedding
from VectorEmbeddingSearchingNick_v1.embed import get_embedding
import os
from openai import AzureOpenAI
import numpy as np
import tiktoken

weaviate_client = None


def get_token_length(text, model="gpt-3.5-turbo"):
    tokenizer = tiktoken.encoding_for_model(model)
    tokens = tokenizer.encode(text)
    return len(tokens)

def calculate_message_length( message_text, model="gpt-3.5-turbo"):
        
        
        total_length = sum(get_token_length(msg['content'], model) for msg in message_text)
        return total_length

# def get_client(url="http://54.203.49.88"):
def get_client(url="http://100.21.60.48", weaviate_api_key = None):   
    auth_config = weaviate.AuthApiKey(api_key=weaviate_api_key)
    global weaviate_client
    if weaviate_client is None:
        weaviate_client = weaviate.Client(
            url=url,
            auth_client_secret=auth_config
        )
    return weaviate_client


def query_bm25(class_name="AlzKB", properties=["knowledge"], query_string="test", additional="score", limit=3):
    return ((get_client().query
        .get(class_name, properties))
        .with_bm25(
            query=query_string,
            properties=properties  # this does not need to be the same as columns
        )
        .with_additional(additional)
        .with_limit(limit)
        .do()
    )


def query_near_text(class_name="AlzKB", properties=["knowledge"], near_text=["gene"], additional="score", limit=3):
    return ((get_client().query
        .get(class_name, properties))
        .with_near_text({
            "concepts": near_text
        })
        .with_limit(limit)
        .with_additional(additional)
        .do()
    )


def query_my_near_text(class_name="AlzKB", properties=["knowledge"], prompt="genes", additional="score", limit=3, config_RAG_TEST = None):
    vector = {
        "vector": get_embedding(prompt, config_RAG_TEST)
    }
    # print(vector["vector"])
    return query_near_vector(class_name, properties, near_vector=vector, additional=additional, limit=limit, config_RAG_TEST=config_RAG_TEST)

def query_near_vector(class_name="AlzKB", properties=["knowledge"], near_vector={}, additional="score", limit=3, config_RAG_TEST = None):
    WEAVIATE_APIKEY = config_RAG_TEST["weaviate"]["api_key"]
    # weaviate url
    weaviate_url = config_RAG_TEST["weaviate"]["url"]
    return ((get_client(url=weaviate_url ,weaviate_api_key = WEAVIATE_APIKEY).query
        .get(class_name, properties))
        .with_near_vector(near_vector)
        .with_limit(limit)
        .with_additional(additional)
        .do()
    )
    
def object_count(class_name="AlzKB"):
    return ((get_client().query
        .aggregate(class_name)
        .with_meta_count()
        .do()))
    
def get_answer(question, knowledge_array, config_RAG_TEST = None):
    

    # Azure 
    AZURE_OPENAI_KEY = config_RAG_TEST["azuregpt35-16k"]["api_key"]

    
    model = config_RAG_TEST["azuregpt35-16k"]["model_id"]
    print("model: ", model)

    # temperature_azure_gpt = 0
    # temperature_azure_gpt = 0.7
    temperature = config_RAG_TEST["azuregpt35-16k"]["temperature"]

    # max_tokens_azure_gpt = 1000
    max_tokens = config_RAG_TEST["azuregpt35-16k"]["max_tokens"]

    # api_version
    api_version = config_RAG_TEST["azuregpt35-16k"]["api_version"]


    # AZURE_OPENAI_ENDPOINT=https://caire-azure-openai.openai.azure.com/openai/deployments/gpt-35-turbo-16k/chat/completions?api-version=2023-07-01-preview

    AZURE_OPENAI_ENDPOINT = config_RAG_TEST["azuregpt35-16k"]["api_base"]+"api-version="+api_version


    # if model is gpt-35-turbo-16k then max
    if model == "gpt-35-turbo-16k":
        max_context_length = 16384
    

    in_context = "\n".join(knowledge_array)

    # print length of in_context
    print("length of in_context: ", len(in_context))
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_KEY,
        api_version=api_version,
        azure_endpoint = AZURE_OPENAI_ENDPOINT
        )
        

    
    message_text = [{"role":"system","content":"""You are an Alzheimer's data specialist AI assistant dedicated to providing information and support related to Alzheimer's disease.
        Your primary goal is to assist users by offering factual and relevant information based on your access to a comprehensive knowledge graph associated with Alzheimer's. 
        Your responses are focused on addressing queries related to Alzheimer's, and you do not provide information unrelated to the topic. 
        You will also only answer based on the knowledge within the knowledge graph. 
        You will notice there will be gene symbols in the knowledge, and there are subtle differences between the gene names.
        You will need to be careful that the names are exact with you use them in context. There may be single differences in numbers and letters.
        For example, the gene "APOE" is not the same as gene "APOE1". Another example is the gene "IQCK" is not the same as gene "IQCG".
        You will need to be careful of specific biological terms. For example, the term "amino" is different from the term "amine".
        If you are providing a list, be sure not to list duplicates. 
        Your demeanor is empathetic and concise as you aim to help users understand and navigate Alzheimer's-related concerns."""},
        {"role":"user","content":"Here is the knowledge from the knowledge graph: "+in_context+"\nThe question is: "+question}]
    

    # initial message length check
    message_length = calculate_message_length(message_text)
    print("Initial token length of message_text: ", message_length)

    # Reduce in_context size until it fits within max_context_length
    while message_length + max_tokens > max_context_length:
        # Reduce in_context size
        knowledge_array = knowledge_array[:-1]
        in_context = "\n".join(knowledge_array)


        message_text = [{"role":"system","content":"""You are an Alzheimer's data specialist AI assistant dedicated to providing information and support related to Alzheimer's disease.
        Your primary goal is to assist users by offering factual and relevant information based on your access to a comprehensive knowledge graph associated with Alzheimer's. 
        Your responses are focused on addressing queries related to Alzheimer's, and you do not provide information unrelated to the topic. 
        You will also only answer based on the knowledge within the knowledge graph. 
        You will notice there will be gene symbols in the knowledge, and there are subtle differences between the gene names.
        You will need to be careful that the names are exact with you use them in context. There may be single differences in numbers and letters.
        For example, the gene "APOE" is not the same as gene "APOE1". Another example is the gene "IQCK" is not the same as gene "IQCG".
        You will need to be careful of specific biological terms. For example, the term "amino" is different from the term "amine".
        If you are providing a list, be sure not to list duplicates. 
        Your demeanor is empathetic and concise as you aim to help users understand and navigate Alzheimer's-related concerns."""},
        {"role":"user","content":"Here is the knowledge from the knowledge graph: "+in_context+"\nThe question is: "+question}]
        

        message_length = calculate_message_length(message_text)
        print("Reduced token length of message_text: ", message_length)
    
    # text that may help:
    # When you reply, please provide a followup response that includes the exact knowledge from the knowledge graph that you used to generate your response, and please do not include the knowledge that are not used.
        
    response = client.chat.completions.create(
        model=model, # model = "deployment_name".
        max_tokens=max_tokens,
        messages=message_text,
        temperature=temperature
    )
    # response = client.completions.create(model=deployment_name, prompt=message_text, max_tokens=100)
    # print(response)
    # print(response.choices[0].message.content)
    return response.choices[0].message.content


def get_answer_without_ICL_Azure_gpt_35_turbo_16k(question, config_RAG_TEST = None):
    temperature = config_RAG_TEST["azuregpt35-16k"]["temperature"]
    max_tokens = config_RAG_TEST["azuregpt35-16k"]["max_tokens"]

    # in_context = "\n".join(knowledge_array)
    client = AzureOpenAI(
        api_key=config_RAG_TEST["azuregpt35-16k"]["api_key"],
        api_version=config_RAG_TEST["azuregpt35-16k"]["api_version"],
        azure_endpoint = config_RAG_TEST["azuregpt35-16k"]["api_base"]+"api-version="+config_RAG_TEST["azuregpt35-16k"]["api_version"]
        )
        
    deployment_name=config_RAG_TEST["azuregpt35-16k"]["model_id"] #This will correspond to the custom name you chose for your deployment when you deployed a model.
    
    
    message_text = [{"role":"system","content":"""You are an Alzheimer's data specialist AI assistant dedicated to providing information and support related to Alzheimer's disease.
        Your primary goal is to assist users by offering factual and relevant information based on your access to a comprehensive knowledge graph associated with Alzheimer's. 
        Your responses are focused on addressing queries related to Alzheimer's, and you do not provide information unrelated to the topic. 
        You will also only answer based on the knowledge within the knowledge graph. 
        You will notice there will be gene symbols in the knowledge, and there are subtle differences between the gene names.
        You will need to be careful that the names are exact with you use them in context. There may be single differences in numbers and letters.
        For example, the gene "APOE" is not the same as gene "APOE1". Another example is the gene "IQCK" is not the same as gene "IQCG".
        You will need to be careful of specific biological terms. For example, the term "amino" is different from the term "amine".
        If you are providing a list, be sure not to list duplicates. 
        Your demeanor is empathetic and concise as you aim to help users understand and navigate Alzheimer's-related concerns."""},
                    # {"role":"user","content":"Here is the knowledge from the knowledge graph: "+in_context+"\nThe question is: "+question}]
                    {"role":"user","content":"The question is: "+question}]
    
    # text that may help:
    # When you reply, please provide a followup response that includes the exact knowledge from the knowledge graph that you used to generate your response, and please do not include the knowledge that are not used.
        
    response = client.chat.completions.create(
        model=deployment_name, # model = "deployment_name".
        max_tokens=max_tokens,
        messages=message_text,
        temperature=temperature



    )
    # response = client.completions.create(model=deployment_name, prompt=message_text, max_tokens=100)
    # print(response)
    # print(response.choices[0].message.content)
    return response.choices[0].message.content

def get_answer_without_ICL_Azure_gpt_4(question,temperature = 0, max_tokens=1000):
    # in_context = "\n".join(knowledge_array)

    # AZURE_OPENAI_KEY_GPT_4=035a7cc7324d4e01a3f71d25dfb2165e
    # AZURE_OPENAI_ENDPOINT_GPT_4=https://caire-gpt4.openai.azure.com/
    # api-version_GP4=2024-02-15-preview

    client = AzureOpenAI(

        # api_key=os.environ["AZURE_OPENAI_KEY_GPT_4"],
        # api_version=os.environ["api-version_GP4"],
        # azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT_GPT_4"]
        api_key="035a7cc7324d4e01a3f71d25dfb2165e",
        api_version="2024-02-15-preview",
        azure_endpoint = "https://caire-gpt4.openai.azure.com/"
        )
        
    deployment_name='gpt-4' #This will correspond to the custom name you chose for your deployment when you deployed a model.
    
    
    message_text = [{"role":"system","content":"""You are an Alzheimer's data specialist AI assistant dedicated to providing information and support related to Alzheimer's disease.
        Your primary goal is to assist users by offering factual and relevant information based on your access to a comprehensive knowledge graph associated with Alzheimer's. 
        Your responses are focused on addressing queries related to Alzheimer's, and you do not provide information unrelated to the topic. 
        You will also only answer based on the knowledge within the knowledge graph. 
        You will notice there will be gene symbols in the knowledge, and there are subtle differences between the gene names.
        You will need to be careful that the names are exact with you use them in context. There may be single differences in numbers and letters.
        For example, the gene "APOE" is not the same as gene "APOE1". Another example is the gene "IQCK" is not the same as gene "IQCG".
        You will need to be careful of specific biological terms. For example, the term "amino" is different from the term "amine".
        If you are providing a list, be sure not to list duplicates. 
        Your demeanor is empathetic and concise as you aim to help users understand and navigate Alzheimer's-related concerns."""},
                    # {"role":"user","content":"Here is the knowledge from the knowledge graph: "+in_context+"\nThe question is: "+question}]
                    {"role":"user","content":"The question is: "+question}]
    
    # text that may help:
    # When you reply, please provide a followup response that includes the exact knowledge from the knowledge graph that you used to generate your response, and please do not include the knowledge that are not used.
        
    response = client.chat.completions.create(
        model=deployment_name, # model = "deployment_name".
        max_tokens=max_tokens,
        messages=message_text,
        temperature=temperature
    )
    # response = client.completions.create(model=deployment_name, prompt=message_text, max_tokens=100)
    # print(response)
    # print(response.choices[0].message.content)
    return response.choices[0].message.content


# bm25 (keyword) search
# result = query_bm25(query_string="list all genes connected to alzheimers", limit=20)
# result = query_bm25(query_string="TPSAB1", additional="vector")
# result = query_bm25(query_string="TPSAB1", additional="id")
# result = query_bm25(query_string="TPSAB1", additional="distance") # returns: None

# # Similarity / Vector search
# near_vector = {
#     # what genes code for acetyl-CoA carboxylase?
# }
# result = query_near_vector(near_vector=near_vector, additional="distance", limit=10)

# My nearText search
# print(os.environ)

def condense_knowledge(knowledge_array):
    in_context = "\n".join(knowledge_array)
    client = AzureOpenAI(
        api_key=os.environ["OPENAI_KEY"],
        api_version="2023-07-01-preview",
        azure_endpoint = "https://caire-azure-openai.openai.azure.com/openai/deployments/gpt-35-turbo-16k/chat/completions?api-version=2023-07-01-preview"
        )
        
    deployment_name='gpt-35-turbo-16k' #This will correspond to the custom name you chose for your deployment when you deployed a model. 
    
    message_text = [{"role":"system","content":"""You are a data specialist AI assistant dedicated to providing a more concise version of the knowledge graph associated with Alzheimer's disease.
                     You will receive a fragmented list of knowledge from the knowledge graph, and you will need to condense the knowledge into a more concise version.
                     Be sure to include all the knowledge from the knowledge graph, and do not include knowledge that are not in the knowledge graph.
                     You will notice there will be gene symbols in the knowledge, and there are subtle differences between the gene names.
                     You will need to be careful that the names are exact with you use them in context. There may be single differences in numbers and letters.
                     For example, the gene "APOE" is not the same as gene "APOE1". Another example is the gene "IQCK" is not the same as gene "IQCG".
                     You will need to be careful of specific biological terms. For example, the term "amino" is different from the term "amine".
                     If you are providing a list, be sure not to list duplicates."""},
                    {"role":"user","content":"Here is the knowledge from the knowledge graph: "+in_context}]
    
    # text that may help:
    # When you reply, please provide a followup response that includes the exact knowledge from the knowledge graph that you used to generate your response, and please do not include the knowledge that are not used.
        
    response = client.chat.completions.create(
        model=deployment_name, # model = "deployment_name".
        max_tokens=1000,
        messages=message_text
    )
    return response.choices[0].message.content

def compare_answers(answer, ground_truth):
    client = AzureOpenAI(
        api_key=os.environ["OPENAI_KEY"],
        api_version="2023-07-01-preview",
        azure_endpoint = "https://caire-azure-openai.openai.azure.com/openai/deployments/gpt-35-turbo-16k/chat/completions?api-version=2023-07-01-preview"
        )
        
    deployment_name='gpt-35-turbo-16k' #This will correspond to the custom name you chose for your deployment when you deployed a model. 
    
    message_text = [{"role":"system","content":"""You are a data specialist AI that is comparing an answer to a ground truth.
                     You will receive an answer and a ground truth, and you will need to compare the answer to the ground truth.
                     In the end, you will need to provide a score between 0 and 1, where 0 is the worst score and 1 is the best score.
                     You will need to be careful that the names are exact with you use them in context. There may be single differences in numbers and letters.
                     For example, the gene "APOE" is not the same as gene "APOE1". Another example is the gene "IQCK" is not the same as gene "IQCG".
                     You will need to be careful of specific biological terms. For example, the term "amino" is different from the term "amine".
                     If you are comparing lists, be very sure that the lists are exactly the same and understand that it may not be in the same order."""},
                    {"role":"user","content":"Here is the answer: "+answer+"\nHere is the ground truth: "+ground_truth}]
    
    # text that may help:
    # When you reply, please provide a followup response that includes the exact knowledge from the knowledge graph that you used to generate your response, and please do not include the knowledge that are not used.
        
    response = client.chat.completions.create(
        model=deployment_name, # model = "deployment_name".
        max_tokens=1000,
        messages=message_text
    )
    return response.choices[0].message.content

def get_score_from_comparison(comparison):
    client = AzureOpenAI(
        api_key=os.environ["OPENAI_KEY"],
        api_version="2023-07-01-preview",
        azure_endpoint = "https://caire-azure-openai.openai.azure.com/openai/deployments/gpt-35-turbo-16k/chat/completions?api-version=2023-07-01-preview"
        )
        
    deployment_name='gpt-35-turbo-16k' #This will correspond to the custom name you chose for your deployment when you deployed a model. 
    
    message_text = [{"role":"system","content":"""You are a data specialist AI that is given a description of comparing an answer to the truth and possibly a numeric score.
                     You will extract the score from the description and output only the score.
                     Your output should not include any other text. For example, if the description is "The score is 0.5", you should output "0.5".
                     If there is no score, you will output nothing.
                     If there is no score, but determine that the description says it is a good answer, you will output "1"."""},
                    {"role":"user","content":"Here is the description: "+comparison}]
    
    # text that may help:
    # When you reply, please provide a followup response that includes the exact knowledge from the knowledge graph that you used to generate your response, and please do not include the knowledge that are not used.
        
    response = client.chat.completions.create(
        model=deployment_name, # model = "deployment_name".
        max_tokens=1000,
        messages=message_text
    )
    return response.choices[0].message.content

def get_knowledge(question, config_RAG_TEST = None):

    # weaviate
    max_tokens = config_RAG_TEST["weaviate"]["max_tokens_weaviate"]
    quantile = config_RAG_TEST["weaviate"]["quantile_weaviate"]
    max_distance = config_RAG_TEST["weaviate"]["max_distance_weaviate"]
    

    

    # print question, max_tokens, quantile, max_distance, show_distances using print function
    # print("question: ", question)
    # print("max_tokens: ", max_tokens)
    # print("quantile: ", quantile)
    # print("max_distance: ", max_distance)
    # print("show_distances: ", show_distances)

    knowledge_array = query_my_near_text(prompt=question, additional=["distance"], limit=2000,config_RAG_TEST=config_RAG_TEST)

    knowledge_array = knowledge_array["data"]["Get"]["AlzKB"]
    
    all_distances = [x['_additional']["distance"] for x in knowledge_array]
    quant_dist = np.quantile(all_distances, quantile)
    in_context = []
    distances = []
    cur_tokens = 0
    for knowledge in knowledge_array:
        # print(knowledge['_additional']["distance"])
        # print(knowledge["knowledge"])
        if knowledge['_additional']["distance"] > quant_dist or knowledge['_additional']["distance"] > max_distance:
            break
        
        cur_tokens += len(knowledge["knowledge"].split(" "))
        if cur_tokens > max_tokens:
            break
        in_context.append(knowledge["knowledge"])
        distances.append(knowledge['_additional']["distance"])
        
    return in_context, distances

# import keys



"""

if __name__ == '__main__':
    # bm25 (keyword) search
    # result = query_bm25(query_string="list all genes connected to alzheimers", limit=20)
    # result = query_bm25(query_string="TPSAB1", additional="vector")
    # result = query_bm25(query_string="TPSAB1", additional="id")
    # result = query_bm25(query_string="TPSAB1", additional="distance") # returns: None

    # # Similarity / Vector search
    # near_vector = {
    #     # what genes code for acetyl-CoA carboxylase?
    # }
    # result = query_near_vector(near_vector=near_vector, additional="distance", limit=10)

    # My nearText search
    # print(os.environ)


    question = "what drugs not associated with alzheimer's disease may be effective in treating alzheimer's disease and why?"

    # quantile = 0.25 # 500
    # quantile = 0.15 # 300
    # quantile = 0.1 # 200
    # quantile = 0.05 # 100
    # quantile = 0.01 # 20
    quantile = 0.005 # 10

    knowledge_array,distances = get_knowledge(question, max_tokens=10000, quantile = quantile, max_distance=0.25, show_distances = True)
    print("knowledge array length: ", len(knowledge_array))
    # knowledge = condense_knowledge(knowledge_array)
    answer = get_answer(question,knowledge_array,temperature = 0)
    print(answer)

"""
