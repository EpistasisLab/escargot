import openai

def get_embedding(text_to_embed, config_RAG_TEST):

    # Azure 
    openai.api_type ="azure"
    openai.api_key = config_RAG_TEST["azuregpt35-16k"]["api_key"]

    openai.api_version = "2023-05-15"

    openai.azure_endpoint = "https://caire-azure-openai.openai.azure.com/"


    embedding_id = config_RAG_TEST["azuregpt35-16k"]["embedding_id"]

    response = openai.embeddings.create(
        model=embedding_id,
        input=text_to_embed
    )

    embedding = response.data[0].embedding

    return embedding