import os
import openai
from dotenv import load_dotenv


# Load the .env file
load_dotenv()


openai.api_type ="azure"
# openai.api_key = os.environ["AZURE_OPENAI_KEY"]
openai.api_key = os.getenv('AZURE_OPENAI_KEY')
openai.api_version = "2023-05-15"
openai.azure_endpoint = "https://caire-azure-openai.openai.azure.com/"


def get_embedding(text_to_embed):
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=text_to_embed
    )

    embedding = response.data[0].embedding

    return embedding