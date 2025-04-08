from azure.identity import get_bearer_token_provider, EnvironmentCredential
from openai import AzureOpenAI
from dotenv import load_dotenv
import os

load_dotenv(override=True)

def main():
    env_token_provider = get_bearer_token_provider(
        EnvironmentCredential(), "https://cognitiveservices.azure.com/.default"
    )

    aoai_client_demo = AzureOpenAI(
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_ad_token_provider=env_token_provider,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )

    gpt_response = aoai_client_demo.chat.completions.create(
        model = os.getenv("AZURE_OPENAI_MODEL_GPT4o"),
        messages = [
            {"role": "system", "content": "You are a friendly chatbot"},
            {"role": "user", "content": "Choose a random animal and describe it to me in 1 sentence."}
        ]
    )

    print(gpt_response.model_dump_json(indent=2))

    ada_response = aoai_client_demo.embeddings.create(
        model = os.getenv("AZURE_OPENAI_MODEL_ADA2"),
        input = "Demo text for embeddings"
    )

    print(ada_response.model_dump_json(indent=2))

if __name__ == "__main__":
    main()
