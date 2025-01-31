from langchain_aws import BedrockEmbeddings
from app_env import AppEnv


class Embedding:
    # Define the embedding model
    model = BedrockEmbeddings(
        credentials_profile_name=AppEnv.AWS_PROFILE,
        region_name=AppEnv.AWS_REGION,
        model_id=AppEnv.EMBEDDING_MODEL
    )
