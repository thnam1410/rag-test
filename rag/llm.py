from langchain_aws import ChatBedrock
from app_env import AppEnv


class LLM:
    # Initialize the language model
    model = ChatBedrock(
        region=AppEnv.AWS_REGION,
        credentials_profile_name=AppEnv.AWS_PROFILE,
        model_id=AppEnv.LLM_MODEL
    )
