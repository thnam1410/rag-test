from typing import Optional
from dotenv import dotenv_values

# Load environment variables from .env file
config = dotenv_values()


class AppEnv:
    """Static class to expose environment variables with type hints"""

    # AWS Related
    AWS_PROFILE: str = config.get('AWS_PROFILE', 'default')
    AWS_REGION: str = config.get('AWS_REGION', 'ap-southeast-2')

    # Redis
    REDIS_URL: str = config.get('REDIS_URL', 'redis://localhost:6379')

    # Vector DB Related
    PERSIST_DIRECTORY: str = config.get('PERSIST_DIRECTORY', 'db/my_db')

    # LLM Related
    LLM_MODEL: str = config.get('LLM_MODEL')

    # Embedding Related
    EMBEDDING_MODEL: str = config.get('EMBEDDING_MODEL')

    @classmethod
    def is_development(cls) -> bool:
        """Check if running in development environment"""
        return cls.APP_ENV.lower() == 'development'
