from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os


load_dotenv()

class Settings(BaseSettings):
    BINANCE_API_KEY: str = ""
    BINANCE_API_SECRET: str = ""
    BINANCE_TESTNET: bool = True

    GROQ_API_KEY: str
    NEWSDATA_API_KEY: str
    PROJECT_NAME: str = "Optimizer"

    class Config:
        env_file =".env"

settings = Settings()