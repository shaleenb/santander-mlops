from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "Santander Customer Transaction Prediction"

    MODEL_PATH: str = "models/model.joblib"

    DEFAULT_ID_COLUMN: str = "ID_code"

    class config:
        case_sensitive = True


settings = Settings()
