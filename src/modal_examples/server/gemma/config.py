from pydantic_settings import BaseSettings


class GemmaSettings(BaseSettings):

    # Model settings

    model_name: str = "google/gemma-4-26B-A4B-it"
    model_revision: str = "47b6801b24d15ff9bcd8c96dfaea0be9ed3a0301"

    speculative_model_name: str = "google/gemma-4-26B-A4B-it-assistant"
    speculative_model_revision: str = "f188f476dc11dd5bb3014dc861529d316bce49d3"


gemma_settings = GemmaSettings()
