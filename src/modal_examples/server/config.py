from pydantic_settings import BaseSettings

MINUTES = 60 # Seconds

class ServerSettings(BaseSettings):

    # Image settings

    default_base_image: str = "vllm/vllm-openai:v0.31.0"
    hf_xet_high_performance: str = "1"
    vllm_log_stats_interval: str = "1"

    # Server settings
    # General

    app_name: str = "example-vllm-inference"

    # GPU

    gpu_sku: str = "H200"
    number_gpus: int = 1

    # Volumes

    hf_volume_name: str = "huggingface-cache"
    vllm_volume_name: str = "vllm-cache"
    hf_cache_path: str = "/root/.cache/huggingface"
    vllm_cache_path: str = "/root/.cache/vllm"

    # vLLM

    vllm_port: int = 8000

    # Misc

    routing_region: str = "us-east"
    scaledown_window: int = 5 * MINUTES
    startup_timeout: int = 10 * MINUTES
    target_concurrency: int = 20


server_settings = ServerSettings()
