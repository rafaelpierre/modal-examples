import modal
from modal_examples.server.config import server_settings


BASE_IMAGE = server_settings.default_base_image
HF_XET_HIGH_PERFORMANCE = server_settings.hf_xet_high_performance
VLLM_LOG_STATS_INTERVAL = server_settings.vllm_log_stats_interval


vllm_image = (
    modal.Image.from_registry(BASE_IMAGE, add_python = "3.14")
    .entrypoint([])
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": HF_XET_HIGH_PERFORMANCE,
            "VLLM_LOG_STATS_INTERVAL": VLLM_LOG_STATS_INTERVAL
        }
    )
    .uv_pip_install("pydantic-settings>=2.15.0")
    .add_local_python_source("modal_examples")
)
