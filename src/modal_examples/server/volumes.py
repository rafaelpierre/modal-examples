import modal
from modal_examples.server.config import server_settings


HF_VOLUME_NAME = server_settings.hf_volume_name
HF_CACHE_PATH = server_settings.hf_cache_path
VLLM_VOLUME_NAME = server_settings.vllm_volume_name
VLLM_CACHE_PATH = server_settings.vllm_cache_path


def get_volume(volume_name: str, create_if_missing: bool = True):

    return modal.Volume.from_name(
        name=volume_name,
        create_if_missing=create_if_missing
    )


def get_volumes() -> dict:

    volumes = {}

    for cache_path, volume_name in zip (
        [HF_CACHE_PATH, VLLM_CACHE_PATH],
        [HF_VOLUME_NAME, VLLM_VOLUME_NAME]
    ):
        volume = get_volume(volume_name)
        volumes[cache_path] = volume

    return volumes
