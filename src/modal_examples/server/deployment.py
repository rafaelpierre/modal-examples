import modal

from modal_examples.server.config import server_settings
from modal_examples.server.gemma.config import gemma_settings
from modal_examples.server.images.vllm import vllm_image
from modal_examples.server.utils import get_vllm_command
from modal_examples.server.volumes import get_volumes

app = modal.App(server_settings.app_name)

GPU_SKU = server_settings.gpu_sku
N_GPU = server_settings.number_gpus


@app.server(
    image=vllm_image,
    gpu=f"{GPU_SKU}:{N_GPU}",
    scaledown_window=server_settings.scaledown_window,
    startup_timeout=server_settings.startup_timeout,
    volumes=get_volumes(),
    routing_region=server_settings.routing_region,
    target_concurrency=server_settings.target_concurrency
)
class Server:
    @modal.enter()
    def start(self):

        import subprocess

        vllm_command = get_vllm_command(
            model_name = gemma_settings.model_name,
            model_revision = gemma_settings.model_revision,
            speculative_model_name = gemma_settings.speculative_model_name,
            speculative_model_revision = gemma_settings.speculative_model_revision,
            number_gpus = N_GPU
        )

        self.process = subprocess.Popen(vllm_command)

    @modal.exit()
    def stop(self):
        self.process.terminate()
