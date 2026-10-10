"""
TODO: replace with vLLM API
from vllm.entrypoints.cli.serve import ServeSubcommand
from vllm.entrypoints.launchers.cli_args import (
    make_arg_parser,
    validate_parsed_serve_args,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser


def serve_vllm(
    model_name: str,
    model_revision: str,
    speculative_model_name: str,
    speculative_model_revision: str,
    number_gpus: int,
    host: str = "0.0.0.0",
    vllm_port: int = 8000,
    uvicorn_log_level: str = "info",
    async_scheduling: bool = True,
    fast_boot: bool = True,
    reasoning_parser: str = "gemma4",
    tool_call_parser: str = "gemma4",
    num_speculative_tokens: int = 4,
):
    # Obtain all the server's default configuration values.
    parser = make_arg_parser(FlexibleArgumentParser())
    args = parser.parse_args([model_name])

    # Model
    args.revision = model_revision
    args.served_model_name = [model_name, "llm"]

    # HTTP server
    args.host = host
    args.port = vllm_port
    args.uvicorn_log_level = uvicorn_log_level

    # Engine configuration
    args.async_scheduling = async_scheduling
    args.enforce_eager = fast_boot
    args.tensor_parallel_size = number_gpus

    # Text only
    args.limit_mm_per_prompt = {
        "image": 0,
        "video": 0,
        "audio": 0,
    }

    # Reasoning and tool calling
    args.enable_auto_tool_choice = True
    args.reasoning_parser = reasoning_parser
    args.tool_call_parser = tool_call_parser

    # Speculative decoding
    args.speculative_config = {
        "model": speculative_model_name,
        "revision": speculative_model_revision,
        "num_speculative_tokens": num_speculative_tokens,
    }

    validate_parsed_serve_args(args)

    # Start the actual OpenAI-compatible HTTP server.
    # Blocks until the server shuts down.
    ServeSubcommand.cmd(args)
"""

import json
import textwrap


def get_vllm_command(
    model_name: str,
    model_revision: str,
    speculative_model_name: str,
    speculative_model_revision: str,
    number_gpus: int,
    host: str = "0.0.0.0",
    vllm_port: int = 8000,
    uvicorn_log_level: str = "info",
    async_scheduling: bool = True,
    fast_boot: bool = True,
    reasoning_parser: str = "gemma4",
    tool_call_parser: str = "gemma4",
    num_speculative_tokens: int = 4
):

    base_command = textwrap.dedent(f"""
        vllm
        serve
        {model_name}
        --revision
        {model_revision}
        --served-model-name
        {model_name}
        llm
        --host
        {host}
        --port
        {vllm_port}
        --uvicorn-log-level={uvicorn_log_level}
    """)

    if async_scheduling:
        base_command += "\n--async_scheduling"

    # Split command string into list of args
    # prior to adding extra params

    command_args = base_command.split()

    # enforce-eager disables both Torch compilation and CUDA graph capture
    # default is no-enforce-eager. see the --compilation-config flag for tighter control
    command_args += ["--enforce-eager" if fast_boot else "--no-enforce-eager"]

    # assume multiple GPUs are for splitting up large matrix multiplications
    command_args += ["--tensor-parallel-size", str(number_gpus)]

    # add model-specific configuration
    command_args += [
        # skip multimedia support, just language
        "--limit-mm-per-prompt",
        json.dumps({"image": 0, "video": 0, "audio": 0}),
        # enable reasoning and tool use
        "--enable-auto-tool-choice",
        "--reasoning-parser",
        reasoning_parser,
        "--tool-call-parser",
        tool_call_parser,
    ]

    # add speculative decoding
    command_args += [
        "--speculative-config",
        json.dumps(
            {
                "model": speculative_model_name,
                "revision": speculative_model_revision,
                "num_speculative_tokens": num_speculative_tokens,
            }
        ),
    ]

    return command_args
