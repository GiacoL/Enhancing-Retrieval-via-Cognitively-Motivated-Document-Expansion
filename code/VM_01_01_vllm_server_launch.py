import subprocess
#import torch

from VM_00_01_folders_and_global_variables import project_dir,intermediate_results_dir,results_dir,prompts_dir,info_g_variables

#torch.cuda.empty_cache()

g_llm_value = info_g_variables['g_llm']
command = f"python -m vllm.entrypoints.openai.api_server --model {g_llm_value} --disable-log-requests &"

#command = "python -m vllm.entrypoints.openai.api_server --model HuggingFaceH4/zephyr-7b-beta --disable-log-requests --dtype float16 &"
#command = "python -m vllm.entrypoints.openai.api_server --model TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ --disable-log-requests --dtype float16 --enforce-eager &"
#command = "python -m vllm.entrypoints.openai.api_server --model SweatyCrayfish/llama-3-8b-quantized --disable-log-requests &"

#command = "python -m vllm.entrypoints.openai.api_server --model MaziyarPanahi/Meta-Llama-3-8B-Instruct-GPTQ --disable-log-requests --quantization gptq --dtype float16 --enforce-eager &"

#command = "python -m vllm.entrypoints.openai.api_server --model TheBloke/zephyr-7B-beta-GPTQ --disable-log-requests --quantization gptq --enforce-eager &"

#command = "python -m vllm.entrypoints.openai.api_server --model microsoft/Phi-4 --disable-log-requests --quantization gptq --enforce-eager --dtype float16 &"

#command = "python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 --disable-log-requests --quantization gptq --enforce-eager &"

# Execute the shell command
subprocess.run(command, shell=True)


