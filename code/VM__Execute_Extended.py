import subprocess
import time
import os
from itertools import product

llms_names = {
    #'qwen2-5_7b_instruct': 'Qwen/Qwen2.5-7B-Instruct',
    #'Qwen2.5-14B': 'Qwen/Qwen2.5-14B'
    #'Qwen2.5-14B-Instruct': 'Qwen/Qwen2.5-14B-Instruct',
    #'Qwen2.5-3B-Instruct': 'Qwen/Qwen2.5-3B-Instruct',
    #'Llama-3.1-8B-Instruct': 'meta-llama/Llama-3.1-8B-Instruct',
    'Llama-3.2-3B-Instruct': 'meta-llama/Llama-3.2-3B-Instruct'

}

#embeddings_names = ['SFR-Mistral','gte-Qwen2-1.5']

#embeddings_names = ['E5','gte-Qwen2-1.5']

embeddings_names = ['infly']

#dataset_names = ['scifactnano','fiqanano','nfcorpusnano','nqnano','msmarconano','climatefevernano',
 #                'dbpedianano','fevernano','hotpotqanano','quoraretrievalnano','scidocsnano','arguananano']

# List of datasets
dataset_names = ['scifactnano']

# List of scripts to execute
scripts_to_execute = ["VM_01_02_script_generate_text_Beir_v4.py"]

#scripts_to_execute = ["VM_02_01_Embeddings_v3.py", "VM_02_02_Embeddings_queries_v2.py", "VM_03_01_Similarity_v3_parallel_batch_only_test_split.py","VM_ranx.py"]

#scripts_to_execute = ["VM_02_01_Embeddings_v3.py"]
#scripts_to_execute = ["VM_02_02_Embeddings_queries_v2.py"]
#scripts_to_execute = ["VM_03_01_Similarity_v3_parallel_batch_only_test_split.py"]
#scripts_to_execute = ["VM_ranx.py"]

# Loop through all combinations of LLMs, embeddings, and datasets
for llm, emb, ds in product(llms_names, embeddings_names, dataset_names):
    env = os.environ.copy()
    env["llm_to_use_env"] = llm
    env["llm_to_use_url_env"] = llms_names[llm]
    env["emb_to_use_env"] = emb
    env["dataset_to_use_env"] = ds

    # Append the current combination to the dataset log
    with open("current_dataset.txt", "a") as dataset_file:
        dataset_file.write(f"{llm} | {emb} | {ds}\n")

    # Run all scripts with the current environment
    for script in scripts_to_execute:
        try:
            subprocess.run(["python", script], check=True, env=env)
            print(f"✅ {llm} - {emb} - {ds} - Script {script} completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ {llm} - {emb} - {ds} - Error executing {script}: {e}")
