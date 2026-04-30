# !!!!!!!!!!!!Check Superset, Gen, emb, sim and folder name!!!!!!!

import sys
import os

# import the name of the current Nano BEIR dataset to use


#from VM__Execute_Extended import dataset_to_use_env

# Retrieve dataset name from environment variable
dataset_to_use_env = os.getenv("dataset_to_use_env")

# create a dictionary with all relevant information to be used in the scripts
#for text generation, embeddings, similarity and ranx

info_g_variables= {
  'SuperSet':f'qwen2-5_7b_instruct_t_08_{dataset_to_use_env}',

  #Gen text
  'g_exp_gen':f'qwen2-5_7b_instruct_t_08_{dataset_to_use_env}', #$ name of the generated text experiment
  'g_llm':'Qwen/Qwen2.5-7B-Instruct',
  'g_beir_dataset':dataset_to_use_env,
  'g_prompts_to_use_script':'prompts_llama3_quant_reduced.py',
  #Define a step_size to execute the function in batches
  'g_step_size_gen':1000,
  'vllm_processes':200,
  'temperature': 0.8,

  #Embeddings
  'g_exp_emb':f'emb_qwen2-5_7b_instruct_t_08_E5_{dataset_to_use_env}', #$
  'g_embedding_model':'E5', #$
  'g_step_size_embeddings':1000,
  'g_batch_embeddings':500,

  #Similarity
  'g_exp_sim':f'sim_qwen2-5_7b_instruct_E5_t_08_{dataset_to_use_env}', #$ name infofile - similarity 
  'g_step_size_sim':1000,
  'g_batch_sim':1000,
  
  #Ranx
  'g_keys_rank_to_merge' : [
    ('EN2','EN4'),
    ('EL3','EL5'),
    ('EX1', 'EX2'),
    ('EN2','EN4','EL3','EL5'),
    ('EN2','EN4','EX1','EX2'),
    ('EN2','EN4','Original'),
    ('EX1','EX2','EL3','EL5'),
    ('EL3','EL5','Original'),
    ('EX1','EX2','Original'),
    ('EN2','EN4','EX1','EX2','EL3','EL5'),
    ('EN2','EN4','EL3','EL5','Original'),
    ('EN2','EN4','EX1','EX2','Original'),
    ('EX1','EX2','EL3','EL5','Original'),
    ('EN2','EN4','EX1','EX2','EL3','EL5','Original')
  ],
  'g_metrics': ['hits@', 'hit_rate@', 'precision@', 'recall@', 'f1@', 'r-precision', 'bpref', 'rbp.95', 'mrr@', 'dcg@', 'dcg_burges@', 'ndcg_burges@'],
  'g_k': (1, 5, 10, 20, 50,100),
  'g_fusion_methods_unsupervised': [
    "min",  # Alias for CombMIN
    "max",  # Alias for CombMAX
    "med",  # Alias for CombMED
    "sum",  # Alias for CombSUM
    "anz",  # Alias for CombANZ
    "mnz",  # Alias for CombMNZ
    "isr", # Alias for ISR
    "log_isr", # Alias for log_ISR
    "logn_isr" # Alias for logN_ISR
    #"bordafuse" #Alias for BordaFuse
    #"condorcet" #Alias for Condorcet --> TOO LONG TO EXECUTE
  ],
  'g_fusion_methods_supervised': [#'gmnz',
                             'rrf',
                             #'probfuse', #excluded. Problems with qrels: AttributeError: 'dict' object has no attribute 'to_typed_list'
                             #'slidefuse', #same problem as probfuse
                             #'bayesfuse', #same problem as probfuse
                             #'wmnz', # Probably too long to execute
                             #'rbc',
                             #'logn_isr',
                             #'posfuse', #same problem as probfuse
                             #'segfuse', #same problem as probfuse
                             'mapfuse'
                             #'wsum',
                             #'mixed',
                             #'w_bordafuse'
                             #'w_condorcet', # Probably too long to execute
                             ]



}


# define the folder where to save Ranx reports
ranx_reports_dir = '_ranx_reports'


# Set the project directory related to a particular NANO BEIR dataset
project_dir = dataset_to_use_env
# Check if the folder exists, and create it if not
if not os.path.exists(project_dir):
    os.makedirs(project_dir) 

# Add the directory to the Python module search path

sys.path.append(project_dir)
sys.path.append(ranx_reports_dir)

# Define the other directories

## Save intermediate results (embeddings or similarity analysis) #former: embeddings_save_dir (directory name: "embeddings")
intermediate_results_dir=os.path.join(project_dir, 'intermediate_results')
## Check if the folder exists, and create it if not
if not os.path.exists(intermediate_results_dir):
    os.makedirs(intermediate_results_dir)

### Folder to save or load final results (included generated texts, embeddings etc.) 
results_dir=os.path.join(project_dir, '_results')
## Check if the folder exists, and create it if not
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

### Folders with prompts
#prompts_dir=os.path.join(project_dir, 'prompts')
prompts_dir = 'prompts'
## Check if the folder exists, and create it if not
#if not os.path.exists(prompts_dir):
 #   os.makedirs(prompts_dir)

### Folder to save or load ranx objects 
ranx_dir=os.path.join(project_dir, '_ranx')
## Check if the folder exists, and create it if not
if not os.path.exists(ranx_dir):
    os.makedirs(ranx_dir)
