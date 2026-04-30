#Import the inputs related to the experiments
from VM_00_01_folders_and_global_variables import project_dir,intermediate_results_dir,results_dir,prompts_dir,ranx_dir,info_g_variables 

# Define some global info variables
g_exp_gen=info_g_variables['g_exp_gen'] #$ name of the generated text experiment
g_llm=info_g_variables['g_llm']
g_beir_dataset=info_g_variables['g_beir_dataset']
g_prompts_to_use_script=info_g_variables['g_prompts_to_use_script']
#Define a step_size to execute the function in batches
g_step_size=info_g_variables['g_step_size_gen']

#print(g_exp_gen, g_llm,g_beir_dataset,g_prompts_to_use_script,g_step_size)

##



### INSTALL AND IMPORT

# import folder structure


import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from pickle import TRUE
from vllm import LLM, SamplingParams
import numpy as np
import time
from itertools import islice
from beir import util, LoggingHandler
from multiprocessing import Pool
import logging
import pathlib, os
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
import subprocess
from tqdm.auto import tqdm
from typing import List
import pandas as pd
import numpy as np
from numpy.linalg import norm
import joblib
import random
from itertools import islice
from itertools import combinations


# import the function to generate text and to load the BEIR Dataset
from f_beir_gen_text import f_beir_gen_text
from f_load_beir import f_load_beir


### LOAD THE DATA FOR THE EXPERIMENT: queries, corpus, qrels from BEIR
# xxx_queries, xxx_qrels are not needed
# split="test" by default. Here the split DOES NOT play any role because we are working with the corpus and the split affects only queries/qrels

corpus, xxx_queries, xxx_qrels = f_load_beir(ds=g_beir_dataset,split="test")

corpus_to_use=corpus

#docs_specific_keys = ['13072112','15488881','8126244','13497630','15058155','16237005','15305881']
#corpus_to_use = {key: value for key, value in corpus.items() if key in docs_specific_keys}

###

### Load the prompts to generate text 

load_prompts_filename = os.path.join(prompts_dir,g_prompts_to_use_script)
# Read the content of the script file
with open(load_prompts_filename, 'r') as file:
    script_code = file.read()
# Execute the script code
exec(script_code)

###

start_index_to_use_gen=0
prompts_set=prompts_to_use.copy()

### CODE TO USE ONLY IF THE SCRIPTS STOPS BEFORE THE END
## start_index: in case the execution stops before the end, this variable set the starting point of the index.
##### But in the execution script, the subset of prompts still to be used has to be passed and the first one must be 
##### the one which was only partially completed
##### (e.g. EN1 only until index 4000, then start index is 4000 + step_size and the first prompt needs to be EN1) 
### START

#e.g. start_index_to_use_gen= xxx + g_step_size  ## the index in the file name of the last embedding file saved
#start_index_to_use_gen= 0 + g_step_size

# set variable prompts_set with the "subset not executed entirely yet" of the prompts keys to use
#e.g. prompts_set=["EL6":"Prompt EL6","EL7":"Prompt EL7"]
#prompts_set={key:value for key,value in prompts_set.items() if key in ("EL6","EL7")}

###END



# Initialize an empty dictionary to store the combined results
generated_text_dic={}

start=time.time()

for p_key,p_value in prompts_set.items():
  iterable = []
  for doc_key in corpus_to_use:
      iterable.append(({doc_key: corpus_to_use[doc_key]}, {p_key:p_value}))

  if __name__ == "__main__":
      with torch.no_grad():
        for i in tqdm(range(int(start_index_to_use_gen/g_step_size), int(len(iterable) / g_step_size) + 1)):
          start_id = i * g_step_size
          end_id = start_id + g_step_size
          
          temp_iterable = iterable[start_id: end_id]
          with Pool(processes=info_g_variables['vllm_processes']) as pool:
            result = pool.starmap(f_beir_gen_text, temp_iterable)
          if intermediate_results_dir is not None:
              joblib.dump(result, os.path.join(intermediate_results_dir, f"{g_exp_gen}_{p_key}_{start_id}.pbz2"))
          #elif encoded.shape[0] > 0:
           #   full_encoded.append(encoded)
          start_index_to_use_gen=0
  

# Rebuild the saved files

for p_key,p_value in prompts_to_use.items():
  # Initialize an empty dictionary to store the combined results
  ## the part with the variable "iterable" is a copy/paste of the part above 
  ## not the best way to write the code 
  result_dict = {}
  iterable = []
  for doc_key in corpus_to_use:
      iterable.append(({doc_key: corpus_to_use[doc_key]}, {p_key:p_value}))

  if intermediate_results_dir is not None:
        for i in tqdm(range(0, int(len(iterable) / g_step_size) + 1)):
            start_id = i * g_step_size
            result = joblib.load(os.path.join(intermediate_results_dir, f"{g_exp_gen}_{p_key}_{start_id}.pbz2"))
            #if result.shape[0] > 0:
            for item in result:
              result_dict.update(item)      

  generated_text_dic.update({p_key:{'prompt':p_value, 'generated_documents':result_dict}})

end=time.time()
execution_time_minutes = (end-start) / 60
####################################################################################

# Save the output
import pickle
with open(os.path.join(results_dir,'generated_text_dic_'+g_exp_gen+'.pkl'), 'wb') as file:
    pickle.dump(generated_text_dic, file)

#txt_file_path = os.path.join(results_dir, 'generated_text_dic_' + g_exp_gen + '.txt')
# Save the dictionary to a text file
#with open(txt_file_path, 'w', encoding='utf-8') as file:
#    for key, value in generated_text_dic.items():
 #       file.write(f'{key}: {value}\n')

# Collect all experiment infos in a dictionary and add prompts keys
infos_g_exp_gen={
  'g_exp_gen':g_exp_gen, #generation text - experiment name
  'llm': g_llm,
  'Beir dataset':g_beir_dataset,
  #'Beir split':g_beir_split,
  'Prompts':prompts_to_use,
  'Prompts_keys': list(prompts_to_use.keys()),
  'Step size':g_step_size,
  'Total execution time - minutes': execution_time_minutes

}
# Save the experiment infos - pkl and txt
## txt
with open(os.path.join(results_dir,'infos_'+g_exp_gen+'.txt'), 'w') as file:
    # Write strings to the file
   for key, value in infos_g_exp_gen.items():
        file.write(f"{key}: {value}\n")
## pkl
with open(os.path.join(results_dir,'infos_'+g_exp_gen+'.pkl'), "wb") as file:
    pickle.dump(infos_g_exp_gen, file)
