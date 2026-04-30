
#### Compute embeddings for a given set (experiment) of generated texts

# INSTALL AND IMPORT

# import folder structure
from VM_00_01_folders_and_global_variables import project_dir,intermediate_results_dir,results_dir,prompts_dir,ranx_dir,info_g_variables 



import os
os.system('pip install sentence-transformers')
os.system('pip install datasets')
os.system('pip install beir')

import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from pickle import TRUE
#from vllm import LLM, SamplingParams
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
import joblib
import random
from itertools import islice
import numpy as np
from numpy.linalg import norm
import pickle
from itertools import combinations
import json

# import the function to generate text and to load the BEIR Dataset
#from f_beir_gen_text import f_beir_gen_text
from f_load_beir import f_load_beir
from f_embeddings_v5 import f_embeddings

######


# Define some global info variables
g_exp_emb=info_g_variables['g_exp_emb'] #$
g_embedding_model=info_g_variables['g_embedding_model'] #$
g_exp_gen=info_g_variables['g_exp_gen'] #$ generated text to load
g_step_size_embeddings=info_g_variables['g_step_size_embeddings']
g_batch_embeddings=info_g_variables['g_batch_embeddings']

print(g_exp_emb, g_embedding_model, g_exp_gen, g_step_size_embeddings,g_batch_embeddings)



## collect them in a dictionary
infos_g_exp_emb = {   
    'g_exp_emb': g_exp_emb,
    'g_embedding_model': g_embedding_model,
    'g_exp_gen': g_exp_gen,
    'g_step_size_embeddings': g_step_size_embeddings,
    'g_batch_embeddings': g_batch_embeddings
}

# Save the metadata
with open(os.path.join(results_dir, 'infos_' + g_exp_emb + '.pkl'), "wb") as file:
    pickle.dump(infos_g_exp_emb, file)


# Load the dictionary with the generated text to be used along with other metadata
## Generated text - metadata
with open(os.path.join(results_dir,'infos_'+g_exp_gen+'.pkl'), "rb") as file:
    infos_g_exp_gen=pickle.load(file)
## Generated text - dictionary with generated texts
with open(os.path.join(results_dir,'generated_text_dic_'+infos_g_exp_gen['g_exp_gen']+'.pkl'), 'rb') as file:
    generated_text_dic=pickle.load(file)


### LOAD THE DATA FOR THE EXPERIMENT: queries, corpus, qrels from BEIR
# split="test" by default. Here the split DOES NOT play any role because we are working with the corpus and the split affects only queries/qrels

corpus, queries, qrels = f_load_beir(ds=infos_g_exp_gen['Beir dataset'],split="test")

#docs_specific_keys = ['13072112','15488881','8126244','13497630','15058155','16237005','15305881']
#corpus = {key: value for key, value in corpus.items() if key in docs_specific_keys}

#queries_specific_keys = ['1314', '1361','1405']  #query/docs 1314: 13072112, 16237005; 1361: 15488881, 15058155
#queries = {key: value for key, value in queries.items() if key in queries_specific_keys}


# append the original corpus to the dictionary of generated texts because the embeddings are needed also for the original corpus
generated_text_dic.update({'Original':{'prompt':"No prompt. This is the original text from the corpus",'generated_documents':corpus}})

### Compute embeddings for the generated texts (plus original corpus) and save them

start_index_to_use_embeddings=0
keys_set=generated_text_dic.keys()

### CODE TO USE ONLY IF THE SCRIPTS STOPS BEFORE THE END
## start_index: in case the execution stops before the end, this variable set the starting point of the index.
##### But in the execution script, the subset of prompts still to be used has to be passed and the first one must be 
##### the one which was only partially completed
##### (e.g. EN1 only until index 4000, then start index is 4000 + step_size and the first prompt needs to be EN1) 
### START
#start_index_to_use_embeddings= xxx + g_step_size_embeddings  ## the index in the file name of the last embedding file saved
#start_index_to_use_embeddings= 4 + g_step_size_embeddings
# set variable keys_set with the "subset not executed entirely yet" of generated_text_dic.keys()
#keys_set=["EL6","EL7","Original"]

###END


for item in keys_set:  
  #Embeddings - generate and save
  file_name_embeddings=g_exp_emb+'_'+infos_g_exp_gen['g_exp_gen']+'_'+item+'_'+g_embedding_model+'_embeddings.pkl'  
  result_embeddings=f_embeddings(text_objects=generated_text_dic[item]['generated_documents'], embedding_model=g_embedding_model, Beir_element='beir_corpus',save_dir=intermediate_results_dir,file_name=file_name_embeddings[:-4],batch=g_batch_embeddings,step_size=g_step_size_embeddings,start_index=start_index_to_use_embeddings)
  with open(os.path.join(results_dir,file_name_embeddings), 'wb') as file:
      pickle.dump(result_embeddings, file)
  start_index_to_use_embeddings=0

