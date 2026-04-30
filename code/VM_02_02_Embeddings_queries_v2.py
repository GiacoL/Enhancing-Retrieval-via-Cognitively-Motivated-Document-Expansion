
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
g_embedding_model=info_g_variables['g_embedding_model'] #$
g_beir_dataset=info_g_variables['g_beir_dataset']
g_step_size_embeddings=info_g_variables['g_step_size_embeddings']
g_batch_embeddings=info_g_variables['g_batch_embeddings']

print(g_embedding_model, g_beir_dataset, g_step_size_embeddings, g_batch_embeddings)



### LOAD THE DATA FOR THE EXPERIMENT: queries, corpus, qrels from BEIR, compute and save embeddings for each split
# xxxcorpus, xxxqrels are not relevant since only queries concern this script

for split in ('train','test','dev'):
  try:
    xxxcorpus, queries, xxxqrels = f_load_beir(ds=g_beir_dataset,split=split)
      # Generate embeddings
    query_embeddings_file_name='query_embeddings_Beir_'+g_beir_dataset+'_'+split+'_'+g_embedding_model+'.pkl'
    query_embeddings= f_embeddings(text_objects=queries,embedding_model=g_embedding_model,Beir_element='beir_query',batch=g_batch_embeddings,step_size=g_step_size_embeddings,save_dir=intermediate_results_dir,file_name=query_embeddings_file_name[:-4])
    #save the query embeddings
    import pickle
    with open(os.path.join(results_dir,query_embeddings_file_name), 'wb') as file:
        pickle.dump(query_embeddings, file)
    print("Dataset loaded successfully.")
  except Exception as e:
      print(f"Split not found or other error: {e}")


### 1 - END