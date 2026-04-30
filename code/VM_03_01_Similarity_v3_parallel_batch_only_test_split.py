
#### Compute (cosine) similarity for a given set (experiment) of generated texts embeddings

# INSTALL AND IMPORT

# import folder structure
from VM_00_01_folders_and_global_variables import project_dir,intermediate_results_dir,results_dir,prompts_dir,ranx_dir,info_g_variables 



import os
#os.system('pip install sentence-transformers')
os.system('pip install datasets')
os.system('pip install beir')

import torch
#from sentence_transformers import SentenceTransformer, util
#from transformers import pipeline
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
from f_similarity_v4_parallel_multithread_vect_batch import f_similarity_score 
######


# Define some global info variables
g_exp_sim=info_g_variables['g_exp_sim'] #$ name infofile - similarity 
g_exp_emb=info_g_variables['g_exp_emb']
g_step_size_sim=info_g_variables['g_step_size_sim']
g_batch_sim=info_g_variables['g_batch_sim']

print(g_exp_sim, g_exp_emb, g_step_size_sim, g_batch_sim)


# Load the dictionary with the metadata on embeddings of generated texts
## Embeddings - Generated text - metadata
with open(os.path.join(results_dir,'infos_'+g_exp_emb+'.pkl'), "rb") as file:
    infos_g_exp_emb=pickle.load(file)

# Load the dictionary with the metadata on generated text
## Generated text - metadata
with open(os.path.join(results_dir,'infos_'+infos_g_exp_emb['g_exp_gen']+'.pkl'), "rb") as file:
    infos_g_exp_gen=pickle.load(file)


# Compute similarity for the relevant splits

for g_split in ['test']: #$ split: train, test, dev
  
  ## collect infos on this sim experiment in a dictionary
  infos_g_exp_sim = {
      'g_exp_sim': g_exp_sim,
      'g_exp_emb': g_exp_emb,
      'g_step_size_sim': g_step_size_sim,
      'g_batch_sim': g_batch_sim,
      'Beir dataset':infos_g_exp_gen['Beir dataset'],
      'g_split': g_split,
      'Embedding model': infos_g_exp_emb['g_embedding_model'] 

  }

  # Save the metadata
  with open(os.path.join(results_dir, 'infos_' + g_exp_sim + '.pkl'), "wb") as file:
      pickle.dump(infos_g_exp_sim, file)



  #######################################

  ## 1 - START: load the query embeddings
  query_embeddings_file_name='query_embeddings_Beir_'+infos_g_exp_gen['Beir dataset']+'_'+g_split+'_'+infos_g_exp_emb['g_embedding_model']+'.pkl'
  #query_embeddings_file_name='query_embeddings_Beir_'+infos_g_exp_gen['Beir dataset']+'_'+infos_g_exp_gen['Beir split']+'.pkl'
  file_path = os.path.join(results_dir, query_embeddings_file_name)
  # Check if the file exists before attempting to load it
  if os.path.exists(file_path):
      # Open the file for reading in binary mode ('rb')
      with open(file_path, 'rb') as f:
          # Load the data from the file
          query_embeddings= pickle.load(f)

  #concatenate prompt keys and 'Original'
  keys_to_use=list(infos_g_exp_gen['Prompts_keys'])
  keys_to_use.append('Original')
  print(keys_to_use)

  g_batch_docs = 10000 #each batch of docs to be used to compute similarity
  top_k = 100  # Define the number of top documents to keep for each q_id

  for obj in tqdm(keys_to_use, desc='prompt keys progression'): #infos_g_exp_gen['Prompts_keys']
    file_name_embeddings=g_exp_emb+'_'+infos_g_exp_gen['g_exp_gen']+'_'+obj+'_'+infos_g_exp_emb['g_embedding_model']+'_embeddings.pkl'

    with open(os.path.join(results_dir,file_name_embeddings), 'rb') as file:
        embeddings=pickle.load(file)

    # create a disctionary to save similarities 
    similarity_to_save = {}

    doc_ids = list(embeddings.keys())  # Get a list of document IDs

    # iterate over the documents in batches for efficient computations
    for batch_start in tqdm(range(0, len(embeddings), g_batch_docs), desc='batch progression'):
        batch_end = min(batch_start + g_batch_docs, len(embeddings))
        # Get the batch of document IDs
        batch_doc_ids = doc_ids[batch_start:batch_end]
    
        # Create a batch of embeddings by selecting the appropriate keys from the dictionary
        batch_embeddings = {doc_id: embeddings[doc_id] for doc_id in batch_doc_ids}

        # compute similarity
        similarity_to_save_temp = f_similarity_score(query_embeddings=query_embeddings,documents_embeddings=batch_embeddings)

        for q_id, doc_scores in similarity_to_save_temp.items():
            # Check if q_id exists in similarity_to_save
            if q_id not in similarity_to_save:
                # If q_id does not exist, initialize it with an empty dictionary
                similarity_to_save[q_id] = {}

            # Iterate over each document and its score in the temp dictionary
            for doc_id, score in doc_scores.items():
                if doc_id in similarity_to_save[q_id]:
                    # If the document already exists, append the score to the list
                    if isinstance(similarity_to_save[q_id][doc_id], list):
                        similarity_to_save[q_id][doc_id].append(score)
                    else:
                        similarity_to_save[q_id][doc_id] = [similarity_to_save[q_id][doc_id], score]
                else:
                    # If the document does not exist, add it
                    similarity_to_save[q_id][doc_id] = score
            
            # After updating, keep only the top k documents with the highest scores
            sorted_docs = sorted(similarity_to_save[q_id].items(), key=lambda item: max(item[1]) if isinstance(item[1], list) else item[1], reverse=True)
            similarity_to_save[q_id] = dict(sorted_docs[:top_k])
      
      #similarity_to_save.update(similarity_to_save_temp)

      

    # save results
    file_name_similarity_result=g_exp_sim+'_'+infos_g_exp_emb['g_exp_emb']+'_'+infos_g_exp_gen['g_exp_gen']+'_'+obj+'_split_'+g_split+'_similarity_scores.pkl'
    with open(os.path.join(results_dir, file_name_similarity_result), 'wb') as file:
        pickle.dump(similarity_to_save, file)
