

import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from pickle import TRUE
import os
from tqdm.auto import tqdm
from typing import List
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm



### Function to compute embeddings for documents or queries in the format as in the BEIR dataset.
#### By default die embeddings are normalised
#### Each model comes with some text prompts to optimize embeddings. See docu for each model in sentence trasnformers

# INPUTS:
## text_objects: queries or documents (corpus) in the same format as 'queries' and 'corpus' in the BEIR dataset
## embedding_model: model to compute embeddings. Available options:  E5, Distill_Roberta_Base
## Beir_element: quueries (='beir_query') or documents ('beir_corpus')
## batch: the size of the batch. Default is 64
## normalize: dummy to indicate if the embeddigs are to be normalized. Default is True
## start_index: in case the execution stops before the end, this variable set the starting point of the index.
##### But in the execution script, the subset of prompts still to be used has to be passed and the first one must be 
##### the one which was only partially completed
##### (e.g. EN1 only until index 4000, then start index is 4000 + step_size and the first prompt needs to be EN1) 
##### Default is zero

# OUTPUT: a dictionary with embeddings for every doc_id/q_id and infos on embedding dimension, model and truncation of input

# ChatGPT Embeddings

def f_embeddings(
    text_objects,
    embedding_model,
    save_dir,
    file_name,
    Beir_element=None,
    batch=64,
    step_size=1000,
    normalize=True,
    start_index=0,
    model_to_use=None  # Optional: pre-loaded model
):
    import torch
    import gc
    import joblib
    import os
    import numpy as np
    from tqdm import tqdm
    from sentence_transformers import SentenceTransformer

    # Model selection
    model_options = {
        "E5": 'embaas/sentence-transformers-e5-large-v2',
        "Distill_Roberta_Base": 'sentence-transformers/all-distilroberta-v1',
        "bge": 'BAAI/bge-large-en-v1.5',
        "SFR-Mistral": 'Salesforce/SFR-Embedding-Mistral',
        "UAE": 'WhereIsAI/UAE-Large-V1',
        "MiniLMv2": 'sentence-transformers/all-MiniLM-L12-v2',
        "gte-Qwen2-1.5": 'Alibaba-NLP/gte-Qwen2-1.5B-instruct',
        "infly": "infly/inf-retriever-v1-1.5b"
    }

    # Models that require prepended string prompts (e.g., instruction-tuned models)
    string_prompts = {
        "bge": {
            "beir_query": "Represent this sentence for searching relevant passages: ",
            "beir_corpus": None
        },
        "SFR-Mistral": {
            "beir_query": "Given a query, retrieve relevant passages that answer the query: ",
            "beir_corpus": None
        },
        "gte-Qwen2-1.5": {
            "beir_query": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
            "beir_corpus": None
        }
    }

    # Models that expect prompt_name passed to encode()
    prompt_name_prompts = {
        "infly": {
            "beir_query": "query",
            "beir_corpus": None
        }
    }
# old prompts
#    task_prompts = {
 #       "E5_queries": None,
  #      "bge_queries": "Represent this sentence for searching relevant passages: ",
   #     "SFR-Mistral_queries": "Given a query, retrieve relevant passages that answer the query: ",
    #    "UAE_queries": None,
     #   "MiniLMv2_queries": None,
      #  "gte-Qwen2-1.5_queries": "Instruct: Given a query, retrieve relevant documents that answer the query\nQuery: "
    #}

    task_prompt = None
    prompt_mode = None  # "string" | "name" | None

    if embedding_model in string_prompts and Beir_element in string_prompts[embedding_model]:
        task_prompt = string_prompts[embedding_model][Beir_element]
        prompt_mode = "string"

    elif embedding_model in prompt_name_prompts and Beir_element in prompt_name_prompts[embedding_model]:
        task_prompt = prompt_name_prompts[embedding_model][Beir_element]
        prompt_mode = "name"

    # Load model if not passed
    if model_to_use is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_to_use = SentenceTransformer(model_options[embedding_model], device=device)
    
    result = {}

    # Extract texts
    if Beir_element == 'beir_corpus':
        id_list = list(text_objects.keys())
        texts = [text_objects[k]['text'] for k in id_list if 'text' in text_objects[k]]
    else:
        id_list = list(text_objects.keys())
        texts = list(text_objects.values())

    # Encode in chunks
    full_encoded = []
    with torch.no_grad():
        for i in tqdm(range(int(start_index / step_size), int(len(texts) / step_size) + 1)):
            start_id = i * step_size
            end_id = start_id + step_size
            inp = texts[start_id:end_id]

            if not inp:
                continue

            # Decide input based on prompt mode
            if prompt_mode == "name":
                encoded = model_to_use.encode(
                    inp,
                    normalize_embeddings=normalize,
                    prompt_name=task_prompt,
                    batch_size=batch,
                    convert_to_numpy=True
                )
            elif prompt_mode == "string" and task_prompt:
                prefixed_inp = [task_prompt + text for text in inp]
                encoded = model_to_use.encode(
                    prefixed_inp,
                    normalize_embeddings=normalize,
                    batch_size=batch,
                    convert_to_numpy=True
                )
            else:
                encoded = model_to_use.encode(
                    inp,
                    normalize_embeddings=normalize,
                    batch_size=batch,
                    convert_to_numpy=True
                )
            
            if save_dir:
                joblib.dump(encoded, os.path.join(save_dir, f"{file_name}_{start_id}_embs.pbz2"))
            else:
                full_encoded.append(encoded)

            torch.cuda.empty_cache()
            gc.collect()

    # Load back if needed
    if save_dir:
        full_encoded = []
        for i in tqdm(range(0, int(len(texts) / step_size) + 1)):
            start_id = i * step_size
            path = os.path.join(save_dir, f"{file_name}_{start_id}_embs.pbz2")
            if os.path.exists(path):
                encoded = joblib.load(path)
                full_encoded.append(encoded)

            torch.cuda.empty_cache()
            gc.collect()

    embeddings = np.vstack(full_encoded)

    # Batched tokenization for truncation check
    truncated = []
    #for i in range(0, len(texts), batch):
     #   toks = model_to_use.tokenize(texts[i:i + batch])
      #  truncated.extend([len(t) > model_to_use.max_seq_length for t in toks['input_ids']])

    # Build result dictionary
    for i, emb in enumerate(embeddings):
        result[id_list[i]] = {
            "embedding": emb,
            #"truncated": truncated[i],
            "model": embedding_model,
            "embeddings_dim": len(emb)
        }

    return result
