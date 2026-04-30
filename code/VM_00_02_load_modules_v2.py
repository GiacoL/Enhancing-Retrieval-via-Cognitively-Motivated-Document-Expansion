
import subprocess

# import folder structure
from VM_00_01_folders_and_global_variables import project_dir,intermediate_results_dir,results_dir,prompts_dir,info_g_variables 


# Import modules

import torch
import subprocess
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from pickle import TRUE
#subprocess.run('pip install vllm', shell=True)
from vllm import LLM, SamplingParams
import numpy as np
import time
from itertools import islice
from multiprocessing import Pool
import openai
#from beir import util, LoggingHandler
import logging
import pathlib, os
#from beir.datasets.data_loader import GenericDataLoader
#from beir.retrieval.evaluation import EvaluateRetrieval
#from beir.retrieval import models
#from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from tqdm.auto import tqdm
from typing import List
import pandas as pd
import numpy as np
import joblib
import importlib.util
import copy
import pandas as pd

#from f_load_beir import f_load_beir
#from f_embeddings import f_embeddings
#from f_similarity_v2 import f_similarity
#from f_fusion import f_fusion
#from f_performance import f_performance

