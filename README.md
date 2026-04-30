# Enhancing Information Retrieval Via Cognitively Motivated Document Expansion
This study explores the possibility of using the capabilities of Large Language Models (LLMs) to improve performance in document retrieval tasks. Using human-written prompts based on the 5E Instructional Model (5E), alternative versions of documents in a given corpus are generated via a LLM, tapping into its vast knowledge base. These generated texts are then used in retrieval tasks, replacing the original corpus. The retrieval results with different generated corpora can be then combined with fusion algorithms. While individually, the generated texts do not outperform the original corpus, fusing retrieval results from multiple generated corpora with the retrieval results of the original corpus often leads to performance improvements. This suggests that LLMs-generated documents, while not a substitute for the original, can complement it, enhancing retrieval outcomes.

## Code
The pipeline works as follow: load the metadata and necessary modules (0), launch the vllm server and then the generate text (1); Afterward, embeddings can be calculated for queries and documents (2), similarity calculated (3) and lastly fusion and eveluation (4). 

*VM__Execute.py*: execute more scripts in one go and retain information on execution times
**Nota Bene**: only 2, 3, and 4 can be executed automatically. For text generation, manually launch the vllm server and then execute the script. 

### Requirements
requirements.txt

### 0 - Load 
*VM_00_01_folders_and_global_variables.py*: metadata for the experiment to be executed
*VM_00_02_load_modules_v2.py*

### 1 - Text generation
*VM_01_01_vllm_server_launch.py*
*VM_02_02_script_generate_text_Beir_v4.py*

### 2 - Embeddings calculation
*VM_02_01_Embeddings_v3.py*: embeddigs for documents
*VM_02_01_Embeddings_queries_v2.py*: embeddings for queries

### 3 - Similarity calculation
*VM_03_01_Similarity_v3.py*


### 4 - Fusion and evaluation
*VM_ranx.py*

### Other special versions
*VM_02_02_script_generate_text_Beir_v4_subset.py*: generate texts in batches (start/end to be set by the user and VM_00_01 to be updated with 'g_start_subset'and 'g_end_subset'
*VM_03_01_Similarity_v3_parallel_batch_only_train_split.py*: similarity calculation with execution in batches for the documents; with multithreading and vectorized computations. Only the test-split of the datais considered but the for loop can be expanded with train and dev

### Functions for specific sub-tasks
*f_beir_gen_text.py*: function to generate text given a corpus and prompts
*f_embeddings_v5.py*: embedding calculation
*f_load_beir.py*: load a dataset from the BEIR collection
*f_similarity_v4.py*: similarity calculation
*f_similarity_v4_parallel_multithread_vect_batch*.py: similarity calculation with batch, multithreading and vectorized operations

FOLDER **code/Experiments-Metadata**: all the *VM_00_01_folders_and_global_variables.py* for the experiments performed
