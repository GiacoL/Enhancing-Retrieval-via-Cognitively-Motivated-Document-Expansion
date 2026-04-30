
## no sorting by top k in this script. It has to be done in the main function VM_similarity
############


import numpy as np
from numpy.linalg import norm
import copy
import concurrent.futures
from tqdm.auto import tqdm



def process_query(q_id, document_embeddings_short, query_embeddings_short, query_norms, document_norms, top_k_to_keep):
    # Check if the query exists in query embeddings
    if q_id not in query_embeddings_short:
        return None
    
    embedding_q = query_embeddings_short[q_id]

    # Convert document embeddings dictionary to a matrix for vectorized operations
    doc_ids = list(document_embeddings_short.keys())
    embeddings_d = np.array([document_embeddings_short[doc_id] for doc_id in doc_ids])

    # Calculate the norms for the document embeddings
    doc_norms = np.array([document_norms[doc_id] for doc_id in doc_ids])

    # Compute cosine similarities
    similarities = np.dot(embeddings_d, embedding_q) / (doc_norms * query_norms[q_id])

    # Create a dictionary with document embeddings and scores
    document_embeddings_w_score = {doc_id: {'embedding': document_embeddings_short[doc_id], 'score': similarities[i], 'score_type': 'cosine similarity'} for i, doc_id in enumerate(doc_ids)}

    # Sort the documents by score in descending order
    #sorted_docs = sorted(document_embeddings_w_score.items(), key=lambda item: item[1]['score'], reverse=True)

    # Keep only the top_k_to_keep documents
    #top_k_docs = sorted_docs[:top_k_to_keep]

    # Extract only the scores
    document_embeddings_w_score_only = {doc_id: embedding_dict['score'] for doc_id, embedding_dict in document_embeddings_w_score.items()}

    return q_id, document_embeddings_w_score_only

def f_similarity_score(query_embeddings,documents_embeddings, top_k_to_keep = 100):

## 3 Steps ##

  # create a dictionary to collect for each query all documents, scores and rank
  queries_docs_scores = {}


## STEP 1: extract only the doc/query id and the actual embeddings. Compute the norms (for a more efficient cosine similarity computation)##


  # create a subset of the dictionary with the documents containing just doc_id:embeddings. This will be handy for the calculation of cosine similarity
  document_embeddings_short={}

  for document_id, embedding_info in documents_embeddings.items():
    temp_emb = {document_id: embedding_info['embedding']}
    document_embeddings_short.update(temp_emb)

  # create a subset of the dictionary with the queries containing just q_id:embeddings

  query_embeddings_short={}
  for query_id, embedding_info in query_embeddings.items():
    temp_emb = {query_id: embedding_info['embedding']}
    query_embeddings_short.update(temp_emb)

  # Precompute and store the norms of document and query embeddings
  query_norms = {q_id: norm(embedding) for q_id, embedding in query_embeddings_short.items()}
  document_norms = {doc_id: norm(embedding) for doc_id, embedding in document_embeddings_short.items()}

  ## STEP 1 - END ##

    # Use ProcessPoolExecutor for CPU-bound tasks
  queries_list = list(query_embeddings_short.keys())

  #with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
   #   futures = [
    #      executor.submit(process_query, q_id, document_embeddings_short, query_embeddings_short, query_norms, document_norms, top_k_to_keep)
     #     for q_id in queries_list
      #]
      
      #results = [future.result() for future in concurrent.futures.as_completed(futures)]
  
  with concurrent.futures.ThreadPoolExecutor(max_workers=18) as executor:
        futures = [
            executor.submit(process_query, q_id, document_embeddings_short, query_embeddings_short, query_norms, document_norms, top_k_to_keep)
            for q_id in queries_list
        ]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

  for q_id, document_embeddings_w_score_only in results:
      queries_docs_scores[q_id] = document_embeddings_w_score_only

  return queries_docs_scores
  
