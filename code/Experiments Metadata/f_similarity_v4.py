# Define a function to compute the rank given a list of documents and similarity scores
def assign_ranks(docs_and_scores):
  # make a copy of the dictionary
  docs_and_scores_copy=docs_and_scores.copy()
  # Sort the items by score in descending order
  sorted_items = sorted(docs_and_scores_copy.items(), key=lambda x: x[1]['score'], reverse=True)
  # Iterate over the sorted items and assign ranks
  rank = 0
  prev_score = None
  for doc_id, doc_data in sorted_items:
      if doc_data['score'] != prev_score:
          rank += 1
          prev_score = doc_data['score']
      doc_data['rank'] = rank

  return docs_and_scores_copy

############


import numpy as np
from numpy.linalg import norm
import copy

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


  ## STEP 2: Define a function to compute cosine similarity between documents and queries and compute the rank (with another ad-hoc function --> developed with ChatGPT) ##

  # Function to compute cosine similarity between queries and document embeddings
  def cosine_similarity(q_id, doc_id):
      if doc_id not in document_embeddings_short.keys() or q_id not in query_embeddings_short.keys():
          return None  # Handle missing documents
      embedding_q = query_embeddings_short[q_id]
      embedding_d = document_embeddings_short[doc_id]
      similarity = np.dot(embedding_q, embedding_d) / (document_norms[doc_id] * query_norms[q_id])
      return similarity


  # Compute cosine similarity for a large number of documents efficiently

  ## Iterate over all queries and documents

  for q_id in query_embeddings_short.keys():
    # create a copy of the document embeddings input to be expanded with scores
    document_embeddings_w_score = copy.deepcopy(documents_embeddings)

    #iterate over all documents and compute similarity and rank
    for doc_id in document_embeddings_short.keys():
            similarity_score = cosine_similarity(q_id, doc_id)
            #update the document embedding dictionary with the computed cosine similarity
            document_embeddings_w_score[doc_id]['score']=similarity_score
            # add the score_type
            document_embeddings_w_score[doc_id]['score_type']='cosine similarity'
    #compute rank
    #document_embeddings_w_score_rank=assign_ranks(document_embeddings_w_score)

    # sort by score
    document_embeddings_w_score = dict(sorted(document_embeddings_w_score.items(), key=lambda item: item[1]['score'],reverse=True))

    # Extract only score and keep the top_k_to_keep
    document_embeddings_w_score_only = {doc_id: embedding_dict['score'] for doc_id, embedding_dict in list(document_embeddings_w_score.items())[:top_k_to_keep]}

    #update dictionary with queries, docs, scores and ranks
    queries_docs_scores.update({q_id:document_embeddings_w_score_only})


  ## STEP 2 - END ##

  return queries_docs_scores