# Adapted function to use the NanoBEIR suite

# INPUTS
## ds: the name of the BEIR dataset to be loaded (eg: "scifact")
#NOT RELEVANT ## split: the split of the dataset to load (e.g. "train")

## NOTA BENE 1: check BEIR documentation for exact names and splits composition as there is no further control in this function
## NOTA BENE 2: the content of this function is adapted from https://colab.research.google.com/drive/1HfutiEhHMJLXiWGT8pcipxT5L2TpYEdt?usp=sharing

# OUTPUTS: all outputs mimicks the format in the BEIR datasets
## corpus: the corpus of documents in the form {doc_id:{text:text, title:title}}
## queries: the dictionary of queries in the form {query_id: query}
## qrels: the list of real query/docs matches, in the form {query_id:{doc_id:1, doc_id:1}}

## NOTA BENE 3: corpus, queries, qrels are returned by the function in this precise order!

def f_load_beir(ds, split):
    import pandas as pd
    import warnings
    # Warning
    warnings.warn(
        "This function allows loading datasets from HF Nano Beir. Only the training split is available, "
        "which is why the provided split input does not affect the output of the function.",
        category=UserWarning
    )

    # Load the dataset based on the provided `ds` parameter
    if ds == 'scifactnano':
        corpus_pd = pd.read_parquet("hf://datasets/zeta-alpha-ai/NanoSciFact/corpus/train-00000-of-00001.parquet")
        qrels_pd = pd.read_parquet("hf://datasets/zeta-alpha-ai/NanoSciFact/qrels/train-00000-of-00001.parquet")
        queries_pd = pd.read_parquet("hf://datasets/zeta-alpha-ai/NanoSciFact/queries/train-00000-of-00001.parquet")
    elif ds == 'fiqanano':
        corpus_pd = pd.read_parquet("hf://datasets/zeta-alpha-ai/NanoFiQA2018/corpus/train-00000-of-00001.parquet")
        qrels_pd = pd.read_parquet("hf://datasets/zeta-alpha-ai/NanoFiQA2018/qrels/train-00000-of-00001.parquet")
        queries_pd = pd.read_parquet("hf://datasets/zeta-alpha-ai/NanoFiQA2018/queries/train-00000-of-00001.parquet")
    elif ds == 'nfcorpusnano':
        corpus_pd = pd.read_parquet("hf://datasets/zeta-alpha-ai/NanoNFCorpus/corpus/train-00000-of-00001.parquet")
        qrels_pd = pd.read_parquet("hf://datasets/zeta-alpha-ai/NanoNFCorpus/qrels/train-00000-of-00001.parquet")
        queries_pd = pd.read_parquet("hf://datasets/zeta-alpha-ai/NanoNFCorpus/queries/train-00000-of-00001.parquet")
    elif ds == 'nqnano':
        corpus_pd = pd.read_parquet("hf://datasets/zeta-alpha-ai/NanoNQ/corpus/train-00000-of-00001.parquet")
        qrels_pd = pd.read_parquet("hf://datasets/zeta-alpha-ai/NanoNQ/qrels/train-00000-of-00001.parquet")
        queries_pd = pd.read_parquet("hf://datasets/zeta-alpha-ai/NanoNQ/queries/train-00000-of-00001.parquet")
    elif ds == 'msmarconano':
        corpus_pd = pd.read_parquet("hf://datasets/zeta-alpha-ai/NanoMSMARCO/corpus/train-00000-of-00001.parquet")
        qrels_pd = pd.read_parquet("hf://datasets/zeta-alpha-ai/NanoMSMARCO/qrels/train-00000-of-00001.parquet")
        queries_pd = pd.read_parquet("hf://datasets/zeta-alpha-ai/NanoMSMARCO/queries/train-00000-of-00001.parquet")
    elif ds == 'climatefevernano':
        corpus_pd = pd.read_parquet("hf://datasets/zeta-alpha-ai/NanoClimateFEVER/corpus/train-00000-of-00001.parquet")
        qrels_pd = pd.read_parquet("hf://datasets/zeta-alpha-ai/NanoClimateFEVER/qrels/train-00000-of-00001.parquet")
        queries_pd = pd.read_parquet("hf://datasets/zeta-alpha-ai/NanoClimateFEVER/queries/train-00000-of-00001.parquet")
    elif ds == 'dbpedianano':
        corpus_pd = pd.read_parquet("hf://datasets/zeta-alpha-ai/NanoDBPedia/corpus/train-00000-of-00001.parquet")
        qrels_pd = pd.read_parquet("hf://datasets/zeta-alpha-ai/NanoDBPedia/qrels/train-00000-of-00001.parquet")
        queries_pd = pd.read_parquet("hf://datasets/zeta-alpha-ai/NanoDBPedia/queries/train-00000-of-00001.parquet")
    elif ds == 'fevernano':
        corpus_pd = pd.read_parquet("hf://datasets/zeta-alpha-ai/NanoFEVER/corpus/train-00000-of-00001.parquet")
        qrels_pd = pd.read_parquet("hf://datasets/zeta-alpha-ai/NanoFEVER/qrels/train-00000-of-00001.parquet")
        queries_pd = pd.read_parquet("hf://datasets/zeta-alpha-ai/NanoFEVER/queries/train-00000-of-00001.parquet")
    elif ds == 'hotpotqanano':
        corpus_pd = pd.read_parquet("hf://datasets/zeta-alpha-ai/NanoHotpotQA/corpus/train-00000-of-00001.parquet")
        qrels_pd = pd.read_parquet("hf://datasets/zeta-alpha-ai/NanoHotpotQA/qrels/train-00000-of-00001.parquet")
        queries_pd = pd.read_parquet("hf://datasets/zeta-alpha-ai/NanoHotpotQA/queries/train-00000-of-00001.parquet") 
    elif ds == 'quoraretrievalnano':
        corpus_pd = pd.read_parquet("hf://datasets/zeta-alpha-ai/NanoQuoraRetrieval/corpus/train-00000-of-00001.parquet")
        qrels_pd = pd.read_parquet("hf://datasets/zeta-alpha-ai/NanoQuoraRetrieval/qrels/train-00000-of-00001.parquet")
        queries_pd = pd.read_parquet("hf://datasets/zeta-alpha-ai/NanoQuoraRetrieval/queries/train-00000-of-00001.parquet")
    elif ds == 'scidocsnano':
        corpus_pd = pd.read_parquet("hf://datasets/zeta-alpha-ai/NanoSCIDOCS/corpus/train-00000-of-00001.parquet")
        qrels_pd = pd.read_parquet("hf://datasets/zeta-alpha-ai/NanoSCIDOCS/qrels/train-00000-of-00001.parquet")
        queries_pd = pd.read_parquet("hf://datasets/zeta-alpha-ai/NanoSCIDOCS/queries/train-00000-of-00001.parquet")
    elif ds == 'arguananano':
        corpus_pd = pd.read_parquet("hf://datasets/zeta-alpha-ai/NanoArguAna/corpus/train-00000-of-00001.parquet")
        qrels_pd = pd.read_parquet("hf://datasets/zeta-alpha-ai/NanoArguAna/qrels/train-00000-of-00001.parquet")
        queries_pd = pd.read_parquet("hf://datasets/zeta-alpha-ai/NanoArguAna/queries/train-00000-of-00001.parquet")
    elif ds == 'touche2020nano':
        corpus_pd = pd.read_parquet("hf://datasets/zeta-alpha-ai/NanoTouche2020/corpus/train-00000-of-00001.parquet")
        qrels_pd = pd.read_parquet("hf://datasets/zeta-alpha-ai/NanoTouche2020/qrels/train-00000-of-00001.parquet")
        queries_pd = pd.read_parquet("hf://datasets/zeta-alpha-ai/NanoTouche2020/queries/train-00000-of-00001.parquet")
    else:
        raise ValueError("Error: Dataset name not found. Please provide a valid dataset name.")

    # convert pd dataframes to dictionaries with the suitable format
    
    ## Corpus
    corpus = corpus_pd.set_index('_id').apply(lambda row: {'text': row['text'], 'title': ''}, axis=1).to_dict()
    
    ## Queries
    queries = queries_pd.set_index('_id').apply(lambda row: row['text'], axis=1).to_dict()
    
    ## for qrels, a bit of work
    qrels = {}
    # Iterate over the DataFrame rows
    for _, row in qrels_pd.iterrows():
        key = row['query-id']
        document = row['corpus-id']
        
        # Check if the key already exists in the dictionary
        if key not in qrels:
            # If not, create a new entry with the current document
            qrels[key] = {document: 1}
        else:
            # If the key exists, add/update the document in the dictionary
            qrels[key][document] = 1

    return corpus, queries, qrels
