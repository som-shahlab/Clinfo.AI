import numpy as np 
from rank_bm25 import BM25Okapi
def bm25_return_n_articles(corpus:list, query:str, n:int=20,return_scores  = False):
    tokenized_corpus = [doc.lower().split(" ")for doc in corpus]
    bm25             = BM25Okapi(tokenized_corpus)
    tokenized_query = query.lower().split(" ")
    if return_scores == True:
        doc_scores      = bm25.get_scores(tokenized_query)
        return doc_scores
   
    else:
         top_n = bm25.get_top_n(tokenized_query, corpus, n=n)
         return top_n
    
def bm25_ranked(list_to_oganize,corpus:list,query:str,n:int=20):
    new_order = np.argsort(-1*bm25_return_n_articles(corpus, query,return_scores = True))[0:n]
    new_list = [list_to_oganize[i] for i in new_order]
    return new_list 