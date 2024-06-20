import numpy as np 
from rank_bm25 import BM25Okapi

def bm25_return_n_articles(corpus:list, query:str, n:int=20,return_scores  = False):
    """
    Retrieve top N articles matching a query using the BM25 algorithm.

    Parameters
    ----------
    corpus : list
        A list of documents (each document is a string) to search within.
    query : str
        The query string to search for within the corpus.
    n : int, optional
        The number of top documents to return (default is 20).
    return_scores : bool, optional
        If True, return the BM25 scores for all documents. If False, return the top N documents (default is False).

    Returns
    -------
    list or numpy.ndarray
        If return_scores is False, returns a list of the top N documents matching the query.
        If return_scores is True, returns a numpy array of BM25 scores for all documents in the corpus.

    Examples
    --------
    >>> corpus = ["This is a document.", "This is another document.", "Yet another document."]
    >>> query = "document"
    >>> bm25_return_n_articles(corpus, query, n=2)
    ['This is a document.', 'This is another document.']
    
    >>> bm25_return_n_articles(corpus, query, return_scores=True)
    array([0.7, 0.6, 0.5])
    """
    
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
    """
    Rank and reorder a list based on the BM25 scores of documents in a corpus.

    Parameters
    ----------
    list_to_oganize : list
        The list to reorder based on the ranking of the corresponding documents.
    corpus : list
        A list of documents (each document is a string) to search within.
    query : str
        The query string to search for within the corpus.
    n : int, optional
        The number of top documents to consider for reordering the list (default is 20).

    Returns
    -------
    list
        A new list reordered according to the BM25 scores of the documents in the corpus.

    Examples
    --------
    >>> list_to_oganize = ["item1", "item2", "item3"]
    >>> corpus = ["This is a document.", "This is another document.", "Yet another document."]
    >>> query = "document"
    >>> bm25_ranked(list_to_oganize, corpus, query, n=2)
    ['item1', 'item2']
    """
    new_order = np.argsort(-1*bm25_return_n_articles(corpus, query,return_scores = True))[0:n]
    new_list = [list_to_oganize[i] for i in new_order]
    return new_list 