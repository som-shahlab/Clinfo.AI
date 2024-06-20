import sys
import os 
from  pathlib import  Path
sys.path.append(str(Path(__file__).resolve().parent))
from  pathlib  import  Path
from  src.clinfoai.pubmed_engine          import PubMedNeuralRetriever
from  src.clinfoai.semanticscholar_engine import SemanticScholarNeuralRetriever
from  src.clinfoai.bm25                   import bm25_ranked



class ClinfoAI:
    def __init__(self,
        architecture_path,
        llm:str        = "gpt-3.5-turbo",
        engine:str     = "PubMed",
        openai_key:str = "YOUR API TOKEN", 
        email:str      = "YOUR EMAIL",
        
      
        verbose:str=False) -> None:

        self.engine             = engine
        self.llm                = llm 
        self.email              = email
        self.openai_key         = openai_key
        self.verbose            = verbose
        self.architecture_path  = architecture_path
        self.init_engine()
    def init_engine(self):
        if self.engine  == "PubMed":
        
            self.NEURAL_RETRIVER   = PubMedNeuralRetriever(
                                        architecture_path=self.architecture_path,
                                        model       = self.llm,
                                        verbose     = self.verbose ,
                                        debug       = False,
                                        open_ai_key = self.openai_key,
                                        email       = self.email)
            print("PubMed Retriever Initialized")

        elif self.engine  == "SemanticScholar":
            self.NEURAL_RETRIVER = SemanticScholarNeuralRetriever(
                                        architecture_path=self.architecture_path,
                                        model       = self.llm, 
                                        verbose= self.verbose ,
                                        debug=False,
                                        open_ai_key=self.openai_key,
                                        email=self.email)
        else:
            raise Exception("Invalid Engine")

        return "OK"
    

    def retrive_articles(self,question,restriction_date = None, ignore=None):
        try:
            if self.engine  == "PubMed":
                queries, article_ids = self.NEURAL_RETRIVER.search_pubmed(question             = question,
                                                                          num_results        = 16,
                                                                          num_query_attempts = 3,
                                                                          restriction_date   = restriction_date) 

            elif self.engine  == "SemanticScholar":
                query        = self.NEURAL_RETRIVER.generate_semantic_query(question=question)
                article_ids  = [1,2,3]
                queries      = [query]
        except: 
            print(f"Internal Service Error, {self.engine } might be down ")
         
        if ignore != None:
            try:
                print("Article dropped")
                article_ids.remove(ignore)
            except:
                pass

        if (not article_ids) or (not queries) or (len(article_ids) == 0) or (len(queries) == 0):
            print(f"Sorry, we weren't able to find any articles in {self.engine} relevant to your question. Please try again.")
            return
    
        try:
            if self.engine == "PubMed":
                articles = self.NEURAL_RETRIVER.fetch_article_data(article_ids)
            
            elif self.engine == "SemanticScholar":
                articles    = self.NEURAL_RETRIVER.search_semantic_scholar(query,limit=50,threshold = 10,minimum_return=5,verbose=True)
                article_ids = articles

            if  self.verbose:
                print(f'Retrieved {len(articles)} articles. Identifying the relevant ones and summarizing them (this may take a minute)')
            

        except:
            print('error',f"Articles could not be fetched from {self.engine}")
        
        if len(articles) ==0:
            print(f"Articles could not be fetched from {self.engine}, 0")
           
        return articles,queries
    

    def summarize_relevant(self,articles,question):
        article_summaries,irrelevant_articles = self.NEURAL_RETRIVER.summarize_each_article(articles, question)
        return   article_summaries,irrelevant_articles 
    

    def synthesis_task(self,article_summaries, question,USE_BM25=False,with_url=True ):
        if USE_BM25:
            if len(article_summaries) > 21:
                print("Using BM25 to rank articles")
                corpus            = [article['abstract'] for article in article_summaries]
                article_summaries = bm25_ranked(list_to_oganize= article_summaries,corpus =  corpus,query = question,n = 20)

        synthesis = self.NEURAL_RETRIVER.synthesize_all_articles(article_summaries, question ,with_url=with_url)
        return synthesis


    def forward(self,question,restriction_date = None, ignore=None,return_articles=True):         
        articles,queries                       = self.retrive_articles(question,restriction_date , ignore)
        article_summaries,irrelevant_articles  = self.summarize_relevant(articles=articles,question=question)
        synthesis                              = self.synthesis_task(article_summaries, question)
        out = dict()
        out["synthesis"] = synthesis 
        if return_articles:
            out["article_summaries"] = article_summaries 
            out["irrelevant_articles"] = irrelevant_articles  
            out["queries"] = queries              
        
        return out 