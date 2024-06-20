import sys
import os 
sys.path.append('..')
from  pathlib  import  Path
from  utils.pubmed_utils     import Neural_Retriever_PubMed
from  utils.semantic_utils   import Neural_Retriever_Semantic_Scholar
from utils.bm25              import bm25_ranked



class ClinfoAI:
    def __init__(self,
        openai_key:str, 
        email:str,
        llm:str= "gpt-3.5-turbo",
        engine:str="SemanticScholar",
        verbose:str=False) -> None:

        self.engine             = engine
        self.llm                = llm 
        self.email              = email
        self.openai_key         = openai_key
        self.verbose            = verbose
        self.architecture_path  = self.init_engine()

    def init_engine(self):
        if self.engine  == "PubMed":
            ARCHITECTURE_PATH      = Path('../prompts/PubMed/Architecture_1/master.json')
            self.NEURAL_RETRIVER   = Neural_Retriever_PubMed(
                                        architecture_path=ARCHITECTURE_PATH,
                                        model       = self.llm,
                                        verbose     = False,
                                        debug       = False,
                                        open_ai_key = self.openai_key,
                                        email       = self.email)
            print("PubMed Retriever Initialized")

        elif self.engine  == "SemanticScholar":
            ARCHITECTURE_PATH    = Path('../prompts/SemanticScholar/Architecture_1/master.json')
            self.NEURAL_RETRIVER = Neural_Retriever_Semantic_Scholar(
                                        architecture_path=ARCHITECTURE_PATH ,
                                        model       = self.llm, 
                                        verbose=True,
                                        debug=False,
                                        open_ai_key=self.openai_key,
                                        email=self.email)
        else:
            raise Exception("Invalid Engine")

        ARCHITECTURE_PATH_STR = str(ARCHITECTURE_PATH)
        return   ARCHITECTURE_PATH_STR 
    

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
        article_summaries,irrelevant_articles = self.NEURAL_RETRIVER.summarize_each_article(articles, question,prompt_dict={"type":"automatic"})
        return   article_summaries,irrelevant_articles 
    

    def synthesis_task(self,article_summaries, question,USE_BM25=False,with_url=True ):
        if USE_BM25:
            if len(article_summaries) > 21:
                print("Using BM25 to rank articles")
                corpus            = [article['abstract'] for article in article_summaries]
                article_summaries = bm25_ranked(list_to_oganize= article_summaries,corpus =  corpus,query = question,n = 20)

        synthesis = self.NEURAL_RETRIVER.synthesize_all_articles(article_summaries, question, prompt_dict={"type":"automatic"} ,with_url=with_url)
        return synthesis


    def forward(self,question,restriction_date = None, ignore=None,return_articles=True):  
        try:
            articles,queries                              = self.retrive_articles(question,restriction_date , ignore)
            article_summaries,irrelevant_articles  = self.summarize_relevant(articles=articles,question=question)
            synthesis                              = self.synthesis_task(article_summaries, question)
        except:
            synthesis = "Internal Error"
        
        if return_articles:
            return synthesis , article_summaries, irrelevant_articles,queries
        
        return synthesis 