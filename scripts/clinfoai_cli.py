import sys
import os
import pdb
from  pathlib import  Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from  config import OPENAI_API_KEY,NCBI_API_KEY,EMAIL
from  src.clinfoai.pubmed_engine import PubMedNeuralRetriever

## INPUTS:
MODEL:str    = "Qwen/Qwen2-beta-7B-Chat"
QUESTION:str = "What is the prevalence of COVID-19 in the United States?"

## Set Path for clinfo.AI prompt architecture:
architecture_path   = os.path.join("src","clinfoai","prompts","PubMed","Architecture_1","master.json")

## Init PubMed Reriever:
nrpm = PubMedNeuralRetriever(
    architecture_path=architecture_path,
    model   = MODEL,
    verbose = True,
    debug   = False,
    open_ai_key = OPENAI_API_KEY,
    email       = EMAIL)

## STEP 1 (Search PubMed): Convert the question into a query using an LLM
# This returns a list of queries (containing MESH terms)
# These queries are used to retrieve articles from NCBI
# Once retrieved we collect a list article ids.
pubmed_queries, article_ids = nrpm.search_pubmed(
    question=QUESTION,
    num_results=10,
    num_query_attempts=1)

## STEP 2 (Fetch article data):
#  Convert  list of Ids into a list of dictionaries (populated by PubMed API) containing metadata (e.g abstract)
articles:list[dict] = nrpm.fetch_article_data(article_ids)

## STEP 3 Summarize each article (only if they are relevant [Step 3]) ###
article_summaries,irrelevant_articles =  nrpm.summarize_each_article(articles, QUESTION)

### STEP 4: Synthesize the results ###
synthesis =   nrpm.synthesize_all_articles(article_summaries, QUESTION)

#synthesis, article_summaries, irrelevant_articles, articles, article_ids, pubmed_queries,
print(synthesis)

pdb.set_trace()





