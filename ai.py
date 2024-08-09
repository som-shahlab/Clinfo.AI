import json
import sys
import os
from pathlib import Path

sys.path.append(str(Path.cwd().parent))
from config import OPENAI_API_KEY, NCBI_API_KEY, EMAIL

from src.clinfoai.pubmed_engine import PubMedNeuralRetriever

# Make Sure you followed at least step 1-2 before running this cell.
from config import OPENAI_API_KEY, NCBI_API_KEY, EMAIL

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

PROMPS_PATH = os.path.join(
    ".", "src", "clinfoai", "prompts", "PubMed", "Architecture_1", "master.json"
)
MODEL: str = "gpt-4o-2024-08-06"

nrpm = PubMedNeuralRetriever(
    architecture_path=PROMPS_PATH,
    model=MODEL,
    verbose=False,
    debug=False,
    open_ai_key=OPENAI_API_KEY,
    email=EMAIL,
)


def search_articles(question: str):
    ### STEP 1: Search PubMed ###
    pubmed_queries, article_ids = nrpm.search_pubmed(
        question, num_results=10, num_query_attempts=1
    )

    ### STEP 2: Fetch article data ###
    articles = nrpm.fetch_article_data(article_ids)

    return articles
