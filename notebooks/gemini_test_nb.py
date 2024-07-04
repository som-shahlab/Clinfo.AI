import sys
import os
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
from config import GOOGLE_API_KEY, NCBI_API_KEY, EMAIL
from src.clinfoai.pubmed_engine import PubMedNeuralRetriever
# Set your Google API key here
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Import and configure the Gemini API
import google.generativeai as genai
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Define the model to use
MODEL = "gemini-1.5-flash"

# Path to the prompts configuration
PROMPS_PATH = os.path.join("..", "src", "clinfoai", "prompts", "PubMed", "Architecture_1", "master.json")

# Initialize the PubMed Neural Retriever with the Gemini model
nrpm = PubMedNeuralRetriever(
    architecture_path=PROMPS_PATH,
    model=MODEL,
    verbose=False,
    debug=False,
    open_ai_key=GOOGLE_API_KEY,  # Use Google API key for the Gemini model
    email=EMAIL
)

# Define the question
QUESTION = "What is the prevalence of COVID-19 in the United States?"

# Step 1: Search PubMed
pubmed_queries, article_ids = nrpm.search_pubmed(
    question=QUESTION,
    num_results=10,
    num_query_attempts=1
)

print(f"Articles retrieved: {len(article_ids)}")
print(pubmed_queries)
print(article_ids)

## Step 2: Fetch article data
# Previously, we only extracted the PMIDs. Now we will use those PMIDs to retrieve the metadata:
articles = nrpm.fetch_article_data(article_ids)

# Print example for the first article: 
article_num = 1
print(f"\nArticle {article_num}:\n")
print(articles[article_num]["MedlineCitation"]["Article"]["Abstract"]["AbstractText"])

## Step 3: Summarize each article
article_summaries, irrelevant_articles = nrpm.summarize_each_article(articles, QUESTION)

## Step 4: Synthesize all summaries to answer the question
synthesis = nrpm.synthesize_all_articles(article_summaries, QUESTION)
print("\nSynthesis:\n")
print(synthesis)