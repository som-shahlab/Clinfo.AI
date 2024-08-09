import os

# Make Sure you followed at least step 1-2 before running this cell.
from config import EMAIL, GOOGLE_API_KEY, NCBI_API_KEY, OPENAI_API_KEY
from src.clinfoai.pubmed_engine import PubMedNeuralRetriever

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
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


def highlight_answer(summary: str) -> str:
    summary = summary.replace("TL;DR:", "**TL;DR:**")
    summary = summary.replace("Literature Summary:", "**Literature Summary:**")

    # There are two references section if with_url = True
    # Remove the first section and only keep the second, it's in html.
    first_ref_index = summary.find("References:")
    second_ref_index = summary.rfind("References:")

    summary = summary[:first_ref_index] + summary[second_ref_index:]

    summary = summary.replace("References:", "**References:**")
    return summary


def highlight_summary(summary: str) -> str:
    summary = summary.replace("Summary:", "**Summary:**")
    summary = summary.replace("Study Design:", "**Study Design:**")
    summary = summary.replace("Sample Size:", "**Sample Size:**")
    summary = summary.replace("Study Population:", "**Study Population:**")
    summary = summary.replace("Risk of Bias:", "**Risk of Bias:**")
    summary = summary.replace("Citation:", "**Citation:**")

    return summary
