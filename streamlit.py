import os
import streamlit as st

from src.clinfoai.pubmed_engine import PubMedNeuralRetriever

# Make Sure you followed at least step 1-2 before running this cell.
from config import OPENAI_API_KEY, NCBI_API_KEY, EMAIL, GOOGLE_API_KEY

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


def main():
    st.set_page_config(
        page_title="HealthLight: Medical Q&A from Scientific Literature",
        page_icon="📗",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Header
    st.title("Answer Medical Questions from Latest Scientific Literature.")

    # Sidebar
    # with st.sidebar:
    #     search_engine = st.selectbox("Search Engine:", ["PubMed"])

    # Main content
    question = st.text_input("Enter your medical question:", key="question")

    if st.button("Search"):
        with st.spinner(
            "Searched PubMed for articles related to your question. Retrieving results..."
        ):
            articles = search_articles(question)

        with st.spinner(
            f"Retrieved {len(articles)} articles. Identifying the relevant ones and summarizing them (this may take a minute)."
        ):
            ### STEP 3 Summarize each article (only if they are relevant [Step 3]) ###
            article_summaries, irrelevant_articles = nrpm.summarize_each_article(
                articles, question
            )
        with st.spinner("Synthesizing the results..."):
            synthesis = nrpm.synthesize_all_articles(
                article_summaries, question, with_url=True
            )

        st.success("Search completed!")

        # Display sample result (you would replace this with actual search results)
        translate_synthesis = nrpm.translate_en_to_vn(synthesis)
        st.markdown(translate_synthesis, unsafe_allow_html=True)
        st.markdown(highlight_answer(synthesis), unsafe_allow_html=True)

        for article in article_summaries:
            with st.expander(f"[{article["title"]}]({article["url"]})"):
                # translate_article = nrpm.translate_en_to_vn(
                #     highlight_summary(article["summary"])
                # )
                # st.markdown(translate_article, unsafe_allow_html=True)
                st.markdown(highlight_summary(article["summary"]), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
