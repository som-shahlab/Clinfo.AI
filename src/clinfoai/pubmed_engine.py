import os
import re
import sys
import string
import time
from pathlib import Path
from typing import List
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from Bio import Entrez
from Bio.Entrez import efetch, esearch
from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages.system import SystemMessage
from vllm import LLM, SamplingParams

import openai
from   openai import OpenAI
import pdb

import google.generativeai as genai

sys.path.append(str(Path(__file__).resolve().parent))
from utils.prompt_compiler import PromptArchitecture, read_json
from dense_search import generate_paths,PubMedDenseSearch


def subtract_n_years(date_str:str,n:int=20) ->str:
    date     = datetime.strptime(date_str, "%Y/%m/%d")  # Parse the given date string
    new_year = date.year - n                            # Subtract n years

    # Check if the resulting year is a leap year
    is_leap_year = (new_year % 4 == 0 and new_year % 100 != 0) or new_year % 400 == 0

    # Adjust the day value if necessary
    new_day = date.day
    if date.month == 2 and date.day == 29 and not is_leap_year:
        new_day = 28

    # Create a new date with the updated year, month, and day
    new_date = datetime(new_year, date.month, new_day)

    # Format the new date to the desired format (YYYY/MM/DD)
    formatted_date = new_date.strftime("%Y/%m/%d")
    return formatted_date


class Gemini_LLM:
    def __init__(self, model_name: str):
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self.model = genai.GenerativeModel(model_name=model_name)

    def inference(self, prompt: str, generation_config=None) -> str:
        response = self.model.generate_content(prompt, generation_config=generation_config)
        if not response.candidates or not response.candidates[0].content or not response.candidates[0].content.parts:
            raise ValueError("Invalid operation: The response contains no valid candidates or content parts.")
        return response.candidates[0].content.parts[0].text

class PubMedNeuralRetriever:
    def __init__(
        self, 
        architecture_path: str, 
        temperature:float=0.5, 
        model:str = "gpt-3.5-turbo",
        dense_search:bool=False,
        verbose: bool = True, 
        debug: bool = False, 
        open_ai_key: str = None, 
        email: str = None, 
        wait: int = 3):

        self.model = model
        self.verbose = verbose
        self.architecture = PromptArchitecture(architecture_path=architecture_path, verbose=verbose)
        self.temperature = temperature
        self.debug = debug
        self.open_ai_key = open_ai_key
        self.email = email
        self.time_out = 61
        self.delay = 2
        self.wait = wait
        self.dense_search = dense_search
        
        if dense_search:
            base_dir = Path("/pasteur", "data", "PubMed")
            embeddings_paths, pmids_paths = generate_paths(base_dir,init_chunk=10,end_chunks=36)
            print("Initalizing Index, this might take some time")
            self.pubme_dense_retrivier = PubMedDenseSearch(
                pubmed_embeds_files = embeddings_paths,
                pmids_files = pmids_paths,
                index_file  = "/pasteur/data/PubMed/pubmed.index")

        if self.verbose:
            self.architecture.print_architecture()

        if "gpt" in self.model.lower():
            openai.api_key = self.open_ai_key

        elif "gemini" in self.model.lower():
            print("Using Gemini model")

        else:
            print("Trying to init model via VLM")
            self.api_base:str = api_base
            self.time_out     = None
            self.delay        = None
    
    def query_api(self, model: str, prompt: list, temperature: float, max_tokens: int = 1024, 
                  n: int = 1, stop: str = None, delay: int = None):

        response = None  # Initialize response
        query = None     # Initialize query

        if "gpt" in self.model.lower():
            chat = ChatOpenAI(
                temperature=temperature,
                model=model,
                max_tokens=max_tokens,
                n=n,
                request_timeout=self.time_out)

            response = chat(prompt)
            query = response.content
            
        elif "gemini" in self.model.lower():
            generation_config = genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )
            # Concatenate messages into a single string
            prompt_str = " ".join([msg.content for msg in prompt])
            chat = Gemini_LLM(model_name=model)
            response = chat.inference(prompt_str, generation_config=generation_config)
            query = response  # The response from Gemini is directly the text content
            
        else:
            chat = ChatOpenAI(
                temperature=temperature,
                model=model,
                max_tokens=max_tokens,
                n=n,
                request_timeout=delay,
                openai_api_key="EMPTY",
                openai_api_base=self.api_base)
            response = chat(prompt)
            query = response.content

        if self.delay:
            time.sleep(self.delay)

        return query

    def generate_pubmed_query(self, question: str, max_tokens: int = 1024, is_reconstruction: bool = False, failure_cases = None) -> str:
        user_prompt = self.architecture.get_prompt("pubmed_query_prompt", "template")
        system_prompt = self.architecture.get_prompt("pubmed_query_prompt", "system").format()

        if self.debug:
            print(f"User prompt: {user_prompt}")
            print(f"System prompt: {system_prompt}")
            
        if is_reconstruction:
            message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt.format(question=question)}]
            return message
        else:         
            system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt)
            human_message_prompt = HumanMessagePromptTemplate(
                prompt=PromptTemplate(template=user_prompt.format(question="{question}"), input_variables=["question"]))

            chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
            chat_prompt = chat_prompt.format_prompt(question=question).to_messages()    
            
            result = self.query_api(
                model=self.model,
                prompt=chat_prompt,
                temperature=self.temperature,
                max_tokens=max_tokens,
                n=1,
                stop=None,
                delay=self.time_out)   

            return result

    def search_pubmed(
        self, 
        question: str, 
        num_results: int = 10, 
        num_query_attempts: int = 1, 
        verbose: bool = False, 
        restriction_date = None) -> list[list[str], list[str]]:
        
        failure_cases = None
        Entrez.email = self.email     
        search_ids = set()
        search_queries = set()

      
        if self.dense_search:
            search_ids = []
            search_queries = question
            results = self.pubme_dense_retrivier.search([question],k=num_results)
            for result in results:
                for res in result['Results']:
                    search_ids.append(res['PMID'])

    

        elif self.dense_search == False:
            for _ in range(num_query_attempts):
                pubmed_query = self.generate_pubmed_query(question, failure_cases=failure_cases)

                if restriction_date != None:
                    if self.verbose:
                        print(f"Date Restricted to : {restriction_date}")
                    lower_limit = subtract_n_years(restriction_date)
                    pubmed_query = pubmed_query + f" AND {lower_limit}:{restriction_date}[dp]"

                if verbose:
                    print("*" * 10)
                    print(f"Generated pubmed query: {pubmed_query}\n")

                search_queries.add(pubmed_query)
                search_results = esearch(db="pubmed", term=pubmed_query, retmax=num_results, sort="relevance")
                try:
                    retrieved_ids = Entrez.read(search_results)["IdList"]
                    search_ids = search_ids.union(retrieved_ids)

                    if len(retrieved_ids) == 0:
                        failure_cases = pubmed_query

                    if verbose:
                        print(f"Retrieved {len(retrieved_ids)} IDs")
                        print(retrieved_ids)

                except:
                    if verbose:
                        failure_cases = pubmed_query
                        print(search_results)
                        print("No IDs retrieved")

            if verbose:
                print(f"Search IDs: {search_ids}")
            
        return list(search_queries), list(search_ids)

    def fetch_article_data(self, article_ids: List[str]):
        articles = efetch(db="pubmed", id=article_ids, rettype="xml")
        article_data = Entrez.read(articles)["PubmedArticle"]
        return article_data

    def is_article_relevant(self, article_text: str, question: str, max_tokens: int = 512, is_reconstruction = False):
        user_prompt = self.architecture.get_prompt("relevance_prompt", "template")
        system_prompt = self.architecture.get_prompt("relevance_prompt", "system").format()

        if self.debug:
            print(f"User prompt: {user_prompt}")
            print(f"System prompt: {system_prompt}")

        if is_reconstruction:
            message_ = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt.format(question=question, article_text=article_text)}]
            return message_
        else:         
            system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt)
            human_message_prompt = HumanMessagePromptTemplate(
                prompt=PromptTemplate(template=user_prompt.format(question="{question}", article_text="{article_text}"), input_variables=["question", "article_text"]))

            chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
            chat_prompt = chat_prompt.format_prompt(question=question, article_text=article_text).to_messages()
            
            result = self.query_api(
                model=self.model,
                prompt=chat_prompt,
                temperature=self.temperature,
                max_tokens=max_tokens,
                n=1,
                stop=None,
                delay=self.time_out)   

            first_word = result.split()[0].strip(string.punctuation).lower()
            return first_word not in {"no", "n"}

    def construct_citation(self, article):
        if len(article["PubmedData"]["ReferenceList"]) == 0 or len(article["PubmedData"]["ReferenceList"][0]["Reference"]) == 0:
            return self.generate_ama_citation(article)
        else:
            try:
                citation = article["PubmedData"]["ReferenceList"][0]["Reference"][0]["Citation"]
                return citation
            except IndexError as err:
                print(f"IndexError: {err}")

    def generate_ama_citation(self, article):
        try:
            authors = article["MedlineCitation"]["Article"]["AuthorList"]
            author_names = ", ".join([f"{author['LastName']} {author['Initials']}" for author in authors])
        except KeyError:
            author_names = ""

        try:
            title = article["MedlineCitation"]["Article"]["ArticleTitle"]
        except KeyError:
            title = ""

        try:
            journal = article["MedlineCitation"]["Article"]["Journal"]["Title"]
        except KeyError:
            journal = ""

        try:
            pub_date = article["PubmedData"]["History"][0]["Year"]
        except KeyError:
            pub_date = ""

        try:
            volume = article["MedlineCitation"]["Article"]["Journal"]["JournalIssue"]["Volume"]
        except KeyError:
            volume = ""

        try:
            issue = article["MedlineCitation"]["Article"]["Journal"]["JournalIssue"]["Issue"]
        except KeyError:
            issue = ""

        try:
            pages = article["MedlineCitation"]["Article"]["Pagination"]["MedlinePgn"]
        except KeyError:
            pages = ""

        return f"{author_names}. {title}. {journal}. {pub_date};{volume}({issue}):{pages}."

    def write_results_to_file(self, filename, ama_citation, summary, append=True):
        mode = "a" if append else "w"
        with open(filename, mode, encoding="utf-8") as f:
            f.write(f"Citation: {ama_citation}\n")
            f.write(f"{summary}")
            f.write("\n###\n\n")

    def reconstruct_abstract(self, abstract_elements):
        reconstructed_abstract = ""
        for element in abstract_elements:
            label = element.attributes.get("Label", "")
            if reconstructed_abstract:
                reconstructed_abstract += "\n\n"

            if label:
                reconstructed_abstract += f"{label}:\n"
            reconstructed_abstract += str(element)
        return reconstructed_abstract

    def summarize_study(self, article_text, question, prompt_dict={"type": "automatic"}, model="gemini-1.5-flash", is_reconstruction=False) -> str:
        system_prompt = self.architecture.get_prompt("summarization_prompt", "system").format()
       
        if prompt_dict["type"] == "automatic":
            user_prompt = self.architecture.get_prompt("summarization_prompt", "template")
        elif prompt_dict["type"] == "Custom":
            user_prompt = prompt_dict["Summary"]

        if is_reconstruction:
            message_ = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt.format(question=question, article_text=article_text)}]
            return message_
        else:         
            system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt)
            human_message_prompt = HumanMessagePromptTemplate(
                prompt=PromptTemplate(template=user_prompt.format(question="{question}", article_text="{article_text}"), input_variables=["question", "article_text"]))

            chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
            chat_prompt = chat_prompt.format_prompt(question=question, article_text=article_text).to_messages()
            result = self.query_api(
                model=self.model,
                prompt=chat_prompt,
                temperature=self.temperature,
                max_tokens=1024,
                n=1,
                stop=None,
                delay=self.time_out)

            return result

    def process_article(self, article, question):
        try:
            abstract = article["MedlineCitation"]["Article"]["Abstract"]["AbstractText"]
            abstract = self.reconstruct_abstract(abstract)
            article_is_relevant = self.is_article_relevant(abstract, question)
            citation = self.construct_citation(article)
            if self.verbose:
                print(citation)
                print("~" * 10 + f"\n{abstract}")
                print("~" * 10 + f"\nArticle is relevant? = {article_is_relevant}")
            
            title = article["MedlineCitation"]["Article"]["ArticleTitle"]
            url = (f"https://pubmed.ncbi.nlm.nih.gov/"
                   f"{article['MedlineCitation']['PMID']}/")
            article_json = {
                "title": title,
                "url": url,
                "abstract": abstract,
                "citation": citation,
                "is_relevant": article_is_relevant,
                "PMID": article['MedlineCitation']['PMID']
            }
            
            if article_is_relevant:
                summary = self.summarize_study(article_text=abstract, question=question, model="gemini-1.5-flash")
                article_json["summary"] = summary
            
            return article_json
        except KeyError as err:
            if "PMID" in article['MedlineCitation'].keys():
                print(f"Could not find {err} for article with PMID = {article['MedlineCitation']['PMID']}")
            else:
                print("Error retrieving article data:", err)
            return None
        except ValueError as err:
            print("Error: ", err)
            return None

    def summarize_each_article(self, articles, question, num_workers=8):
        relevant_article_summaries = []
        irrelevant_article_summaries = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.process_article, article, question) for article in articles]
            for future in as_completed(futures):
                try:
                    result = future.result()
                except:
                    print("Error processing article. Server is probably overloaded. waiting 10 seconds")
                    time.sleep(20)
                    print("Lets try agian")
                    result = future.result()
                try:
                    if result["is_relevant"]:
                        relevant_article_summaries.append(result)
                    else:
                        irrelevant_article_summaries.append(result)
                except:
                    pass
            
        return relevant_article_summaries, irrelevant_article_summaries

    def build_citations_and_summaries(self, article_summaries: dict, with_url: bool = False) -> tuple:
        article_summaries_with_citations = []
        citations = []
        for i, summary in enumerate(article_summaries):
            citation = re.sub(r'\n', '', summary['citation'])
            article_summaries_with_citations.append(f"[{i+1}] Source: {citation}\n\n\n {summary['summary']}")
            citation_with_index = f"[{i+1}] {citation }"
            if with_url:
                citation_with_index = f"<li><a href=\"{summary['url']}\"   target=\"_blank\"> {citation_with_index}</a></li>"

            citations.append(citation_with_index)
        article_summaries_with_citations = "\n\n--------------------------------------------------------------\n\n".join(article_summaries_with_citations)

        citations = "\n".join(citations)

        if with_url:
            citations = f"<ul>{citations}</ul>"
         
        return article_summaries_with_citations, citations
    
    def synthesize_all_articles(self, summaries, question, prompt_dict={"type": "automatic"}, model="gemini-1.5-flash", is_reconstruction=False, with_url=False):
        article_summaries_str, citations = self.build_citations_and_summaries(article_summaries=summaries, with_url=with_url)
 
        system_prompt = self.architecture.get_prompt("synthesize_prompt", "system").format()

        if prompt_dict["type"] == "automatic":
            user_prompt = self.architecture.get_prompt("synthesize_prompt", "template")
        elif prompt_dict["type"] == "Custom":
            user_prompt = prompt_dict["Synthesis"]

        if self.debug:
            print(f"User prompt: {user_prompt}")
            print(f"System prompt: {system_prompt}")
            
        if is_reconstruction:
            message_ = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt.format(question=question, article_summaries_str=article_summaries_str)}]
            return message_
        else:         
            system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt)
            human_message_prompt = HumanMessagePromptTemplate(
                prompt=PromptTemplate(template=user_prompt.format(question="{question}", article_summaries_str="{article_summaries_str}"), input_variables=["question", "article_summaries_str"]))

            chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
            chat_prompt = chat_prompt.format_prompt(question=question, article_summaries_str=article_summaries_str).to_messages()
            result = self.query_api(
                model=self.model,
                prompt=chat_prompt,
                temperature=self.temperature,
                max_tokens=1024,
                n=1,
                stop=None,
                delay=self.time_out)
    
            if with_url:
                result = result + "\n\n" + "References:\n" + citations

            return result

    def PIPE_LINE(self, question: str):
        pubmed_queries, article_ids = self.search_pubmed(question, num_results=4, num_query_attempts=1)
        articles = self.fetch_article_data(article_ids)
        article_summaries, irrelevant_articles = self.summarize_each_article(articles, question)
        synthesis = self.synthesize_all_articles(article_summaries, question)

        return synthesis, article_summaries, irrelevant_articles, articles, article_ids, pubmed_queries

    def reconstruct_relevant_helper(self, dict_, question):
        for id in range(0, len(dict_)):
            dict_[id]["relevant_prompt"] = self.is_article_relevant(dict_[id]["abstract"], question, is_reconstruction=True)

    def reconstruct_summary_helper(self, dict_, question):
        for id in range(0, len(dict_)):
            dict_[id]["summary_prompt"] = self.summarize_study(dict_[id]["abstract"], question, is_reconstruction=True)

    def reconstruct_from_json_pubmed_arch_1(self, json_: dict):
        question = json_["INPUT"]["question"]
        query_generation_message = self.generate_pubmed_query(question, is_reconstruction=True)
        
        self.reconstruct_relevant_helper(json_["ARTICLES"]["relevant"], question)
        self.reconstruct_summary_helper(json_["ARTICLES"]["relevant"], question)
        self.reconstruct_relevant_helper(json_["ARTICLES"]["irrelevant"], question)

        synthesize_message = self.synthesize_all_articles(json_["ARTICLES"]["relevant"], question, model="gemini-1.5-flash", is_reconstruction=True)
        synthesis = json_["OUTPUT"]

        reconstruction = {}
        reconstruction["question"] = question
        reconstruction["query_prompt"] = query_generation_message
        reconstruction["pumed_query"] = json_["SEACH_QUERY"]["pumed_query"]
        reconstruction["articles_ids"] = json_["SEACH_QUERY"]["articles_ids"]
        reconstruction["relevant_articles"] = json_["ARTICLES"]["relevant"]
        reconstruction["irrelevant_articles"] = json_["ARTICLES"]["irrelevant"]
        reconstruction["synthesize_message"] = synthesize_message
        reconstruction["synthesis"] = synthesis

        return reconstruction

    def print_double_api_call(self, recon):
        for i, r_ in enumerate(recon):
            print(f"\n{i+1}.-TITLE: {r_['title']}")
            self.print_api_call(r_['relevant_prompt'])
            is_relevant = r_['is_relevant']
            print(f"RELEVANT? {is_relevant}")

            if is_relevant:
                self.print_api_call(r_['summary_prompt'])
                print(r_['summary'])
            print()

    def print_api_call(self, recon):
        print("\n######################## GPT-API CALL #########################")
        for message in recon:
            print()
            print(f"{message['role']}:")
            print()
            print(message['content'])
        print("###############################################################\n")

    def print_architecture_v1(self, reconstruction):
        print(f"USER: {reconstruction['question']}\n")
        print("\nSTEP 1: GETTING PUBMED QUERIES FROM GPT-3\n")
        self.print_api_call(reconstruction['query_prompt'][0])
        print("--------------------------------------------------------------------------------------------------------")
        print(f"GPT  Queries: {reconstruction['pumed_query']}")
        print(f"PUBMED IDS:   {reconstruction['articles_ids']}")
        print("--------------------------------------------------------------------------------------------------------")
        print("\nSTEP 2: RETRIVING PAPERS FROM PUBMED LOOKING FOR RELEVANT ARTICLES IF RELEVANT SUMMARIZE\n")
        self.print_double_api_call(reconstruction['irrelevant_articles'])
        self.print_double_api_call(reconstruction['relevant_articles'])

        print("\nSTEP 3: SYNTHESIZE THE ANSWER\n")
        self.print_api_call(reconstruction["synthesize_message"])
        print('""""""""""""""""""""""""""""""""""""""""""" ANSWER """"""""""""""""""""""""""""""""""""""')
        print(reconstruction["synthesis"])
        print('"' * 20)
