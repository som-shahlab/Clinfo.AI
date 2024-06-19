import openai
from Bio import Entrez
from Bio.Entrez import efetch, esearch
from typing import List
import string
from concurrent.futures import ThreadPoolExecutor, as_completed

import time
import os
import re
import sys


import sys
import openai
sys.path.append('..')

from utils.prompt_compiler import PromptArchitecture, read_json

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
  
    HumanMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI
from vllm import LLM, SamplingParams

from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages.system import SystemMessage
import pdb
from datetime import datetime, timedelta
from openai import OpenAI

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


class VLM_LLM: 
    def __init__(self,
        model:str,
        params_dict:dict={"temperature":0.8, "top_p":0.95}) -> None: 

        self.llm = LLM(model=model)
        self.sp  = SamplingParams(temperature=0.8, top_p=0.95) 

    def inference(self,prompt:str) -> str:
        output = self.llm.generate(prompt, self.sp) 
        return output[0].outputs[0].text

############ OPP version ############################
class Neural_Retriever_PubMed:
    def __init__(self, 
        architecture_path:str, 
        temperature = 0.5,
        model:str= "gpt-3.5-turbo",
        api_base:str="http://localhost:8000/v1",
        verbose:bool=True, 
        debug:bool=False,
        open_ai_key:str=None,
        email:str=None,
        wait:int=3):

        self.model        = model
        self.verbose      = verbose
        self.architecture = PromptArchitecture(architecture_path=architecture_path,verbose=verbose)
        self.temperature  = temperature
        self.debug        = debug
        self.open_ai_key  = open_ai_key
        self.email        = email
        self.time_out     = 61
        self.wait         = wait
    
        if self.verbose:
            self.architecture.print_architecture()

        if "gpt"  in self.model.lower():
            openai.api_key    = self.open_ai_key
        else:
            print("Trying to init model via VLM")
            self.api_base:str = api_base
            self.time_out     = None
            #self.llm        = VLM_LLM(model="facebook/opt-125m")
            #self.sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
            #pdb.set_trace()
    
    def query_api(self,
                  model:str,
                  prompt:str,
                  temperature:float,
                  max_tokens:int = 1024,
                  n:int     = 1,
                  stop:str  = None,
                  delay:int = None):

    
        if "gpt" in self.model.lower():
            chat = ChatOpenAI(
                    temperature = temperature,
                    model       = model,
                    max_tokens  = max_tokens,
                    n           = n,
                    stop        = stop,
                    request_timeout=delay)
        else:
            chat = ChatOpenAI(
                    temperature = temperature,
                    model       = model,
                    max_tokens  = max_tokens,
                    n           = n,
                    stop        = stop,
                    request_timeout = delay,
                    openai_api_key  = "EMPTY",
                    openai_api_base = self.api_base)


        response = chat(prompt)

        if delay:
            time.sleep(delay)

        query:str = response.content   
        
        return query 

    def generate_pubmed_query(
        self,
        question:str,
        max_tokens:int         = 1024,
        is_reconstruction:bool = False,
        failure_cases          = None) -> str:
        """
        Generates a PubMed query from a clinical question using OpenAI's API

        Args:
            question (str): The clinical question to generate a PubMed query for
            model (str, optional): The OpenAI model to use. Defaults to "gpt-3.5-turbo".
            is_reconstruction (bool, optional): use method for reconstruction. Defaults to False.

        Returns:
            str: The PubMed query
        """
        user_prompt    =  self.architecture.get_prompt("pubmed_query_prompt","template")
        system_prompt  =  self.architecture.get_prompt("pubmed_query_prompt","system").format()

        if self.debug:
            print(f"User prompt: {user_prompt}")
            print(f"System prompt: {system_prompt}")
            
        if is_reconstruction:
            message =  [{"role":"system","content":system_prompt},{"role": "user","content":  user_prompt.format(question=question)}]
            return  message

        else:         
            system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt)
            human_message_prompt  = HumanMessagePromptTemplate(
                                                prompt    = PromptTemplate(template = user_prompt.format(question="{question}"),
                                                input_variables   = ["question"],
                                             ))

            chat_prompt:ChatPromptTemplate  = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
            chat_prompt:list[SystemMessage] = chat_prompt.format_prompt(question=question).to_messages()    
            
            ### Make API CALL ###
            result = self.query_api(
                  model       = self.model ,
                  prompt      = chat_prompt,
                  temperature = self.temperature,
                  max_tokens  = max_tokens,
                  n           = 1,
                  stop        = None,
                  delay       = self.time_out)   

            #pdb.set_trace() 

            return result
                            

    def search_pubmed(self,question: str, num_results: int, num_query_attempts: int = 1,verbose:bool=False,restriction_date=None) -> set:
        """
        Searches PubMed for articles relevant to the given question

        Args:
            question (str): The question to search PubMed for
            num_results (int): The number of results to return
            email (str): The email to use for the Entrez API
            num_query_attempts (int, optional): The number of times to attempt to
                generate a query. Defaults to 1.
        
        Returns:
            A tuple containing the set of search IDs and the set of search queries used
        """
        verbose = True
        failure_cases = None
        Entrez.email   = self.email     
        search_ids     = set()
        search_queries = set()
        for _ in range(num_query_attempts):
            pubmed_query = self.generate_pubmed_query(question,failure_cases = failure_cases)

            if restriction_date != None:
                if self.verbose:
                    print(f"Date Restricted to : {restriction_date}")
                lower_limit  =  subtract_n_years(restriction_date)
                pubmed_query = pubmed_query + f" AND {lower_limit}:{restriction_date}[dp]"

            if verbose:
                print("*"*10)
                print(f"Generated pubmed query: {pubmed_query}\n")

            search_queries.add(pubmed_query)
            # TODO: Debug query '("paxlovid" OR "nirmatrelvir-ritonavir") AND ("COVID-19" OR "SARS-CoV-2") AND ("symptom rebound" OR "relapse" OR "recurrence")' returning 3 articles
            search_results = esearch(db="pubmed",
                                    term=pubmed_query,
                                    retmax=num_results,
                                    sort="relevance")
            try:
                retrived_ids = Entrez.read(search_results)["IdList"]
                search_ids   = search_ids.union(retrived_ids)

                if len(retrived_ids) == 0:
                    failure_cases = pubmed_query

                if verbose:
                    print(f"Retrieved {len(retrived_ids)} IDs")
                    print(retrived_ids)

            except:
                if verbose:
                    failure_cases = pubmed_query

                    print(search_results)
                    print("No IDs retrieved")

            if verbose:
                print(f"Search IDs: {search_ids}")
            
        return list(search_queries), list(search_ids)                            # Return as list for serialization


    def fetch_article_data(self,article_ids: List[str]):  # TODO: What type is article_data?
        """
        Fetches the article data for the given article IDs

        Args:
            article_ids (list): The list of article IDs to fetch data for

        Returns:
            list: The list of article data
        """
        articles     = efetch(db="pubmed", id=article_ids, rettype="xml")
        article_data = Entrez.read(articles)["PubmedArticle"]
        return article_data
     


    def is_article_relevant(self,
        article_text:str, 
        question:str, 
        max_tokens:int  = 512,
        is_reconstruction=False):
        """Returns True if the article is relevant to the query, False otherwise

        Args:
            article_text (str): The text of the article
            query (str): The query to check relevance against
            model (str, optional): The OpenAI model to use. Defaults to "gpt-3.5-turbo".

        Returns:
            bool: True if the article is relevant to the query, False otherwise
        """
        user_prompt   =  self.architecture.get_prompt("relevance_prompt","template")
        system_prompt  =  self.architecture.get_prompt("relevance_prompt","system").format()


        if self.debug:
            print(f"User prompt: {user_prompt}")
            print(f"System prompt: {system_prompt}")

        if is_reconstruction:
            message_ =  [{"role":"system","content":system_prompt}, {"role": "user","content": user_prompt.format(question=question,article_text=article_text)}]
            return message_ 

        else:         
            system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt)
            human_message_prompt  = HumanMessagePromptTemplate(
                                                prompt    = PromptTemplate(template = user_prompt.format(question="{question}",article_text="{article_text}"),
                                                input_variables   = ["question","article_text"],
                                             ))

            ### Make API CALL ###
            chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
            chat_prompt = chat_prompt.format_prompt(question=question,article_text=article_text).to_messages()
            result      = self.query_api(
                model       = self.model ,
                prompt      = chat_prompt,
                temperature = self.temperature,
                max_tokens  = max_tokens,
                n           = 1,
                stop        = None,
                delay       = self.time_out)   

            
          
        
            first_word  = result.split()[0].strip(string.punctuation).lower()
            return first_word not in {"no", "n"}  # TODO


    def construct_citation(self,article):
        """
        Constructs citation for article object fetched via the Biopython package.

        The function attempts to extract the following information from the article object:
        - Author names (last name and initials)
        - Article title
        - Journal title
        - Publication date (year)
        - Journal volume
        - Journal issue
        - Page numbers

        If any of the above information is missing (resulting in a KeyError), the function will
        gracefully handle the error and exclude the missing information from the citation string.

        Args:
            article (dict): A dictionary representing an article object fetched via the Biopython package.

        Returns:
            str: The citation string for the given article, with missing information excluded.

        Example:   # TODO: Fix this example
            >>> article = fetch_article_data(["33100000"])
            >>> construct_citation(article[0])
            'Kumvet al. (2020) A case of paxlovid-induced rebound hypertension in a patient with COVID-19. J Clin Hypertens (Greenwich). 2020;22(12):e1-e3.'
        """
        if len(article["PubmedData"]["ReferenceList"]) == 0 or len(
                article["PubmedData"]["ReferenceList"][0]["Reference"]) == 0:
            # print("### CITATION USING MEDLINECITATION ###")
            return self.generate_ama_citation(article)
        else:
            # print("### CITATION USING REFERENCELIST ###")
            try:
                citation = article["PubmedData"]["ReferenceList"][0]["Reference"][
                    0]["Citation"]
                return citation
            except IndexError as err:
                print(f"IndexError: {err}")


    def generate_ama_citation(self,article):
        """
        Constructs an AMA citation string for a given article object fetched via the Biopython package.
        
        The function attempts to extract the following information from the article object:
        - Author names (last name and initials)
        - Article title
        - Journal title
        - Publication date (year)
        - Journal volume
        - Journal issue
        - Page numbers
        
        If any of the above information is missing (resulting in a KeyError), the function will
        gracefully handle the error and exclude the missing information from the citation string.
        
        Args:
            article (dict): A dictionary representing an article object fetched via the Biopython package.
        
        Returns:
            str: The AMA citation string for the given article, with missing information excluded.
        """
        # TODO: Handle https://pubmed.ncbi.nlm.nih.gov/36126225/ (systematic review)
        try:  # TODO: Construct the author list iteratively and catch errors for each author
            authors = article["MedlineCitation"]["Article"]["AuthorList"]
            author_names = ", ".join([
                f"{author['LastName']} {author['Initials']}" for author in authors
            ])
        except KeyError:  # TODO: Should be able to handle gruops with just first names
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
            volume = article["MedlineCitation"]["Article"]["Journal"][
                "JournalIssue"]["Volume"]
        except KeyError:
            volume = ""

        try:
            issue = article["MedlineCitation"]["Article"]["Journal"][
                "JournalIssue"]["Issue"]
        except KeyError:
            issue = ""

        try:
            pages = article["MedlineCitation"]["Article"]["Pagination"][
                "MedlinePgn"]
        except KeyError:
            pages = ""

        # TODO: Construct the AMA title so that if an element is missing its simply not appended to the string
        return f"{author_names}. {title}. {journal}. {pub_date};{volume}({issue}):{pages}."


    def write_results_to_file(self,filename, ama_citation, summary, append=True):
        """
        Writes the results of the summarization to a file.

        Args:
            filename (str): The name of the file to write the results to.
            ama_citation (str): The AMA citation string for the article.
            summary (str): The summary of the article.
            append (bool): Whether to append the results to the file or overwrite it. Defaults to True.

        Returns:
            None
        """
        mode = "a" if append else "w"
        with open(filename, mode, encoding="utf-8") as f:
            f.write(f"Citation: {ama_citation}\n")
            f.write(f"{summary}")
            f.write("\n###\n\n")


    def reconstruct_abstract(self,abstract_elements):
        """
        Reconstructs the abstract of an article from the abstract elements fetched via the Biopython package.

        Args:
            abstract_elements (list): A list of abstract elements fetched via the Biopython package.

        Returns:
            str: The reconstructed abstract of the article.

        Example:  # TODO: Verify this works
            >>> article = fetch_article_data(["33100000"])
            >>> abstract_elements = article[0]["MedlineCitation"]["Article"]["Abstract"]["AbstractText"]    
            >>> reconstruct_abstract(abstract_elements)
        """
        reconstructed_abstract = ""
        for element in abstract_elements:
            label = element.attributes.get("Label", "")
            if reconstructed_abstract:
                reconstructed_abstract += "\n\n"

            if label:
                reconstructed_abstract += f"{label}:\n"
            reconstructed_abstract += str(element)
        return reconstructed_abstract


    def summarize_study(self,article_text, question,prompt_dict={"type":"automatic"}, model="gpt-3.5-turbo",is_reconstruction=False) -> str:
        """ TASK#: Summarizes the study in the given article text
        Args:
            article_text (str): The text of the article
            question (str): The question to summarize the study for
            model (str, optional): The OpenAI model to use. Defaults to "gpt-3.5-turbo".

        Returns:
            str: The summary of the study
        """
        system_prompt  =  self.architecture.get_prompt("summarization_prompt","system").format()
       
        if prompt_dict["type"] == "automatic":
            user_prompt   =  self.architecture.get_prompt("summarization_prompt","template")
            
        elif prompt_dict["type"] == "Custom":
            user_prompt = prompt_dict["Summary"] 

       
            
        if is_reconstruction:
            message_ =  [{"role":"system","content":system_prompt},{"role": "user","content": user_prompt.format(question=question,article_text=article_text)}]
            return message_

            
        else:         
            system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt)
            human_message_prompt  = HumanMessagePromptTemplate(
                                                prompt    = PromptTemplate(template = user_prompt.format(question="{question}",article_text="{article_text}"),
                                                input_variables   = ["question","article_text"],
                                             ))

            ### Make API CALL ###
            chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
            chat_prompt = chat_prompt.format_prompt(question=question,article_text=article_text).to_messages()
            result = self.query_api(
                model       = self.model ,
                prompt      = chat_prompt,
                temperature = self.temperature,
                max_tokens  = 1024,
                n           = 1,
                stop        = None,
                delay       = self.time_out)   

            return result


        
    def process_article(self,article,question):
        """  Create a helper function for ThreadPoolExecutor"""
     
        try:
        
            #try:
            abstract            = article["MedlineCitation"]["Article"]["Abstract"]["AbstractText"] # Retrivie the abstract
            abstract            = self.reconstruct_abstract(abstract)                    # Reconstruct the abstract
            article_is_relevant = self.is_article_relevant(abstract, question)           # (GPT) Determine if the article is relevant with gpt
            citation            = self.construct_citation(article)                       # Construct the AMA citation
            print(citation)
            print("~" * 10 + f"\n{abstract}")
            print("~" * 10 + f"\nArticle is relevant? = {article_is_relevant}")
            title = article["MedlineCitation"]["Article"]["ArticleTitle"]
            url   = (f"https://pubmed.ncbi.nlm.nih.gov/"
                    f"{article['MedlineCitation']['PMID']}/")
            artical_json =  {
                    "title": title,
                    "url": url,
                    "abstract": abstract,
                    "citation": citation,
                    "is_relevant": article_is_relevant,
                    "PMID": article['MedlineCitation']['PMID']
                }
            
            if article_is_relevant:
                
                summary = self.summarize_study(article_text = abstract, 
                                          question = question, 
                                          model="gpt-3.5-turbo")
                
                artical_json["summary"] = summary
            
            return artical_json 
            
        except KeyError as err:
             if "PMID" in article['MedlineCitation'].keys():
                 print(f"Could not find {err} for article with PMID = "
                     f"{article['MedlineCitation']['PMID']}")
             else:
                 print("Error retrieving article data:", err)
             return None
        except ValueError as err:
             print("Error: ", err)  # TODO: Handle this better
             return None


    def summarize_each_article(self,articles, question, num_workers=8):
        relevant_article_summaries  = []
        irelevant_article_summaries = []

        # Use ThreadPoolExecutor to process articles in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all articles for processing
            futures = [
                        executor.submit(self.process_article, article,question) for article in articles
            ]

            # Collect results as they become available
            for future in as_completed(futures):
                try:
                    result = future.result()
                except:
                    pass
                    print("Error processing article. Server is probably overloaded. waiting 10 seconds")
                    time.sleep(20)
                    print("Lets try agian")
                    result = future.result()
                ### Organize results into relevant and irrelevant articles
                try:
                    if result["is_relevant"]:
                        relevant_article_summaries.append(result)
                    else:
                        irelevant_article_summaries.append(result)
                except:
                    pass
            
        return relevant_article_summaries, irelevant_article_summaries



    def build_citations_and_summaries(self,article_summaries:dict,with_url:bool=False) -> tuple:
        """ Structures citations and summaries in a readable format for gpt-3.5-turbo
            Input:
                article_summaries: list of dictionaries with keys: citation, summary
            Output: 
                article_summaries_with_citations: str, 
                citations: str
        """
        article_summaries_with_citations = []
        citations                        = []
        for i,summary in enumerate(article_summaries):
            citation = re.sub(r'\n', '',summary['citation'])
            article_summaries_with_citations.append(f"[{i+1}] Source: {citation}\n\n\n {summary['summary']}")
            citation_with_index = f"[{i+1}] {citation }"
            if with_url:
                citation_with_index = f"<li><a href=\"{summary['url']}\"   target=\"_blank\"> {citation_with_index}</a></li>"

            citations.append(citation_with_index)
        article_summaries_with_citations = "\n\n--------------------------------------------------------------\n\n".join(article_summaries_with_citations )

        citations   = "\n".join(citations)

        if with_url:
            citations = f"<ul>{citations}</ul>"
         
        return article_summaries_with_citations,citations
    
    

    def synthesize_all_articles(self,summaries, question,prompt_dict={"type":"automatic"},  model="gpt-4",is_reconstruction=False,with_url=False):
        article_summaries_str,citations = self.build_citations_and_summaries(article_summaries = summaries,  with_url = with_url) 
 
        system_prompt  =  self.architecture.get_prompt("synthesize_prompt","system").format()

        if prompt_dict["type"] == "automatic":
            user_prompt   =  self.architecture.get_prompt("synthesize_prompt","template")
            
        elif prompt_dict["type"] == "Custom":
            user_prompt = prompt_dict["Synthesis"] 

        if self.debug:
            print(f"User prompt: {user_prompt}")
            print(f"System prompt: {system_prompt}")
            
        if is_reconstruction:
            message_ =  [{"role":"system","content":system_prompt}, {"role": "user","content": user_prompt.format(question=question,article_summaries_str=article_summaries_str)}]
            return message_ 
        
        else:         
            system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt)
            human_message_prompt  = HumanMessagePromptTemplate(
                                                prompt    = PromptTemplate(template = user_prompt.format(question="{question}",article_summaries_str="{article_summaries_str}"),
                                                input_variables   = ["question","article_summaries_str"],
                                             ))

            chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
            chat_prompt = chat_prompt.format_prompt(question=question,article_summaries_str=article_summaries_str).to_messages()
            result = self.query_api(
                model       = self.model ,
                prompt      = chat_prompt,
                temperature = self.temperature,
                max_tokens  = 1024,
                n           = 1,
                stop        = None,
                delay       = self.time_out)  
    
            if with_url:
                result = result  + "\n\n" + "References:\n" +  citations

            return result
        
    
    def PIPE_LINE(self,question:str):
        """ This runs the entire pipeline, intended for testing"""
        ## Step 1: Search PubMed ###
        # Convert the question into a query using gpt 
        # This returns a list of queries (used to retrive articles) and a list of article ids that were retrieved
        pubmed_queries, article_ids = self.search_pubmed(question,num_results=4,num_query_attempts=1)

        ## Step 1.a: Fetch article data
        #  Convert  list of Ids into a list of dictionaries (populated by pumbed API)
        articles = self.fetch_article_data(article_ids)

        ###  STEP 2 Summarize each article (only if they are relevant [Step 3]) ###
        article_summaries,irrelevant_articles =  self.summarize_each_article(articles, question)


        ### STEP 4: Synthesize the results ###
        synthesis =   self.synthesize_all_articles(article_summaries, question)

        return synthesis, article_summaries, irrelevant_articles, articles, article_ids, pubmed_queries,


    def reconstruct_relevant_helper(self,dict_,question):
        for id  in range(0,len(dict_)):
            dict_[id]["relevant_prompt"] = self.is_article_relevant( dict_[id]["abstract"], question,is_reconstruction=True)

    def reconstruct_summary_helper(self,dict_,question):
        for id  in range(0,len(dict_)):
            dict_[id]["summary_prompt"] = self.summarize_study(dict_[id]["abstract"], question, is_reconstruction=True) 

    def reconstruct_from_json_pubmed_arch_1(self,json_: dict):

        ### Step 1 : Generate the query
        question = json_["INPUT"]["question"]
        query_generation_message = self.generate_pubmed_query(question,is_reconstruction=True) 
        
        # Step 2: Summarize each article
        #json_["ARTICLES"]["relevant"]
        #json_["ARTICLES"]["irrelevant"]

        self.reconstruct_relevant_helper(json_["ARTICLES"]["relevant"],question)
        self.reconstruct_summary_helper(json_["ARTICLES"]["relevant"],question)
        self.reconstruct_relevant_helper(json_["ARTICLES"]["irrelevant"],question)

        ### Step_3 
        synthesize_message = self.synthesize_all_articles(json_["ARTICLES"]["relevant"], question, model="gpt-4",is_reconstruction=True)
        synthesis           = json_["OUTPUT"]

        reconstruction = {}
        reconstruction["question"]            = question
        reconstruction["query_prompt"]        = query_generation_message
        reconstruction["pumed_query"]         = json_["SEACH_QUERY"]["pumed_query"]
        reconstruction["articles_ids"]        = json_["SEACH_QUERY"]["articles_ids"]
        reconstruction["relevant_articles"]   = json_["ARTICLES"]["relevant"]
        reconstruction["irrelevant_articles"] = json_["ARTICLES"]["irrelevant"]
        reconstruction["synthesize_message"]  = synthesize_message
        reconstruction["synthesis"]           = synthesis

        return reconstruction
    



    ############################# PRINTING FUNCTIONS #############################
    def print_double_api_call(self,recon):
        for i,r_ in enumerate(recon):
            print(f"\n{i+1}.-TITLE: {r_['title']}")
            self.print_api_call(r_['relevant_prompt'])
            is_relevant = r_['is_relevant']
            print(f"RELEVANT? {is_relevant}")

            if is_relevant:
                self.print_api_call(r_['summary_prompt'])
                print(r_['summary'])
            print()

    def print_api_call(self,recon):
        print("\n######################## GPT-API CALL #########################")
        for message in recon:
            print()
            print(f"{message['role']}:")
            print()
            print(message['content'])
        print("###############################################################\n")


    def print_architecture_v1(self,reconstruction):
        print(f"USER: {reconstruction['question']}\n")
        print("\nSTEP 1: GETTING PUBMED QUERIES FROM GPT-3\n")
        self.print_api_call(reconstruction['query_prompt'][0])
        print("--------------------------------------------------------------------------------------------------------")
        print(f"GPT  Queries: { reconstruction['pumed_query']}")
        print(f"PUBMED IDS:   { reconstruction['articles_ids']}")
        print("--------------------------------------------------------------------------------------------------------")
        print("\nSTEP 2: RETRIVING PAPERS FROM PUBMED LOOKING FOR RELEVANT ARTICLES IF RELEVANT SUMMARIZE\n")
        self.print_double_api_call(reconstruction['irrelevant_articles'])
        self.print_double_api_call(reconstruction['relevant_articles'])

        print("\nSTEP 3: SYNTHESIZE THE ANSWER\n")
        self.print_api_call(reconstruction["synthesize_message"])
        print('""""""""""""""""""""""""""""""""""""""""""" ANSWER """"""""""""""""""""""""""""""""""""""')
        print(reconstruction["synthesis"])
        print('"'*20)