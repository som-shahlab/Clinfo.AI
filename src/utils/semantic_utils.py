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

from utils import semantic_scholar_driver



from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
  
    HumanMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI

from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate

############ OPP version ############################
class Neural_Retriever_Semantic_Scholar:
    def __init__(self, architecture_path:str, temperature = 0.5, verbose:bool=True, debug:bool=False,open_ai_key:str=None,email:str=None):
        self.verbose      = verbose
        self.architecture = PromptArchitecture(architecture_path=architecture_path,verbose=verbose)
        self.temperature  = temperature
        self.debug        = debug
        self.open_ai_key  = open_ai_key
        self.email        = email
        openai.api_key    = self.open_ai_key
        self.time_out     = 61
    
       
        if self.verbose:
            self.architecture.print_architecture()

    def generate_semantic_query(self,question: str, model: str = "gpt-3.5-turbo",is_reconstruction=False,failure_cases =None) -> str:
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
            message_ =  [{"role":"system","content":system_prompt},{"role": "user","content":  user_prompt.format(question=question)}]
            return  message_

        else:         
            system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt)
            human_message_prompt  = HumanMessagePromptTemplate(
                                                prompt    = PromptTemplate(template = user_prompt.format(question="{question}"),
                                                input_variables   = ["question"],
                                             ))

            ### Make API CALL ###
            chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
            chat        = ChatOpenAI(temperature = self.temperature,
                                      model      = model,
                                      max_tokens = 1024,
                                      n          = 1,
                                      stop       = None,
                                      request_timeout=self.time_out  
                                                  
                                     )
            response    = chat(chat_prompt.format_prompt(question=question).to_messages())

            self.query = response.content
            self.query =  self.query.replace("-"," ")

            return self.query
        
    def search_semantic_scholar(self,query,limit:int=50,threshold = 10,minimum_return=5,maxmum_return=15,verbose=True):
        fields       = ["title","url","abstract","tldr","citationStyles","paperId","paperId","venue","year","authors","externalIds","abstract","publicationVenue","influentialCitationCount","citationCount","fieldsOfStudy","tldr"]
        SemanticAPI  = semantic_scholar_driver.SemanticScholarAPI(limit= limit ,fields=fields)
        articles     = SemanticAPI.search_with_filter(query,threshold= threshold,minimum_return=minimum_return,maxmum_return=maxmum_return,verbose=verbose)
        return articles


    def is_article_relevant(self,article_text:str, question:str, model="gpt-4",is_reconstruction=False):
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
            chat        = ChatOpenAI(temperature = self.temperature,
                                      model      = model,
                                      max_tokens = 512,
                                      n          = 1,
                                      stop       = None,
                                      request_timeout=self.time_out  
                                                  
                                     )
            response    = chat(chat_prompt.format_prompt(question=question,article_text=article_text).to_messages())
            answer      = response.content
            first_word  = answer.split()[0].strip(string.punctuation).lower()
            return first_word not in {"no", "n"}  # TODO


   
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
        
        try:  
            authors = article[ 'authors']
            author_names = ", ".join([author['name'] for author in authors])
        except KeyError:  # TODO: Should be able to handle gruops with just first names
            author_names = ""

        try:
            title = article["title"]
        except KeyError:
            title = ""

        try:
            journal = article['venue']
        except KeyError:
            journal = ""

        try:
            pub_date = article['year']
        except KeyError:
            pub_date = ""

        return f"{author_names}. {title} {journal}. {pub_date};."


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

    def summarize_study(self,article_text, question, model="gpt-3.5-turbo",prompt_dict={"type":"automatic"},is_reconstruction=False) -> str:
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


        print("user_prompt",user_prompt)

        if self.debug:
            print(f"User prompt: {user_prompt}")
            print(f"System prompt: {system_prompt}")
       
            
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
            chat        = ChatOpenAI(temperature = self.temperature,
                                      model      = model,
                                      max_tokens = 1024,
                                      n          = 1,
                                      stop       = None,
                                      request_timeout=self.time_out  
                                                  
                                     )
            response    = chat(chat_prompt.format_prompt(question=question,article_text=article_text).to_messages())
            answer      = response.content
            return answer


        
    def process_article(self,article,question,prompt_dict,verbose=False,add_tldr=False):
        """  Create a helper function for ThreadPoolExecutor"""
        try:
            if add_tldr:
                tldr = get_dict_value(dct = article, key = "tldr")
                tldr = get_dict_value(dct = tldr, key = "text")
                if tldr is not None:
                    article["abstract"] = article["abstract"] + "\nTLDR:\n" + tldr
                    
            abstract            = article["abstract"]                          # Get articles 
            article_is_relevant = self.is_article_relevant(abstract, question) # (GPT) Determine if the article is relevant with gpt
            citation            = self.generate_ama_citation(article)          # Construct the AMA citation
            title               = article["title"]
            url                 = article["url"]
            if verbose:
                print(citation)
                print("~" * 10 + f"\n{abstract}")
                print("~" * 10 + f"\nArticle is relevant? = {article_is_relevant}")
           
        
            artical_json =  {"title":       title,
                             "url":         url,
                             "abstract":    abstract,
                             "citation":    citation,
                             "is_relevant": article_is_relevant}
            
            if article_is_relevant:
                summary                 = self.summarize_study(article_text = abstract, question = question, model="gpt-3.5-turbo",prompt_dict=prompt_dict)
                artical_json["summary"] = summary
            
            return artical_json

        
        except ValueError as err:
             print("Error: ", err)  # TODO: Handle this better
             return None


    def summarize_each_article(self,articles, question,prompt_dict, num_workers=8):
        relevant_article_summaries  = []
        irelevant_article_summaries = []

        # Use ThreadPoolExecutor to process articles in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all articles for processing
            futures = [
                        executor.submit(self.process_article, article,question,prompt_dict) for article in articles
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
    


    def synthesize_all_articles(self,summaries, question,prompt_dict={"type":"automatic"}, model="gpt-4",is_reconstruction=False,with_url:bool=False):
        article_summaries_str,citations = self.build_citations_and_summaries(article_summaries = summaries,  with_url = with_url) 
       
        
        system_prompt  =  self.architecture.get_prompt("synthesize_prompt","system").format()

        if prompt_dict["type"] == "automatic":
            user_prompt   =  self.architecture.get_prompt("synthesize_prompt","template")
            
        elif prompt_dict["type"] == "Custom":
            user_prompt = prompt_dict["Synthesis"] 

        if self.debug:
            print(f"User prompt: {user_prompt}")
            print(f"System prompt: {system_prompt}")

        print("user_prompt",user_prompt)
            
  
            
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
            chat        = ChatOpenAI(temperature = self.temperature,
                                      model      = model,
                                      max_tokens = 1024,
                                      n          = 1,
                                      stop       = None,
                                      request_timeout=600 
                                                  
                                     )
            response    = chat(chat_prompt.format_prompt(question=question,article_summaries_str=article_summaries_str).to_messages())
            synthesis   = response.content
            print("=#" * 20)
            print(synthesis)

            if with_url:
                synthesis = synthesis  + "\n\n" + "References:\n" +  citations
                
            return synthesis
        
    
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
        print('"""""""""""""""""""""""""""""""""""""""""""""""""""" """"""""""""""""""""""""""""""""""""""')



            

            



     
def get_dict_value(dct: dict, key: str):
    """
    Returns the value of the key in the dictionary.
    If the key is not in the dictionary or the value associated with the key is None, returns None.
    """
    try:
        if key in dct and dct[key] is not None:
            return dct[key]
        else:
            return None
    except:
        return None






















