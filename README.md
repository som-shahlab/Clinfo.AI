 ![logo](images/clinfo_ai.png)
 
 Welcome to our demo for Clinfo.ai, if you woud like to see some functionality or have a comment, open an issue on this repo.

**Paper:** https://arxiv.org/abs/2310.16146
**Demo:** https://clinfo-demo.herokuapp.com/login

Millions of medical research articles are published every year. Healthcare professionals are expected to know the latest research. However, with limited time and a broad field to cover, keeping up-to-date can be a challenging task.

Clinfo.AI **searches** and **synthesizes** medical literature tailored to a **specific** clinical **question**. By analyzing the context of the inquiry, it identifies and presents the most relevant articles. 

## Comparison of Clinfo.AI vs ChatGPT:
![comparison](images/comparison.png)


## What type of questions can I ask? 
Questions based on scientific evidence reported in the literature

* Examples:

1. What percentage of HIV-positive patients transmit the virus to their children?

2. When do most episodes of COVID-19 rebound after stopping paxlovid treatment?

3. Does magnesium consumption significantly improve sleep quality?


## Type of questions you can’t answer with clinfo.AI
**Broad questions:** These types of questions could potentially be answered by clinfo.AI, but it is highly probable you won’t get what you are looking for. How to correct this type of question? Provide context.

Example
Original Query: "Chest pain pediatrics"
Improved Query: "What are common causes of chest pain in pediatric patients?"

<br>
Questions that would need to reference EHR or patient information: Clinfo.ai can’t access EHR data.




## How it works?

![diagram](images/diagram.png)


## How can leverage Clinfo.ai using OpenAI models?

#### OPENAI API:
Create an [OpenAI](https://openai.com/index/openai-api/) account, get an API Key, and edit the key field `OPENAI_API_KEY` in `config.py` with your own key. 

#### NCBI API:
Clinfo.ai retrieves literature using the NCBI API; while access does not require an account, calls are limited for unregistered users. We recommend creating an [NCBI](https://www.ncbi.nlm.nih.gov/home/develop/api/) account. Once generated, save the NCBI API key and email under `NCBI_API_KEY` and `EMAIL`, respectively.

In summary edit the following variables inside config.py:
```python
OPENAI_API_KEY = "YOUR API TOKEN"
NCBI_API_KEY   = "YOUR API TOKEN"  (optional)
EMAIL          = "YOUR EMAIL"      (optional)
```

#### Using Clinfo.AI:

```python
from  src.clinfoai.clinfoai import ClinfoAI
from config   import OPENAI_API_KEY, NCBI_API_KEY, EMAIL
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

question = "What is the prevalence of COVID-19 in the United States?"
clinfo   = ClinfoAI(llm="gpt-3.5-turbo",openai_key=OPENAI_API_KEY, email= EMAIL)
answer   = clinfo.forward(question=question)         
```


```src/notebooks/01_UsingClinfoAI.ipynb``` has a quick run-through and explanation for  each individaul  clinfo.AI component.


## How can leverage Clinfo.ai using Open Source models via VLLM?
Clinfo.ai has full integration with [vLLM](). We can use any open source LLM as a backbone following two simple steps:

## Setting an API server
First, we use vLLM to create an API selecting the model you want to work with:
In the following example we use ```Qwen/Qwen2-beta-7B-Chat```

```bash
 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2-beta-7B-Chat
```

### Switch the LLM model name to the selected model 
Instantiate a clinfoAI object with the desired LLM :


```python
from  src.clinfoai.clinfoai import ClinfoAI
from config   import OPENAI_API_KEY, NCBI_API_KEY, EMAIL
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

question = "What is the prevalence of COVID-19 in the United States?"
clinfo   = ClinfoAI(llm="Qwen/Qwen2-beta-7B-Chat",openai_key=OPENAI_API_KEY, email= EMAIL)
answer   = clinfo.forward(question=question)         
```


### IMPORTANT:
While anyone can use Clinfo.AI, our goal is to augment medical experts not replace them. Read our disclaimer [disclaimer](https://clinfo-demo.herokuapp.com/termsandconditions) and DO NOT use clinfo.AI for diagnosis.



### Cite
If you use Clinfo.ai, please consider citing:

```
@inproceedings{lozano2023clinfo,
  title={Clinfo. ai: An open-source retrieval-augmented large language model system for answering medical questions using scientific literature},
  author={Lozano, Alejandro and Fleming, Scott L and Chiang, Chia-Chun and Shah, Nigam},
  booktitle={PACIFIC SYMPOSIUM ON BIOCOMPUTING 2024},
  pages={8--23},
  year={2023},
  organization={World Scientific}
}

```





