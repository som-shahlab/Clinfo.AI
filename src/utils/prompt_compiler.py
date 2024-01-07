
from langchain.prompts import load_prompt
from pathlib import Path
import os
import json



def save_json(dict_:dict,file_name:str):
    with open(file_name, 'w') as fp:
        json.dump(dict_, fp)
        
        
def read_json(file_name:str):
    with open(file_name, 'r') as fp:
        data = json.load(fp)
    return data




class PromptArchitecture: 
    def __init__(self, architecture_path: str,verbose:bool=True):
        self.verbose             = verbose 
        self.prompt_architecture = Path(architecture_path)
        self.path                = self.prompt_architecture.parent
        self.architecture        = self.read_architecture()
        self.compile_prompts()

    def read_architecture(self) -> dict:
        """ Reads the architecture from a given path"""
        architecture = read_json(file_name=self.prompt_architecture)
        return architecture
    
    def compile_prompts(self) -> dict:
        for key,sub_task in self.architecture["$schema"].items():
            print(f"\nTask Name: {key}")
            print("------------------------------------------------------------------------")
            for key_, sub_task_ in sub_task.items(): 
                if  self.verbose:
                    print(f"Loading prompt: {key_}  from file {sub_task_ }")
                self.architecture["$schema"][key][key_] =  load_prompt( os.path.join(self.path,self.architecture["$schema"][key][key_]) )

    def get_prompt(self,task:str,sub_task:str=None) -> dict:
        if sub_task is None:
            return self.architecture["$schema"][task]
        else:
            return self.architecture["$schema"][task][sub_task]
        
    def get_task_names(self) -> dict:
        return list(self.architecture["$schema"].keys())
    
    def reconstruct_from_json(self,json_:dict) -> dict:
        return json_

    def print_architecture(self):
        print(self.architecture)