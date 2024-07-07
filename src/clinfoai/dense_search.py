import faiss
import torch
import numpy as np
import json
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

def generate_paths(
    base_dir: str,
    init_chunk:int  = 0,
    end_chunks: int = 36) -> (list[str], list[str]):
    
    base_path = Path(base_dir)
    embeddings_paths = []
    pmids_paths      = []

    for i in range(init_chunk,end_chunks+1):
        
        embeddings_paths.append(str(base_path / "embeddings" / f"embeds_chunk_{i}.npy"))
        pmids_paths.append(str(base_path / "pmids" / f"pmids_chunk_{i}.json"))

    return embeddings_paths, pmids_paths

class PubMedDenseSearch:
    def __init__(
        self,
        pubmed_embeds_files: list[str], 
        pmids_files: list[str], 
        model_name: str = "ncbi/MedCPT-Query-Encoder",
        index_file: str = None,
        verbose:bool = True):
        
        self.pubmed_embeds: list = []
        self.pmids: list[str] = [] 
        self.verbose:bool = verbose

        assert len(pubmed_embeds_files) == len(pmids_files)

        if index_file and Path(index_file).exists():
            # Load the Faiss index from the file
            if self.verbose:
                print(f"Generated Index from: {index_file}")
            self.index = faiss.read_index(index_file)
    
        else:
            # Load PubMed embeddings
            if self.verbose:
                print("Initalizing Index from sratched")
            self.load_embeddings(pubmed_embeds_files,index_file)
            
        # Load PMIDs
        for pmid_file in pmids_files:
            with open(pmid_file, 'r') as f:
                pmids_chunk = json.load(f)
                self.pmids.extend(pmids_chunk)

        # Load transformer model and tokenizer
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def load_embeddings(
        self,
        pubmed_embeds_files,
        index_file):
        for embed_file in pubmed_embeds_files:
            embeds_chunk = np.load(embed_file)
            self.pubmed_embeds.append(embeds_chunk)
        
        # Concatenate embeddings into a single numpy array
        self.pubmed_embeds = np.concatenate(self.pubmed_embeds)

        # Build Faiss index
        self.index = faiss.IndexFlatIP(self.pubmed_embeds.shape[1])
        self.index.add(self.pubmed_embeds)
        del self.pubmed_embeds 
        # Save the Faiss index to the file if specified
        if index_file:
            print(f"Saving Index to: {index_file}")
            faiss.write_index(self.index, index_file)

    def search(
        self, 
        queries:list[str], 
        k=10):
        with torch.no_grad():
            # Tokenize queries
            encoded = self.tokenizer(
                queries,
                truncation=True,
                padding=True,
                return_tensors='pt',
                max_length=64,
            )

            # Encode queries
            embeds = self.model(**encoded).last_hidden_state[:, 0, :]

            # Search the Faiss index
            scores, inds = self.index.search(embeds.cpu().numpy(), k=k)

        # Return results
        results = []
        for idx, query in enumerate(queries):
            query_results = []
            for score, ind in zip(scores[idx], inds[idx]):
                pmid = self.pmids[ind]
                query_results.append({'PMID': pmid, 'Score': score})
            results.append({'Query': query, 'Results': query_results})
        
        return results