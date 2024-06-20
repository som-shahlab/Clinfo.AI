
from sentence_transformers import SentenceTransformer
import pinecone

class PineCone_Driver:
    def __init__(self, model,index_name , key, env,embed_dim:int=None,metric='cosine'):
        self.model      = model
        if embed_dim is None:
            try:
                self.embed_dim  = self.model.get_sentence_embedding_dimension()
            except:
                raise ValueError('embed_dim is not provided and model does not have get_sentence_embedding_dimension() method, please provide embed_dim')
        
        self.metric     = metric
        self.index_name = index_name
        self.key        = key
        self.env        = env
        self._init_pincone()
        self._get_index()


    def _init_pincone(self):
        pinecone.init(api_key=self.key, environment=self.env)

    def _get_index(self):
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(name      =  self.index_name,
                                  dimension = self.embed_dim,
                                  metric    = self.metric)
            
        self.index = pinecone.GRPCIndex(self.index_name) #pinecone.Index(self.index_name )


    def semantic_query(self, query, top_k:int=10, include_metadata=True) -> list:
        xq = self.model.encode(query)
        return self.index.query(xq, top_k=top_k, include_metadata=include_metadata)
    
    def print_semantic_query_result(self, result:list):
        for result_ in result['matches']:
            print(f"{round(result_['score'], 2)}: {result_['metadata']['text']}")
            
    def get_most_similar(self, query, top_k:int=10, include_metadata=True,threshold:float=0.80) -> list:
        result = self.semantic_query(query, top_k=top_k, include_metadata=include_metadata)
        return [r for r in result['matches'] if r['score'] > threshold]


    def push(self, ids:list,texts:list,metadatas:list):
        xc      = self.model.encode(texts)      # create embeddings
        records = zip(ids, xc, metadatas)       # create records list for upsert
        self.index.upsert(vectors=records)
   

  


    def fetch(self, ids:list):
        return self.index.fetch(ids)

    
    # def push_single(self, id:str,text:str,metadata:dict):
    #     embeddings = self.model.encode(text)
    #     self.index.upsert(zip(id, embeddings, metadata))


    