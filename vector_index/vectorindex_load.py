import os
import sys

import openai
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, load_index_from_storage
from llama_index.vector_stores import ChromaVectorStore, FaissVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.embeddings import HuggingFaceEmbedding
import chromadb

sys.path.append(os.getcwd())
from constant import VECTORINDEX_PATH, VECTORINDEX_GPT_PATH

def vector_retriever(similarity_top_k:int=5, gpt:bool=True):
    if gpt==True:
        vector_path = VECTORINDEX_GPT_PATH
        openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_base = os.getenv('OPENAI_ENDPOINT')
        service_context = ServiceContext.from_defaults(llm=None)
    else:
        vector_path = VECTORINDEX_PATH
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-zh-v1.5")
        service_context = ServiceContext.from_defaults(embed_model=embed_model,llm=None)
    db = chromadb.PersistentClient(path=vector_path)
    chroma_collection = db.get_or_create_collection("quickstart")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index_rebuild = VectorStoreIndex.from_vector_store(vector_store,service_context=service_context)
    retriever = index_rebuild.as_retriever(similarity_top_k=similarity_top_k)
    return retriever


if __name__ == "__main__" :
    retriever = vector_retriever(5)
    result = retriever.retrieve("如何加速pod启动")
    print(result)
