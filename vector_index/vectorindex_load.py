import os
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, load_index_from_storage
from llama_index.vector_stores import ChromaVectorStore, FaissVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.embeddings import HuggingFaceEmbedding
import chromadb

def vector_retriever(vector_path,similarity_top_k):   
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-zh-v1.5")
    service_context = ServiceContext.from_defaults(chunk_size=300,chunk_overlap=100,embed_model=embed_model,llm=None)
    db = chromadb.PersistentClient(path=vector_path)
    chroma_collection = db.get_or_create_collection("quickstart")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index_rebuild = VectorStoreIndex.from_vector_store(vector_store,service_context=service_context)
    retriever = index_rebuild.as_retriever(similarity_top_k=similarity_top_k)
    return retriever

#embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-zh-v1.5")
#service_context = ServiceContext.from_defaults(chunk_size=300,chunk_overlap=100,embed_model=embed_model,llm=None)

# load from disk
#db2 = chromadb.PersistentClient(path="../store/vector_store")
#chroma_collection = db2.get_or_create_collection("quickstart")
#vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
#index_rebuild = VectorStoreIndex.from_vector_store(vector_store,service_context=service_context)
#retriever = index_rebuild.as_retriever(similarity_top_k=5)


if __name__ == "__main__" :
    retriever = vector_retriever("./store/vector_store", 5)
    result = retriever.retrieve("如何加速pod启动")
    print(result)
