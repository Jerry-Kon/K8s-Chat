import os
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, load_index_from_storage
from llama_index.vector_stores import ChromaVectorStore, FaissVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.embeddings import HuggingFaceEmbedding
import chromadb


documents = SimpleDirectoryReader(
    input_dir="../contents_processed",
    recursive=True
).load_data()

# save
db = chromadb.PersistentClient(path="../store/vector_store")
chroma_collection = db.get_or_create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-zh-v1.5")
service_context = ServiceContext.from_defaults(chunk_size=300,chunk_overlap=100,embed_model=embed_model,llm=None)
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    service_context=service_context,
    show_progress=True
)

