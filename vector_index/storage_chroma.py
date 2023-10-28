import os
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, load_index_from_storage
from llama_index.vector_stores import ChromaVectorStore, FaissVectorStore
from llama_index.storage.storage_context import StorageContext
import chromadb
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv('OPENAI_ENDPOINT')

documents = SimpleDirectoryReader(
    input_dir="../contents_processed",
    recursive=True
).load_data()

# save
db = chromadb.PersistentClient(path="../store/vector_store")
chroma_collection = db.get_or_create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
service_context = ServiceContext.from_defaults(chunk_size=300, chunk_overlap=100)
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    service_context=service_context,
    show_progress=True
)

# load from disk
db2 = chromadb.PersistentClient(path="../store/vector_store")
chroma_collection = db2.get_or_create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index_rebuild = VectorStoreIndex.from_vector_store(vector_store)
retriever = index_rebuild.as_retriever(similarity_top_k=5)
print(retriever)
