import os
import logging
import sys

import chromadb
import openai
from llama_index.vector_stores import ChromaVectorStore
from llama_index.llms import OpenAI
from llama_index.indices.loading import load_index_from_storage
from llama_index import (
    ServiceContext,
    VectorStoreIndex,
    Document,
    StorageContext
)


SUMMARY_PATH = "./store/posts_summary"
VECTOR_PATH = "./store/posts_vector_gpt"

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv('OPENAI_ENDPOINT')


# rebuild storage context (load summary)
storage_context = StorageContext.from_defaults(persist_dir=SUMMARY_PATH)
doc_summary_index = load_index_from_storage(storage_context)

# process Document
documents_vec = []
summary_ids = doc_summary_index.index_struct.summary_id_to_node_ids
for summary_id in summary_ids:
    #print(doc_summary_index.docstore.get_node(summary_id).text)
    summary_text = doc_summary_index.docstore.get_node(summary_id).text
    doc = Document(
        text=summary_text,
        metadata={
            "summary_id": summary_id
        }
    )
    documents_vec.append(doc)

# init chroma
db = chromadb.PersistentClient(path=VECTOR_PATH)
chroma_collection = db.get_or_create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# vector and persist
storage_context = StorageContext.from_defaults(vector_store=vector_store)
service_context = ServiceContext.from_defaults(
    chunk_size=5000,
    llm=None
)
vector_index = VectorStoreIndex.from_documents(
    documents_vec,
    show_progress=True,
    storage_context=storage_context,
    service_context=service_context
)
