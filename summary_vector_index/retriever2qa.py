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
    StorageContext
)
from langchain.prompts import PromptTemplate
from llama_index.embeddings import HuggingFaceEmbedding

SUMMARY_PATH = "../store/posts_summary"
VECTOR_PATH = "../store/posts_vector"

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

class SummaryVectorIndex:
    def __init__(self, summary_path, vector_path):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_base = os.getenv('OPENAI_ENDPOINT')
        self.storage_context = StorageContext.from_defaults(persist_dir=summary_path)
        self.doc_summary_index = load_index_from_storage(self.storage_context)
        self.summary_ids = self.doc_summary_index.index_struct.summary_id_to_node_ids
        self.vector_path = vector_path
    def index_rebuild(self):
        self.db2 = chromadb.PersistentClient(path=self.vector_path)
        self.chroma_collection = self.db2.get_or_create_collection("quickstart")
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-zh-v1.5") 
        self.service_context = ServiceContext.from_defaults(
            chunk_size=5000,
            llm=None,
            embed_model=embed_model
        )
        self.index = VectorStoreIndex.from_vector_store(
            self.vector_store,
            service_context=self.service_context,
        )
        return self.index
   
if __name__ == "__main__":
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_base = os.getenv('OPENAI_ENDPOINT')

    sv_index = SummaryVectorIndex(SUMMARY_PATH,VECTOR_PATH)
    index = sv_index.index_rebuild()

    # retriever
    query = "如何加速pod启动"
    retriever = index.as_retriever(similarity_top_k=2)
    results = retriever.retrieve(query)

    # context
    texts = ""
    s_id = []
    for node in results:
        doc_nodes = sv_index.doc_summary_index.docstore.get_nodes(sv_index.summary_ids[node.metadata["summary_id"]])
        for node in doc_nodes:
            texts = texts + "\n" + node.text
        texts = texts + "\n\n----------------\n\n"

    # prompt and chat
    template = """
    你是一个kubernetes助手，你的回答基于事实，详细且准确。
    请使用与kubernetes相关的先验知识，并根据提供的信息来回答问题。
    可忽略与问题无关信息。
    目前已知信息如下：
    {context}
    """

    PROMPT = PromptTemplate(
        template=template,
        input_variables=["context"],
    )
    prompt = PROMPT.format(context=texts)
    rsp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ]
    )
    answer = rsp.get("choices")[0]["message"]["content"]

    rsp_origin = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "user", "content": query}
        ]
    )
    answer_origin = rsp_origin.get("choices")[0]["message"]["content"]

    print("prompt:\n", prompt)
    print("query:\n", query)
    print("original answer:\n", answer_origin)
    print("answer:\n", answer)
