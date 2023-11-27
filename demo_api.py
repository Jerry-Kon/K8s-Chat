import openai
import os
import tiktoken
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from vector_index.vectorindex_load import vector_retriever
from summary_vector_index.retriever2qa import SummaryVectorIndex
from constant import *
from utils.tool import *


# init openai key
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv('OPENAI_ENDPOINT')

# init summary vector retriever
sv_index_ = SummaryVectorIndex(gpt=True)
sv_index = sv_index_.index_rebuild()
retriever_sv = sv_index.as_retriever(similarity_top_k=2)

# init vector retriever
retriever_v = vector_retriever(similarity_top_k=5, gpt=True)

app = FastAPI()


class History(BaseModel):
    text: list


@app.post("/chat/")
async def chat(history: list):
    intention_prompt = get_intention_prompt(history, INTENTION_PROMPT)
    intention = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": intention_prompt}],
    ).choices[0].message["content"]

    # get whole documents
    retrieve_summary = retriever_sv.retrieve(intention)
    whole_doc = get_whole_doc(retrieve_summary, sv_index_)

    # add chunks
    retrieve_vector = retriever_v.retrieve(intention)
    context = get_context(whole_doc, retrieve_vector)
    # print(context)

    messages = []
    system_line = {"role": "system", "content": SYSTEM_PROMPT_2.format(context=context)}
    messages.append(system_line)
    messages.extend(history)

    # generate reply
    completion = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=messages,
    )
    reply = completion.choices[0].message["content"]

    return {"result": reply}


uvicorn.run(app, host="127.0.0.1", port=7777)
