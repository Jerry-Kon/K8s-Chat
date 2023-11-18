import openai
import os
import tiktoken

from vector_index.vectorindex_load import vector_retriever
from summary_vector_index.retriever2qa import SummaryVectorIndex
from constant import *


# count token numbers
def token_count(input_text: str):
    encoder = tiktoken.encoding_for_model(MODEL_NAME)
    encoded_text = encoder.encode(input_text)
    token_nums = len(encoded_text)
    return token_nums


# get intention prompt
def get_intention_prompt(messages, prompt):
    history_str = ""
    for line in messages[1:]:
        history_str += line["role"] + ":" + line["content"] + "\n"
    intention_prompt = prompt.format(history_str=history_str)
    return intention_prompt


# init openai key
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv('OPENAI_ENDPOINT')

# init system prompt
messages = []
messages.append({"role": "system", "content": SYSTEM_PROMPT_1})

# init summary vector retriever
sv_index_ = SummaryVectorIndex(gpt=True)
sv_index = sv_index_.index_rebuild()
retriever_sv = sv_index.as_retriever(similarity_top_k=2)

# init vector retriever
retriever_v = vector_retriever(similarity_top_k=5, gpt=True)

print("#欢迎来到K8s-Chat，开始聊天吧！"
      "#输入clear来清空聊天历史\n"
      "#输入exit来退出聊天")

while True:
    user_content = input("user: ")
    if user_content == "exit":
        break
    if user_content != "clear":
        question = {"role": "user", "content": user_content}
        messages.append(question)

        # get intention
        intention_prompt = get_intention_prompt(messages, INTENTION_PROMPT)
        intention = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": intention_prompt}],
        ).choices[0].message["content"]
        # print(intention)

        # get whole documents
        retrieve_summary = retriever_sv.retrieve(intention)
        whole_doc = ""
        for node in retrieve_summary:
            doc_nodes = sv_index_.doc_summary_index.docstore.get_nodes(
                sv_index_.summary_ids[node.metadata["summary_id"]])
            for node in doc_nodes:
                whole_doc = whole_doc + "\n" + node.text
                if token_count(whole_doc) > 12000:
                    break
            if token_count(whole_doc) > 12000:
                break
            whole_doc = whole_doc + "\n\n######\n\n"

        # add chunks
        context = whole_doc
        retrieve_vector = retriever_v.retrieve(intention)
        for chunk in retrieve_vector:
            if token_count(chunk.text) > 16000:
                break
            context = context + chunk.text + "\n\n######\n\n"

        # get new system prompt
        system_line = {"role": "system", "content": SYSTEM_PROMPT_2.format(context=context)}
        messages.pop(0)
        messages.insert(0, system_line)

        # generate reply
        completion = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=messages,
        )
        reply = completion.choices[0].message["content"]
        print("assistant: ", reply)

        answer = {"role": "assistant", "content": reply}
        messages.append(answer)

    else:
        print("聊天历史已清空")
        messages = []
        messages.append({"role": "system", "content": SYSTEM_PROMPT_1})
