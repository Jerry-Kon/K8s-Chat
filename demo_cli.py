import openai
import os

from vector_index.vectorindex_load import vector_retriever
from summary_vector_index.retriever2qa import SummaryVectorIndex
from constant import *
from utils.tool import *

# init openai key
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv('OPENAI_ENDPOINT')

# init system prompt and message (history)
messages = []
messages.append({"role": "system", "content": SYSTEM_PROMPT_1})

# init summary vector retriever
sv_index_ = SummaryVectorIndex(gpt=True)
sv_index = sv_index_.index_rebuild()
retriever_sv = sv_index.as_retriever(similarity_top_k=2)

# init vector retriever
retriever_v = vector_retriever(similarity_top_k=5, gpt=True)

print("# 欢迎来到 K8s-Chat ！\n"
      "# 输入 clear 以清空历史！\n"
      "# 输入 exit 以退出！\n")

while True:

    # user input
    user_content = input("user: ")

    # exit the chat
    if user_content == "exit":
        break

    # start or continue chat
    if user_content != "clear":

        # add questions to chat messages (history)
        question = {"role": "user", "content": user_content}
        messages.append(question)

        # get intention
        intention_prompt = get_intention_prompt(messages[1:], INTENTION_PROMPT)
        intention = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": intention_prompt}],
        ).choices[0].message["content"]
        # print(intention)

        # get whole documents
        retrieve_summary = retriever_sv.retrieve(intention)
        whole_doc = get_whole_doc(retrieve_summary, sv_index_)

        # add chunks
        retrieve_vector = retriever_v.retrieve(intention)
        context = get_context(whole_doc, retrieve_vector)
        # print(context)

        # get new system prompt and add it to messages
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

        # update messages
        answer = {"role": "assistant", "content": reply}
        messages.append(answer)

    else:
        # clear messages
        print("历史已清空！")
        messages = []
        messages.append({"role": "system", "content": SYSTEM_PROMPT_1})
