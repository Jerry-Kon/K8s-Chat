import openai
import os

from qa_summary.retriever2qa import SummaryVectorIndex
from constant import SUMMARY_PATH, VECTOR_PATH, INTENTION_PROMPT, SYSTEM_PROMPT_1, SYSTEM_PROMPT_2


def get_intention_prompt(messages, prompt):
    history_str = ""
    for line in messages[1:]:
        history_str += line["role"] + ":" + line["content"] + "\n"
    intention_prompt = prompt.format(history_str=history_str)
    return intention_prompt


openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv('OPENAI_ENDPOINT')

messages = []
messages.append({"role": "system", "content": SYSTEM_PROMPT_1})

sv_index = SummaryVectorIndex(SUMMARY_PATH, VECTOR_PATH)
index = sv_index.index_rebuild()
retriever = index.as_retriever(similarity_top_k=1)

print("可输入clear来清空聊天历史\n"
      "可输入exit来退出聊天")

while True:
    user_content = input("user: ")
    if user_content == "exit":
        break
    if user_content != "clear":
        question = {"role": "user", "content": user_content}
        messages.append(question)

        intention_prompt = get_intention_prompt(messages, INTENTION_PROMPT)
        intention = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=[{"role": "user", "content": intention_prompt}],
        ).choices[0].message["content"]
        retrieve_summary = retriever.retrieve(intention)

        whole_doc = ""
        for node in retrieve_summary:
            doc_nodes = sv_index.doc_summary_index.docstore.get_nodes(sv_index.summary_ids[node.metadata["summary_id"]])
            for node in doc_nodes:
                whole_doc = whole_doc + "\n" + node.text
            whole_doc = whole_doc + "\n\n----------------\n\n"

        system_line = {"role": "system", "content": SYSTEM_PROMPT_2.format(context=whole_doc)}
        messages.pop(0)
        messages.insert(0, system_line)

        completion = openai.ChatCompletion.create(
            model="gpt-4-0613",
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
