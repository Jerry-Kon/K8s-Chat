import openai
import os
import gradio as gr

import tiktoken

from vector_index.vectorindex_load import vector_retriever
from summary_vector_index.retriever2qa import SummaryVectorIndex
from constant import *

# count token numbers
def token_count(input_text: str, model):
    encoder = tiktoken.encoding_for_model(model)
    encoded_text = encoder.encode(input_text)
    token_nums = len(encoded_text)
    return token_nums

# get intention prompt
def get_intention_prompt(messages, prompt):
    history_str = ""
    for line in messages:
        if line["role"] != "system":
            history_str += line["role"] + ":" + line["content"] + "\n"
    intention_prompt = prompt.format(history_str=history_str)
    return intention_prompt


if __name__ == "__main__":
    # init openai key
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_base = os.getenv('OPENAI_ENDPOINT')

    # init summary vector retriever
    sv_index_ = SummaryVectorIndex(gpt=True)
    sv_index = sv_index_.index_rebuild()
    retriever_sv = sv_index.as_retriever(similarity_top_k=2)

    # init vector retriever
    retriever_v = vector_retriever(similarity_top_k=5)

    def respond(input, chat_history, model="gpt-3.5-turbo",temp=0):
        messages = []
        for mes in chat_history:
            messages.append({"role": "user", "content": mes[0]})
            messages.append({"role": "assistant", "content": mes[1]})
        messages.append({"role": "user", "content": input})

        # get intention
        intention_prompt = get_intention_prompt(messages, INTENTION_PROMPT)
        intention = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": intention_prompt}],
        ).choices[0].message["content"]
        print(intention)

        # get whole documents
        retrieve_summary = retriever_sv.retrieve(intention)
        whole_doc = ""
        for node in retrieve_summary:
            doc_nodes = sv_index_.doc_summary_index.docstore.get_nodes(
                sv_index_.summary_ids[node.metadata["summary_id"]])
            for node in doc_nodes:
                whole_doc = whole_doc + "\n" + node.text
                if token_count(whole_doc, model) > 12000:
                    break
            if token_count(whole_doc, model) > 12000:
                break
            whole_doc = whole_doc + "\n\n######\n\n"

        # add chunks
        context = whole_doc
        retrieve_vector = retriever_v.retrieve(intention)
        for chunk in retrieve_vector:
            if token_count(chunk.text, model) > 16000:
                break
            context = context + chunk.text + "\n\n######\n\n"
        print(context)

        system_line = {"role": "system", "content": SYSTEM_PROMPT_2.format(context=context)}
        messages.insert(0, system_line)

        # generate reply
        completion = openai.ChatCompletion.create(
            model=model,
            temperature=temp,
            messages=messages,
        )
        reply = completion.choices[0].message["content"]

        return reply

    with gr.Blocks(title="K8s-Chat") as demo:

        gr.ChatInterface(
            respond,
            retry_btn=None,
            undo_btn=None,
            additional_inputs = [
                gr.Radio(["gpt-3.5-turbo", "gpt-3.5-turbo-1106"], label="model"),
                gr.Slider(0, 2, step=0.1, label="temperature"),
            ]
        )

    demo.launch()