import openai
import os
import gradio as gr

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
    for line in messages:
        if line["role"] != "system":
            history_str += line["role"] + ":" + line["content"] + "\n"
    intention_prompt = prompt.format(history_str=history_str)
    return intention_prompt


if __name__ == "__main__":
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
    retriever_v = vector_retriever(similarity_top_k=5)


    def respond(input, chat_history, source):
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

        # get chunks
        retrieve_vector = retriever_v.retrieve(intention)
        chunks = ""
        for chunk in retrieve_vector:
            chunks = chunks + chunk.text + "\n\n######\n\n"

        # concatenate documents and chunks
        context = whole_doc + chunks

        system_line = {"role": "system", "content": SYSTEM_PROMPT_2.format(context=context)}
        messages.insert(0, system_line)

        # generate reply
        completion = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=messages,
        )
        reply = completion.choices[0].message["content"]

        chat_history.append((input, reply))
        source.append(( input, intention+context))

        return "", chat_history, source

    with gr.Blocks() as demo:
        with gr.Column(scale=2):
            with gr.Row():
                chatbot = gr.Chatbot(label="Chat")
                source = gr.Chatbot(label="Source")
            msg = gr.Textbox()
            clear = gr.Button("clear")


        msg.submit(respond, [msg, chatbot, source], [msg, chatbot, source])
        clear.click(lambda: None, None, chatbot, queue=False)

    demo.launch()