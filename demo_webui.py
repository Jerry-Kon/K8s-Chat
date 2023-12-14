import openai
import os
import gradio as gr

from vector_index.vectorindex_load import vector_retriever
from summary_vector_index.retriever2qa import SummaryVectorIndex
from constant import *
from utils.tool import *

if __name__ == "__main__":

    # init openai key
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_base = os.getenv('OPENAI_ENDPOINT')

    # init summary vector retriever
    sv_index_ = SummaryVectorIndex(gpt=True)
    sv_index = sv_index_.index_rebuild()
    retriever_sv = sv_index.as_retriever(similarity_top_k=3)

    # init vector retriever
    retriever_v = vector_retriever(similarity_top_k=5)

    # main chat flow
    def respond(input, chat_history, model, temp=0):
        if model == None:
            model = "gpt-3.5-turbo-16k"

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
        whole_doc = get_whole_doc(retrieve_summary, sv_index_)

        # add chunks
        retrieve_vector = retriever_v.retrieve(intention)
        context = get_context(whole_doc, retrieve_vector)
        print(context)

        # generate system prompt
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

    # gradio ui with chatinterface (chatbot)
    gr.ChatInterface(
        respond,
        chatbot=gr.Chatbot(height=450),
        textbox=gr.Textbox(placeholder="Chat with me!", container=False, scale=7),
        # description="K8s-Chat is a chatbot with RAG function, which can answer questions related to kubernetes (k8s).",
        title="K8s-Chat UI",
        additional_inputs=[
            gr.Radio(["gpt-3.5-turbo-16k", "gpt-3.5-turbo-1106"], label="model"),
            gr.Slider(0, 2, step=0.1, label="temperature"),
        ],
        additional_inputs_accordion_name="Related configuration",
        # retry_btn=None,
        # undo_btn=None,
    ).queue().launch(share=True)
