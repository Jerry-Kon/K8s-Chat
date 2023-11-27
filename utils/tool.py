import tiktoken


# count token numbers
def token_count(input_text: str, MODEL_NAME: str = "gpt-3.5-turbo-16k"):
    encoder = tiktoken.encoding_for_model(MODEL_NAME)
    encoded_text = encoder.encode(input_text)
    token_nums = len(encoded_text)
    return token_nums


# get intention prompt
def get_intention_prompt(messages, prompt):
    history_str = ""
    for line in messages:
        history_str += line["role"] + ":" + line["content"] + "\n"
    intention_prompt = prompt.format(history_str=history_str)
    return intention_prompt


def get_whole_doc(retrieve_summary, sv_index_):
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
    return whole_doc


def get_context(context, retrieve_vector):
    for chunk in retrieve_vector:
        if token_count(chunk.text) > 16000:
            break
        context = context + chunk.text + "\n\n######\n\n"
    return context
