import openai
import os
from langchain.document_loaders import UnstructuredMarkdownLoader
import tiktoken
import json
import logging
import random
from retyping import retry

logging.basicConfig(filename="ins_gen_log_website_2.txt", format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

system_message = '''
你是一个kubernetes助手，你的回答基于事实，客观且准确。
请使用kubernetes相关的先验知识，并根据user提供的文档，生成五到十组高质量的相关问题和对应的答案。
user所提供的文档是从web上爬取的，因此可能会出现格式错误，请合理理解文档内容。
生成的内容尽可能详细，便于教学使用，帮助初学者理解。
可以生成带有shell、yaml或其他格式的答案。
生成的回答需严格按照如下格式：
问题：*****
答案：*****

问题：*****
答案：*****
'''

instructions = []
paths = []


def recursive_listdir(path, paths):
    files = os.listdir(path)
    for file in files:
        file_path = os.path.join(path, file)
        if os.path.isfile(file_path):
            paths.append(file_path)
        elif os.path.isdir(file_path):
            recursive_listdir(file_path, paths)


@retry(wait_fixed=1000, stop_max_attempt_number=7)
def gen_instruction(context, choose_gpt_4):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_base = os.getenv('OPENAI_ENDPOINT')
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": context}
    ]
    if choose_gpt_4 == True:
        model = "gpt-4-0613"
    else:
        model = "gpt-3.5-turbo-16k"
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.3,
        # n = 2,
        # presence_penalty=1
    )
    logging.info("prompt_tokens: " + str(response["usage"]["prompt_tokens"]))
    logging.info("completion_tokens: " + str(response["usage"]["completion_tokens"]))
    logging.info("total_tokens: " + str(response["usage"]["total_tokens"]))
    reply = response["choices"]
    return reply


def gen_instruction_dataset(data, path, choose_gpt_4):
    responses = gen_instruction(data, choose_gpt_4)
    for res in responses:
        res = res["message"]["content"]
        res_split = res.split("问题：")
        del res_split[0]
        for i in res_split:
            data_pair = {}
            i_split = i.split("答案：")
            data_pair["instruction"] = i_split[0].strip("\n")
            data_pair["input"] = ""
            data_pair["output"] = i_split[1].strip("\n")
            data_pair["source"] = path
            instructions.append(data_pair)


def count_token(data):
    encoder = tiktoken.encoding_for_model("gpt-4")
    encoded_text = encoder.encode(data)
    token_count = len(encoded_text)
    logging.info("count token: " + str(token_count))
    return token_count


if __name__ == "__main__":
    current_path = os.path.dirname(os.path.dirname(__file__))
    path_root = current_path + "/contents_processed"
    recursive_listdir(path_root, paths)
    for path in paths:
        logging.info(path)
        loader = UnstructuredMarkdownLoader(path, strategy="fast")
        data = loader.load()[0].page_content
        token_nums = count_token(data)
        if token_nums > 500 and token_nums < 16000:
            if token_nums < 8000:
                choose_gpt_4 = True
            else:
                choose_gpt_4 = False
            try:
                gen_instruction_dataset(data, path, choose_gpt_4)
            except:
                pass
    with open("instruction_k8s_posts.json", 'w', encoding='utf-8') as f:
        json.dump(instructions, f, indent=4, ensure_ascii=False)
