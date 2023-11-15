import openai
import os
import json
import random

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv('OPENAI_ENDPOINT')

prompt = '''
你是一个教授kubernetes相关知识的老师，你的回答需基于事实，客观且准确。
请使用kubernetes相关先验知识，丰富user提供的问答对中的回答，便于初学者学习理解。
请注意：对于问题的回答必须完整，不要输出对原回答的评价。
输出格式如下：
问题：********（原来的问题）
回答：********
'''

instructions = []
with open("./dataset/instruction_k8s.json", "r", encoding="utf-8") as f:
    lines = json.load(f)
    for l in lines:
        instructions.append(l)

instruction_pro = []
n = [1, 2]  # 抽取50%

for i, data_pair in enumerate(instructions):
    if random.choice(n) == 1:
        qa_pair = data_pair["instruction"] + "\n" + data_pair["output"]
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-4-0613",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": qa_pair}
                ],
                n=random.choice(n),
                temperature=0.7
            )
            for res in completion.choices:
                res_output = res.message["content"]
                dic = {}
                dic["instrucetion"] = data_pair["instruction"]
                dic["input"] = ""
                dic["output"] = res_output.split("回答：")[1]
                instruction_pro.append(dic)
        except:
            pass
    with open("./dataset/instruction_k8s_data.json", 'w', encoding='utf-8') as f:
        json.dump(instruction_pro, f, indent=4, ensure_ascii=False)
