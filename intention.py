import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv('OPENAI_ENDPOINT')

# query = ""
messages_history = [
  {"role":"a","content":"一个岛上有海豹和企鹅，企鹅有翅膀吗？"},
  {"role":"b","content":"企鹅有翅膀"},
  {"role":"a","content":"它们会飞吗？"},
  {"role":"b","content":"企鹅不能够飞翔"},
  {"role":"a","content":"它们能够和睦相处吗？"}
]
history_str = ""
for line in messages_history:
  history_str += line["role"] + ":" + line["content"] + "\n"

USER_PROMPT = '''
分析以下对话内容，检测a的意图，使对话之外的人能更好地理解。
对话内容：
{history_str}
你的回答格式为
a的最后一个原问题：
a的问题完善1:
a的问题完善2:
a的问题完善3:
'''

user_prompt = USER_PROMPT.format(history_str=history_str)
print(USER_PROMPT)
print(user_prompt)

completion = openai.ChatCompletion.create(
  model="gpt-4-0613",
  # model="gpt-3.5-turbo-16k",
  messages=[
    {"role": "user", "content": user_prompt}
  ],
)

print(completion.choices[0].message["content"])



