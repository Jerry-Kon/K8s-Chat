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

USER_PROMPT = """
分析以下对话内容，检测a的意图，使对话之外的人能更好地理解。
对话内容：
{history_str}
使用以下 YAML schema 的格式来回复：
```yaml
origin question：
  type: string
  language: chinese
  description: One last original question of 'a'
improve question:
  type: array
  minItems: 2
  maxItems: 3
  items:
    question:
      type: string
      language: chinese
      description: improve the last question of 'a'
require information:
  type: array
  minItems: 2
  maxItems: 3
  items:
    information:
      type: string
      language: chinese
      description: The information required to answer question of 'a'
```

example:
```yaml
origin question：|-
improve question:
  -question: |-
  -question: |-
require information:
  -information: |-
  -information：|-
```

"""

# ```yaml
# origin question：狮子和老虎是竞争关系吗
# improve question:
#   -question: 狮子和老虎会有冲突吗
#   -question: 狮子和老虎的关系怎样
# require information:
#   -information: 狮子和老虎的食物来源
#   -information：狮子和老虎的栖息地
# ```

user_prompt = USER_PROMPT.format(history_str=history_str)
print(user_prompt)

completion = openai.ChatCompletion.create(
  model="gpt-4-0613",
  # model="gpt-3.5-turbo-16k",
  messages=[
    {"role": "user", "content": user_prompt}
  ],
)

print(completion.choices[0].message["content"])



