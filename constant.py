MODEL_NAME = "gpt-3.5-turbo-16k"

SUMMARY_PATH = "store/posts_summary"
VECTOR_PATH = "store/posts_vector"
VECTOR_GPT_PATH = "./store/posts_vector_gpt"

VECTORINDEX_PATH = "store/vector_store"
VECTORINDEX_GPT_PATH = "store/vector_store_gpt"

INTENTION_PROMPT = '''
user和assistant正在讨论关于kubernetes的话题。
分析以下对话内容，检测user的意图，使对话之外的人能更好地理解。
对话内容：
{history_str}
你的回答格式为:
```
user的最后一个原问题：
###
user的问题完善1:
user的问题完善2:
###
回答user的问题所需要的知识：
###
知识1：
知识2：
###
```
'''

SYSTEM_PROMPT_1 = """
你是一个kubernetes助手，你的回答基于事实，详细且准确。
请使用与kubernetes相关的先验知识，来回答问题。
"""

SYSTEM_PROMPT_2 = """
你是一个kubernetes助手，你的回答基于事实，详细且准确。
请依据kubernetes相关先验知识，并参考提供的信息来回答问题。
你回答的内容尽可能全面，可使用提供信息进行补充，忽略与问题无关的信息。
不同来源的信息可能会有重复，它们之间由"######"分隔。
目前已知信息如下：
###
{context}
###
"""
