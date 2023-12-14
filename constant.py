MODEL_NAME = "gpt-3.5-turbo-16k"

SUMMARY_PATH = "store/posts_summary"
VECTOR_PATH = "store/posts_vector"
VECTOR_GPT_PATH = "./store/posts_vector_gpt"

VECTORINDEX_PATH = "store/vector_store"
VECTORINDEX_GPT_PATH = "store/vector_store_gpt"

INTENTION_PROMPT = '''
user和assistant正在讨论关于kubernetes的话题。

对话内容如下：
###
{history_str}
###

现在你需要帮助assistant理解user的意图，以便于assistant在知识库中查找资料，assistant将会使用这些资料来回答问题。

使用以下yaml格式输出:
```yaml
user的最后一个原问题： <>
user的问题的补全：#补全原问题，脱离对话上下文也能理解其意图
    - 补全1: <>
查找内容： #列出两个查找资料所用的关键词或句子
    - 内容1： <>
    - 内容2： <>
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
