# K8s-Chat
K8s-Chat is a project about k8s knowledge base chatbot 🤖
## Chat Flow
The chat flow is shown in the following firgue:
<div align=center><img src="https://github.com/Jerry-Kon/K8s-Chat/blob/main/image/chat_flow.png" width="600px"></div>  

First, a `Query` enters ``Chat history``. Then the ``Chat history`` is processed through the ``LLM`` to get the Intent. The docs or chunks with high similarity to the intention is retrieved through the knowledge base. Use the search results to update the ``System prompt``. Finally, send the *``System prompt`` and ``Chat history`` to the ``LLM`` and get the relevant response, which will be part of the ``Chat history``. In general, the process repeats itself.

## Quick Start
Prepare(option):
```
python summary_vector_index/doc2summary.py
python summary_vector_index/summary2vector.py
```
Prepare:
```
python vector_index/vectorindex_save.py
```
Usage: 
```
python chatflow.py
```
