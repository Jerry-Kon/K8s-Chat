# K8s-Chat
K8s-Chat is a project or chatbot :robot: with RAG function, which can answer questions related to kubernetes (k8s). Of course, you can also replace k8s with other fields of knowledge.	:sparkles:
## Chat Flow
The chat flow is shown in the following firgue:
<div align=center><img src="https://github.com/Jerry-Kon/K8s-Chat/blob/main/image/k8s-chat-flow-latest.png" width="550px"></div>  

First, a ``Query`` enters ``Chat history``. Then the ``Chat history`` is processed through the ``LLM`` to get the Intent. The docs or chunks with high similarity to the intention is retrieved through the ``Retriever``. Use the search results to update the ``System prompt``. Finally, send the ``System prompt`` and ``Chat history`` to the ``LLM`` and get the relevant response, which will be part of the ``Chat history``. In general, the process repeats itself.

## Quick Start
Prepare(option):  
Through the operation, you can process your own documents(knowledge), .
```
python summary_vector_index/doc2summary.py
python summary_vector_index/summary2vector.py
python summary_vector_index/summary2vector_gpt.py #optional gpt vector
```
Prepare:
```
python vector_index/vectorindex_save.py
python vector_index/vectorindex_save_gpt.py #optional gpt vector
```
Usage: 
```
python cli_demo.py
```
