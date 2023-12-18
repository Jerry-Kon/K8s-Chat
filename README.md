# K8s-Chat
K8s-Chat is a project or chatbot with RAG function, which can answer questions related to kubernetes (k8s). Of course, you can also replace k8s with other fields of knowledge.
## Chat Flow
The chat flow is shown in the following firgue:
<div align=center><img src="https://github.com/Jerry-Kon/K8s-Chat/blob/main/image/k8s-chat-flow-latest.png" width="550px"></div>  

First, a ``Query`` enters ``Chat history``. Then the ``Chat history`` is processed through the ``LLM`` to get the Intent. The docs or chunks with high similarity to the intention is retrieved through the ``Retriever``. Use the search results to update the ``System prompt``. Finally, send the ``System prompt`` and ``Chat history`` to the ``LLM`` and get the relevant response, which will be part of the ``Chat history``. In general, the process repeats itself.

## Quick Start
### Prepare 
Through the operation, you can process your own documents(knowledge).  
Generate summary vector index:
```shell
python summary_vector_index/doc2summary.py
python summary_vector_index/summary2vector.py
python summary_vector_index/summary2vector_gpt.py #optional gpt vector
```
Generate vector index:
```shell
python vector_index/vectorindex_save.py
python vector_index/vectorindex_save_gpt.py #optional gpt vector
```
### Usage 
#### CLI interface
Run the CLI interface demo: 
```shell
python demo_cli.py
```
Show CLI interface:
<div align=center><img src="https://github.com/Jerry-Kon/K8s-Chat/blob/main/image/demo_cli.png" width="320px"></div>  

#### Web UI interface
Run the Web UI interface demo:  
```shell
python demo_webui.py
```
Access Web UI, by local ip:port : <http://127.0.0.1:7860>  
you can also publish service by set gtadio `launch(share=True)` and generate share link (expires in 72 hours).    
Show Web UI interface :
<div align=center><img src="https://github.com/Jerry-Kon/K8s-Chat/blob/main/image/demo_webui.png" width="1100px"></div> 

#### API interface
Use API interface demo:
```shell
python demo_api.py
```
then you can make request:
```python
import requests

url = "http://127.0.0.1:7777/chat/" 
query = {"text": "你好，请做一段自我介绍。"}
response = requests.post(url, json=query)

if response.status_code == 200:
    result = response.json()
    print("BOT:", result["result"])
else:
    print("Error:", response.status_code, response.text)
```

## Roadmap
New features are coming soon :rocket::
+ Local model :fire:: Chat with local LLMs.
+ Fine-tuning model :art:: Fine-tuning model based on k8s.
