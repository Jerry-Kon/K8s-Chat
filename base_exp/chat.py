import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv('OPENAI_ENDPOINT')

messages = [{"role": "system", "content":"你是一个猫娘，每句话结尾都会说喵~"}]

print("输入clear来清空聊天历史\n"
      "输入exit来退出聊天")

while True:
    user_content = input("user: ")
    if user_content == "exit":
        break
    if user_content != "clear":
        question = {"role": "user", "content": user_content}
        messages.append(question)
        completion = openai.ChatCompletion.create(
            # model="gpt-4-0613",
            model="gpt-3.5-turbo",
            messages=messages,
        )
        reply = completion.choices[0].message["content"]
        print("assistant: ", reply)
        answer = {"role": "assistant", "content": reply}
        messages.append(answer)
    else:
        print("聊天历史已清空")
        messages = messages[0]
