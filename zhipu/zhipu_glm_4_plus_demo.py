from zhipuai import ZhipuAI
import os


def chat_with_glm4(system_message, user_message):
    client = ZhipuAI(api_key=os.getenv("GLM_4_PLUS_API_KEY"))
    response = client.chat.completions.create(
        model="glm-4-plus",
        messages=[
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
        stream=True,
    )
    return response


def main():
    system_msg = (
        "你是一个乐于回答各种问题的小助手，你的任务是提供专业、准确、有洞察力的建议。"
    )
    user_msg = "世界上高度排名前10的山峰分别是哪些？"
    response = chat_with_glm4(system_msg, user_msg)

    for chunk in response:
        print(chunk.choices[0].delta.content, end="", flush=True)


if __name__ == "__main__":
    main()
