from openai import OpenAI
import os

class DeepSeekChat:
    def __init__(self, api_key, base_url="https://api.deepseek.com"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(
        self,
        system_message,
        user_message,
        model="deepseek-chat",
        max_tokens=1024,
        temperature=0.7,
        stream=True,
    ):

        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
        )

        if stream:
            return self._stream_response(response)
        else:
            return response.choices[0].message.content

    def _stream_response(self, response):
        full_response = ""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content
        
        print("\r\n===============我是分隔线===============")
        return full_response


# 使用示例
if __name__ == "__main__":
    deepseek_chat = DeepSeekChat(api_key=os.getenv("DEEPSEEK_API_KEY"))
    response = deepseek_chat.chat(
        system_message="你是一个聪明的AI助手",
        user_message="三国演义中战斗力排名前10的武将有谁？",
        stream=True,
    )
    print("完整回答:", response)
