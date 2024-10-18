import logging
from typing import Any, List, Optional
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 设置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class LocalGLM4(CustomLLM):

    context_window: int = 8192  # 默认上下文窗口大小
    num_output: int = 2048  # 默认输出的token数量
    model_name: str = "glm-4-9b-chat"  # 模型名称
    tokenizer: object = None  # 分词器
    model: object = None  # 模型

    def __init__(self, pretrained_model_name_or_path: str):
        super().__init__()

        # GPU方式加载模型
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch.float16,  # 或者使用 torch.bfloat16
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto",
        )

        # CPU方式加载模型
        # self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, device_map="cpu", trust_remote_code=True)
        # self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, device_map="cpu", trust_remote_code=True)
        # self.model = self.model.float()

        # 尝试获取模型的实际上下文窗口大小
        if hasattr(self.model.config, 'seq_length'):
            self.context_window = self.model.config.seq_length
        elif hasattr(self.model.config, 'max_position_embeddings'):
            self.context_window = self.model.config.max_position_embeddings
        logger.info(f"Using context window size: {self.context_window}")

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        # 得到LLM的元数据
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        # 完成函数
        print("完成函数")

        inputs = self.tokenizer.encode(prompt, return_tensors="pt").cuda()  # GPU方式
        # inputs = self.tokenizer.encode(prompt, return_tensors='pt')  # CPU方式
        outputs = self.model.generate(inputs, max_length=self.num_output)
        response = self.tokenizer.decode(outputs[0])
        return CompletionResponse(text=response)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        # 流式完成函数
        print("流式完成函数")

        inputs = self.tokenizer.encode(prompt, return_tensors="pt").cuda()  # GPU方式
        # inputs = self.tokenizer.encode(prompt, return_tensors='pt')  # CPU方式
        outputs = self.model.generate(inputs, max_length=self.num_output)
        response = self.tokenizer.decode(outputs[0])
        for token in response:
            yield CompletionResponse(text=token, delta=token)
