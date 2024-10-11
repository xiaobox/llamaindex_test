import os
from dotenv import load_dotenv
import logging

# 加载环境变量
load_dotenv()

# API密钥配置
ZHIPU_API_KEY = os.getenv("GLM_4_PLUS_API_KEY")
if not ZHIPU_API_KEY:
    raise ValueError("ZHIPU_API_KEY environment variable is not set")

# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)