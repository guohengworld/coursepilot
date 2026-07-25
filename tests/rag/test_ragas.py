import os
import sys
from types import ModuleType

# ── 兼容性修复 ─────────────────────────────────────────────
# ragas 0.4.3 在 llms/base.py 中导入了
#   from langchain_community.chat_models.vertexai import ChatVertexAI
# 但 langchain-community >= 0.3.0 已将该模块独立到 langchain-google-vertexai 包中。
# 这里在 ragas 导入前注册一个存根模块，避免 ModuleNotFoundError。
try:
    import langchain_community.chat_models  # noqa: F811
    # 获取真实的 parent 模块路径作为 __file__ 的参考
    _parent_file = getattr(langchain_community.chat_models, "__file__", None)
except ImportError:
    _parent_file = None

_chat_vertexai = ModuleType("langchain_community.chat_models.vertexai")
_chat_vertexai.__path__ = [_parent_file or ""]
_chat_vertexai.__file__ = _parent_file or __file__  # 借用真实文件路径

class ChatVertexAIStub:
    """存根类，仅用于通过 ragas 的模块加载检查，不会在测试中实际使用。"""
    pass

_chat_vertexai.ChatVertexAI = ChatVertexAIStub
sys.modules["langchain_community.chat_models.vertexai"] = _chat_vertexai
# ── 兼容性修复结束 ─────────────────────────────────────────

from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import faithfulness as faithfulness_metric
from ragas.metrics import answer_relevancy as answer_relevancy_metric
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# --- 1. 配置自定义 LLM (你提供的部分) ---
llm = ChatOpenAI(
    api_key=os.getenv("MIMO_API_KEY"),  # 请确保环境变量已设置
    model="mimo-v2.5-pro",
    base_url="https://api.xiaomimimo.com/v1",
    temperature=0.3,
)

# --- 2. 配置 Embeddings (可选，但强烈建议配置) ---
# 注意：Answer Relevancy 指标需要 Embedding 模型。
# 假设 Mimo 也提供了兼容 OpenAI 的 Embedding 接口。
# 如果不确定具体的 embedding model 名称，请查阅 Mimo 文档，或者先只测试 faithfulness。
# 这里假设支持 standard openai 接口，通常模型名可能是 text-embedding-ada-002 或 text-embedding-3-small
try:
    embeddings = OpenAIEmbeddings(
        api_key=os.getenv("MIMO_API_KEY"),
        base_url="https://api.xiaomimimo.com/v1",
        model="text-embedding-ada-002"  # 如果 Mimo 有特定模型名，请在此处修改
    )
except Exception as e:
    print(f"Embeddings 初始化失败，将无法测试 Answer Relevancy: {e}")
    embeddings = None

# --- 3. 准备测试数据 ---
data_samples = {
    'question': [
        '特斯拉是哪一年成立的？',
        'Python 是一种编译型语言吗？'
    ],
    'answer': [
        '特斯拉成立于 2003 年。',
        '不是，Python 是一种解释型语言。'
    ],
    'contexts': [
        ['特斯拉汽车公司由马丁·艾伯哈德和马克·塔彭宁于 2003 年创立。'],
        ['Python 是一种广泛使用的高级编程语言，它是一种解释型语言。']
    ],
    'ground_truth': [
        '2003年',
        '不是，它是解释型语言'
    ]
}

dataset = Dataset.from_dict(data_samples)

# --- 4. 定义评估指标 ---
# faithfulness: 忠实度 (只需要 LLM)
# answer_relevancy: 答案相关性 (需要 LLM 和 Embeddings)
# 使用旧版指标实例（evaluate() 会自动将 llm 注入到指标中）
metrics = [faithfulness_metric]

# 如果 embeddings 配置成功，则加入 answer_relevancy 指标
if embeddings:
    metrics.append(answer_relevancy_metric)
else:
    print("提示：未配置 Embeddings，本次测试将跳过 'answer_relevancy' 指标。")

# --- 5. 运行评估 ---
print("开始运行 Ragas 评估...")

try:
    # llm 和 embeddings 直接作为 evaluate() 的参数传递
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(),
    )

    print("\n评估成功完成！")
    print("=" * 30)

    # 打印结果
    # result.to_pandas() 可以将结果转换为 DataFrame 格式方便查看
    df = result.to_pandas()
    print(df)

    # 如果想只看分数
    # print(result)

except Exception as e:
    print(f"\n评估过程中出错: {e}")
    print("请检查：")
    print("1. MIMO_API_KEY 是否在环境变量中正确设置？")
    print("2. 网络是否能访问 api.xiaomimimo.com？")
    print("3. 模型名称 'mimo-v2.5-pro' 和 embedding 名称是否正确？")
