"""测试 GPU 可用性 —— 检查 PyTorch、PaddlePaddle 等框架的 GPU 支持状态

用途：
  - 确认是否能使用 GPU 加速 AI/ML 任务
  - 诊断性能问题（CPU vs GPU）
  - 配置优化前的环境检查

运行方式：
    python tests/test_gpu_availability.py
"""
import torch


def test_pytorch_gpu():
    """测试 PyTorch 的 GPU 支持"""
    print("=" * 60)
    print("PyTorch GPU 支持检测")
    print("=" * 60)

    # CUDA (NVIDIA GPU)
    cuda_available = torch.cuda.is_available()
    print(f"✓ CUDA (NVIDIA GPU): {'可用' if cuda_available else '不可用'}")

    if cuda_available:
        gpu_count = torch.cuda.device_count()
        current_gpu = torch.cuda.get_device_name(0)
        print(f"  - GPU 数量: {gpu_count}")
        print(f"  - 当前 GPU: {current_gpu}")
        print(f"  - CUDA 版本: {torch.version.cuda}")

        # 显存信息
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"  - 已分配显存: {memory_allocated:.2f} GB")
        print(f"  - 已预留显存: {memory_reserved:.2f} GB")

    # MPS (Apple Silicon)
    mps_available = torch.backends.mps.is_available()
    print(f"✓ MPS (Apple Silicon): {'可用' if mps_available else '不可用'}")

    if mps_available:
        print(f"  - MPS 后端已启用")

    # CPU 信息
    print(f"✓ CPU 线程数: {torch.get_num_threads()}")
    print()

    return cuda_available or mps_available



def test_recommendations(has_gpu: bool):
    """根据检测结果给出建议"""
    print("=" * 60)
    print("配置建议")
    print("=" * 60)

    if has_gpu:
        print("✓ 检测到 GPU，建议在以下场景使用 GPU 加速：")
        print("  1. PaddleOCR: device='gpu'")
        print("  2. Embedding 模型: device='cuda'")
        print("  3. Reranker 模型: device='cuda'")
        print("  4. Magic-PDF/MinerU: 自动使用 GPU")
        print()
        print("⚠ 注意：")
        print("  - 确保显存充足（建议 >= 4GB）")
        print("  - 监控显存使用，避免 OOM 错误")
    else:
        print("⚠ 未检测到 GPU，将使用 CPU 运行")
        print("  - OCR 识别速度会较慢")
        print("  - Embedding/Reranker 推理时间较长")
        print()
        print("建议：")
        print("  - 小批量处理文档")
        print("  - 考虑增加 CPU 线程数")
        print("  - 如需高性能，建议添加 NVIDIA GPU")

    print()


if __name__ == "__main__":
    print("\n🔍 CoursePilot GPU 环境检测\n")

    # 测试 PyTorch
    pytorch_has_gpu = test_pytorch_gpu()

    # 给出建议
    has_gpu = pytorch_has_gpu
    test_recommendations(has_gpu)

    print("✅ 检测完成！\n")
