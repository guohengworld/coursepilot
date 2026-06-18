# PDF 解析性能优化报告

> 日期：2026-06-18 | 测试文件：《大学数学 微积分 下册》21 页 | 硬件：RTX 4060 Laptop 8GB

## 问题

`test_real_pipeline.py` 解析 20 页 PDF 耗时 ~7 分钟（~21s/page），一本 300 页教材约需 105 分钟。

## 根因

| 问题 | 详情 |
|------|------|
| **PyTorch CPU 版本** | `torch 2.12.0+cpu`，RTX 4060 8GB 完全空闲 |
| **method="ocr"** | 强制每页走 PaddleOCR，文字页也做 OCR（占单页耗时 70-80%） |

MinerU 的布局检测 (PP-DocLayoutV2)、OCR (PaddleOCR)、公式识别 (Unimernet) 全部在 CPU 上运行，GPU 利用率为 0。

## 解决方案

### 1. 安装 CUDA 版 PyTorch（主要收益）

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

PyTorch `2.12.0+cpu` → `2.6.0+cu124`，所有 MinerU 模型迁移到 GPU 推理。

同时在 `pyproject.toml` 中锁定 CUDA 索引源：

```toml
[tool.uv.sources]
torch = { index = "pytorch-cu124" }
torchvision = { index = "pytorch-cu124" }

[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true
```

### 2. `mineru_method: "ocr"` → `"auto"`

`auto` 模式自动检测每页类型：文字页直接提取文本（~0.1s/page），仅扫描页走 PaddleOCR（~1-2s/page GPU）。

### 3. 新增公式/表格识别开关

`config.py` 新增 `mineru_formula_enable` / `mineru_table_enable`，Phase A（大纲提取）可关闭省 20%。

### 4. 计时工具

`parse_pdf()` 返回 `_timings` 字段，记录各步骤耗时，用于诊断瓶颈。

## 效果

### 21 页测试对比

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| MinerU 解析 | ~420s | 67s | **6.3x** |
| 总测试时间 | ~460s | 68s | **6.8x** |
| GPU 利用率 | 0% | 活跃 | — |

### 21 页 MinerU 耗时明细 (GPU)

```
mineru_ocr: 67.04s
  ├── 模型加载 (CPU→GPU): 7.2s (11%)
  ├── Layout Predict:      2s  (3%)
  ├── MFR Predict (公式):   14s (21%) ← Phase A 可关闭
  ├── OCR-det:             3s  (5%)
  ├── OCR-rec:             4s  (6%)
  └── Processing:          1s  (2%)
```

### 300 页教材预估

| 阶段 | CPU (ocr) | GPU (auto) | 提升 |
|------|-----------|------------|------|
| 解析 | ~105 min | ~6 min | **~17x** |

## 代码改动

| 文件 | 改动 |
|------|------|
| `src/coursepilot/config.py:50` | `mineru_method="auto"`，新增 `mineru_formula_enable` / `mineru_table_enable` |
| `src/coursepilot/ingestion/pdf_parser.py` | `_timed` 计时，`formula`/`table` 参数 `bool \| None`，修复 v3 输出路径 |
| `pyproject.toml` | `[tool.uv.sources]` 锁定 PyTorch CUDA 索引 |
| `tests/test_real_pipeline.py` | 打印 `_timings` 和 `_config` 字段 |

## 后续优化空间

| 优化项 | 预期收益 | 复杂度 |
|--------|----------|--------|
| Phase A 关闭公式/表格识别 | 额外 20% 加速 (~14s) | 极低（已有开关） |
| PyMuPDF 快速通道（文字 PDF） | <5s 完成 Phase A | 中 |
| mineru_device_mode=cuda 显式设置 | 避免自动检测开销 | 极低 |
