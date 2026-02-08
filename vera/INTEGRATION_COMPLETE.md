# VERA 包集成完成总结

## 完成时间
2025-02-08

## 概述

成功将完整的三阶段分析逻辑整合到 VERA 软件包中，现在用户可以通过简洁的 API 调用运行完整的分析流程。

## 完成的工作

### 1. 更新 `vera/analysis/__init__.py`

**修改内容:**
- 添加了 `run_full_analysis` 函数的导出
- 现在可以访问: `analysis.run_full_analysis()`

**修改前:**
```python
from vera.analysis.heatmap import create_heatmap, get_top_k_patches
__all__ = ["create_heatmap", "get_top_k_patches"]
```

**修改后:**
```python
from vera.analysis.heatmap import create_heatmap, get_top_k_patches
from vera.analysis.full_analysis import run_full_analysis
__all__ = ["create_heatmap", "get_top_k_patches", "run_full_analysis"]
```

### 2. 创建 `vera/analysis/full_analysis.py`

**功能:**
- 封装完整的三阶段分析逻辑
- Phase 1: 扫描所有文件夹，找到全局 Top 5 heads（按平均分数）
- Phase 2: 使用固定的 Top 5 heads 生成可视化
- Phase 3: 生成全局热力图和统计信息

**主要函数:**

#### `run_full_analysis()`
```python
def run_full_analysis(
    root_dir: str,
    output_folder_name: str = "result",
    top_k_patches: int = 10,
    num_workers: int = 100,
    mode: str = "all",  # "all", "scan", "viz"
    red_lower: Tuple[int, int, int] = (0, 0, 150),
    red_upper: Tuple[int, int, int] = (100, 100, 255),
    kernel_size: Tuple[int, int] = (12, 3),
    dilation_iterations: int = 1,
    debug_box_color: Tuple[int, int, int] = (255, 0, 0),
    debug_box_thickness: int = 2,
    debug_heatmap_alpha: float = 0.6,
    top_k_color: Tuple[int, int, int] = (0, 0, 255),
    save_debug: bool = True,
    save_plots: bool = True,
    sample_count: int = 10
):
    """运行完整的三阶段分析"""
```

**参数说明:**
- `root_dir`: 数据根目录
- `output_folder_name`: 输出文件夹名称（默认 "result"）
- `top_k_patches`: Top K patches 数量（默认 10）
- `num_workers`: 并行处理 worker 数量（默认 100）
- `mode`: 运行模式
  - `"all"`: 运行全部三个阶段（默认）
  - `"scan"`: 仅运行 Phase 1 扫描
  - `"viz"`: 仅运行 Phase 2 可视化

**返回值:**
```python
{
    "num_folders": int,      # 文件夹总数
    "success_count": int,    # 成功处理数量
    "skipped_count": int,    # 跳过数量
    "error_count": int,      # 错误数量
    "errors": List[Tuple[str, str]]  # 错误详情 [(folder, error), ...]
}
```

### 3. 创建示例脚本 `anylasis/data_anylasis_vera_api.py`

**功能:** 使用 vera API 进行完整分析的命令行工具

**用法:**
```bash
# 基本用法 - 运行全部三个阶段
python anylasis/data_anylasis_vera_api.py \
    --root_dir tem/qasper_qwen_img \
    --output_folder result \
    --top_k 10

# 仅运行扫描阶段
python anylasis/data_anylasis_vera_api.py \
    --root_dir tem/qasper_qwen_img \
    --mode scan

# 仅运行可视化阶段
python anylasis/data_anylasis_vera_api.py \
    --root_dir tem/qasper_qwen_img \
    --mode viz
```

### 4. 创建使用指南 `vera/USAGE.md`

**内容包括:**
- 安装说明
- 快速开始示例
- 所有模块的 API 参考
  - models 模块
  - rendering 模块
  - retrieval 模块
  - analysis 模块
- 命令行使用示例
- 完整的实验流程示例
- 迁移指南（从旧代码迁移到新 API）
- 常见问题解答

### 5. 更新测试脚本 `test_vera_api.py`

**新增测试:**
- Test 4: 测试所有模块的 API
- Test 5: 测试 `run_full_analysis` 函数签名

**测试结果:**
```
🎉 All tests passed!
✅ PASS: Imports
✅ PASS: API Signature
✅ PASS: Engine Classes
✅ PASS: All Modules API
✅ PASS: run_full_analysis
```

## 使用示例

### Python API 使用

```python
from vera import analysis

# 运行完整分析
stats = analysis.run_full_analysis(
    root_dir="tem/qasper_qwen_img",
    output_folder_name="result",
    top_k_patches=10,
    num_workers=100
)

print(f"分析完成: {stats['num_folders']} 文件夹")
print(f"成功: {stats['success_count']}")
print(f"错误: {stats['error_count']}")
```

### 命令行使用

```bash
python anylasis/data_anylasis_vera_api.py \
    --root_dir tem/qasper_qwen_img \
    --output_folder result \
    --top_k 10 \
    --num_workers 100
```

## 输出文件

运行完整分析后会生成以下文件:

### 单个文件夹结果
每个 question 文件夹的 `result/` 子文件夹中:
- `{folder_name}_GlobalTop5_heatmap.png` - 全局 Top 5 heads 热力图
- `{folder_name}_GlobalTop5_Top10Patch.png` - Top 10 patches 高亮图

### 全局结果
在根目录的 `{output_folder_name}/` 文件夹中:
- `GLOBAL_attention_heatmap.png` - 全局平均 attention 热力图
- `GLOBAL_attention_matrix_normalized.json` - 归一化的全局 attention 矩阵
- `extracted_evidence.txt` - 提取的证据文本
- `global_top_5_heads.txt` - 全局 Top 5 heads 信息

## 技术特点

### 1. 三阶段分析流程

**Phase 1: 扫描**
- 遍历所有文件夹
- 计算每个 head 的平均分数
- 确定全局 Top 5 heads

**Phase 2: 可视化**
- 使用固定的 Top 5 heads
- 为每个文件夹生成热力图
- 提取 Top 10 patches 对应的文本

**Phase 3: 全局统计**
- 生成全局平均热力图
- 保存归一化 attention 矩阵
- 输出统计信息

### 2. 并行处理

- 使用 `ProcessPoolExecutor` 进行多进程处理
- 默认 100 个 worker，可配置
- 支持大规模数据集的快速分析

### 3. 灵活的模式选择

- `mode="all"`: 运行全部阶段（默认）
- `mode="scan"`: 仅运行扫描，快速找到 Top 5 heads
- `mode="viz"`: 使用已知的 Top 5 heads 生成可视化

### 4. 可配置的参数

支持自定义:
- 颜色范围（red_lower, red_upper）
- 膨胀核大小（kernel_size）
- 边框样式（debug_box_color, debug_box_thickness）
- 热力图透明度（debug_heatmap_alpha）
- Top K patches 数量

## 与旧版本的对比

### 旧版本（`data_anylasis_dev_20_best_5_raw.py`）
- 817 行代码
- 需要直接运行脚本
- 所有逻辑在一个文件中
- 难以复用和集成

### 新版本（vera API）
```python
from vera import analysis
stats = analysis.run_full_analysis(
    root_dir="tem/qasper_qwen_img",
    top_k_patches=10
)
```
- 3 行代码调用核心功能
- 可以作为模块导入使用
- 逻辑封装在 `vera/analysis/full_analysis.py`
- 易于复用和集成

## 向后兼容

### 原有脚本保留

- `anylasis/data_anylasis_dev_20_best_5.py` 仍然可用
- `anylasis/data_anylasis_dev_20_best_5_raw.py` 保留作为参考

### 新增脚本

- `anylasis/data_anylasis_vera_api.py` - 使用 vera API 的新版本

## 文件结构

```
vera/
├── __init__.py                      # 主包入口
├── USAGE.md                         # 使用指南
├── models/                          # 模型模块
│   ├── __init__.py
│   ├── base.py
│   ├── qwen.py
│   └── glm.py
├── rendering/                       # 渲染模块
│   ├── __init__.py
│   └── text_to_image.py
├── retrieval/                       # 检索模块
│   ├── __init__.py
│   ├── qwen_embedding.py
│   └── colpali.py
└── analysis/                        # 分析模块
    ├── __init__.py                  # 导出 run_full_analysis ✓
    ├── heatmap.py                   # 热力图生成
    └── full_analysis.py             # 完整分析逻辑 ✓ 新增

anylasis/
├── data_anylasis_dev_20_best_5.py           # 原有脚本（保留）
├── data_anylasis_dev_20_best_5_raw.py       # 原始版本（参考）
└── data_anylasis_vera_api.py                # vera API 版本 ✓ 新增

test_vera_api.py                    # API 测试脚本 ✓ 更新
```

## 测试验证

### 运行测试

```bash
python test_vera_api.py
```

### 测试结果

```
🎉 All tests passed!
✅ PASS: Imports
✅ PASS: API Signature
✅ PASS: Engine Classes
✅ PASS: All Modules API
✅ PASS: run_full_analysis
```

## 下一步

### 可选的改进

1. **添加单元测试**
   - 为 `run_full_analysis()` 添加更详细的测试
   - 测试不同模式（scan, viz, all）
   - 测试错误处理

2. **性能优化**
   - 优化并行处理逻辑
   - 添加进度条支持
   - 支持断点续传

3. **文档完善**
   - 添加更多使用示例
   - 添加最佳实践指南
   - 添加故障排除指南

4. **功能扩展**
   - 支持自定义 head 选择策略
   - 支持更多可视化选项
   - 支持导出为其他格式（PDF、HTML）

## 总结

✅ **成功将完整的三阶段分析逻辑整合到 VERA 软件包中**

用户现在可以:
- 通过简单的 API 调用运行完整分析
- 使用命令行工具快速处理数据
- 将分析功能集成到自己的脚本中
- 选择性地运行特定阶段（scan/viz/all）

所有测试通过，API 设计符合 Python 包的最佳实践！
