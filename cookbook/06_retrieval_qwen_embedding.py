#!/usr/bin/env python3
"""
VERA Cookbook - 06: Qwen Embedding检索

本示例展示如何：
1. 使用Qwen模型的embedding进行检索
2. 将文档切分为句子
3. 计算问题与句子的相似度
4. 返回Top-K相关句子

注意：此示例需要已运行过推理的数据目录
"""

import sys
import json
from pathlib import Path

# 添加项目根目录到路径
ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vera import retrieval


def main():
    print("=" * 60)
    print("VERA Cookbook - 06: Qwen Embedding检索")
    print("=" * 60)

    # ==================== 配置 ====================
    MODEL_PATH = "/data3/guofang/peirongcan/vllm_log/Qwen3-VL-8B-Instruct"

    # 数据目录（需要包含input_tokens.json和context.txt的文件夹）
    # 这里我们创建一个示例目录结构
    DATA_DIR = ROOT / "cookbook" / "output" / "06_qwen_embedding_data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ==================== 1. 准备示例数据 ====================
    print("\n[Step 1] 准备示例数据...")

    # 示例问题
    sample_question = "What is visual evidence retrieval?"

    # 示例文档内容
    sample_context = """Visual Evidence Retrieval and Analysis (VERA) is a comprehensive framework for document understanding.

Attention-based retrieval mechanisms extract relevant information from document images. The system uses large vision-language models to understand both visual and textual content.

Key features include multi-modal document understanding, evidence extraction, and question answering capabilities. The framework supports various document types including academic papers, technical reports, and business documents.

The retrieval process combines multiple approaches: attention-based methods, embedding-based similarity, and visual-textual matching techniques. Each approach has its strengths in different scenarios.

For complex queries, the system can combine multiple retrieval methods to improve accuracy and coverage. The extracted evidence is then used to generate accurate answers to user questions."""

    # 创建context.txt
    context_path = DATA_DIR / "context.txt"
    with open(context_path, 'w', encoding='utf-8') as f:
        f.write(sample_context)

    print(f"✓ 创建了示例context.txt: {context_path}")

    # 创建模拟的input_tokens.json（包含问题）
    # 在实际使用中，这会从模型推理中获取
    input_tokens_path = DATA_DIR / "input_tokens.json"
    mock_tokens = [
        "<|vision_start|>", "<|image|>", "<|vision_end|>",
        "<|im_start|>user",
        "<|vision_start|>", "<|vision_end|>",  # 占位符
        f"Please answer the question based on the document images provided. {sample_question}<|im_end|>",
        "<|im_start|>assistant"
    ]
    with open(input_tokens_path, 'w', encoding='utf-8') as f:
        json.dump(mock_tokens, f, ensure_ascii=False)

    print(f"✓ 创建了示例input_tokens.json: {input_tokens_path}")

    # ==================== 2. 运行Embedding检索 ====================
    print("\n[Step 2] 运行Qwen Embedding检索...")
    print(f"模型路径: {MODEL_PATH}")
    print(f"数据目录: {DATA_DIR}")

    try:
        stats = retrieval.qwen_embedding(
            model_path=MODEL_PATH,
            data_path=str(DATA_DIR),
            save_dir=str(DATA_DIR),
            top_k=5  # 返回Top-5句子
        )

        # ==================== 3. 显示结果 ====================
        print("\n[Step 3] 显示结果...")
        print("-" * 60)

        print(f"\n检索统计:")
        print(f"  总样本数: {stats.get('total', 0)}")
        print(f"  成功: {stats.get('success', 0)}")
        print(f"  失败: {stats.get('failed', 0)}")

        # 读取提取的证据
        evidence_path = DATA_DIR / "extracted_evidence_qwen_embedding.txt"
        if evidence_path.exists():
            with open(evidence_path, 'r', encoding='utf-8') as f:
                extracted_text = f.read()

            print(f"\n提取的Top-5相关句子:")
            print("-" * 60)
            lines = extracted_text.split('\n')
            for i, line in enumerate(lines[:5], 1):
                if line.strip():
                    print(f"{i}. {line}")
            print("-" * 60)

        print("\n✓ 检索完成")

    except Exception as e:
        print(f"\n⚠ 检索过程出错: {e}")
        print("\n这可能是因为:")
        print("  1. 模型路径不正确")
        print("  2. 需要安装transformers和torch")
        print("  3. CUDA不可用或显存不足")
        print("\n提示：这个示例需要实际的模型才能运行")
        print("你可以修改MODEL_PATH指向正确的Qwen模型")

    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)
    print(f"\n数据和结果保存在: {DATA_DIR}")
    print("\n说明:")
    print("  - input_tokens.json: 包含输入的token序列")
    print("  - context.txt: 文档的文本内容")
    print("  - extracted_evidence_qwen_embedding.txt: 提取的相关句子")


if __name__ == "__main__":
    main()
