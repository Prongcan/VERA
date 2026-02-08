#!/usr/bin/env python3
"""
VERA Cookbook - 快速运行所有示例

使用方法:
    python run_all_examples.py --example 01    # 只运行01示例
    python run_all_examples.py --example all    # 运行所有示例
    python run_all_examples.py --skip-model    # 跳过需要模型的示例
"""

import sys
import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).parent


def run_example(example_num: str, description: str):
    """运行单个示例"""
    print("\n" + "=" * 70)
    print(f"运行示例 {example_num}: {description}")
    print("=" * 70)

    example_file = ROOT / f"{example_num}_*.py"
    matching_files = list(ROOT.glob(example_file))

    if not matching_files:
        print(f"❌ 找不到示例文件: {example_num}")
        return False

    script_path = matching_files[0]

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(ROOT),
            capture_output=False,
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="VERA Cookbook - 运行示例",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例说明:
  01 - 基础模型推理
  02 - 掩盖注意力头的推理
  03 - 文本渲染为图像（基础版）
  04 - 文本渲染为图像（带evidence高亮）
  05 - 基于注意力的检索
  06 - Qwen Embedding检索
  07 - 热力图生成
  08 - 完整分析流程
  09 - 端到端RAG示例

使用示例:
  python run_all_examples.py --example 03        # 只运行文本渲染示例
  python run_all_examples.py --example all       # 运行所有示例
  python run_all_examples.py --skip-model        # 跳过需要模型的示例
        """
    )

    parser.add_argument(
        "--example",
        type=str,
        default="help",
        help="示例编号（01-09）或'all'运行所有示例"
    )

    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="跳过需要加载模型的示例（01, 02, 06, 09）"
    )

    args = parser.parse_args()

    # 定义所有示例
    examples = {
        "01": ("01_models_basic_inference", "基础模型推理", True),
        "02": ("02_models_masked_inference", "掩盖注意力头的推理", True),
        "03": ("03_rendering_basic", "文本渲染为图像（基础版）", False),
        "04": ("04_rendering_with_evidence", "文本渲染为图像（带evidence高亮）", False),
        "05": ("05_retrieval_attention", "基于注意力的检索", False),
        "06": ("06_retrieval_qwen_embedding", "Qwen Embedding检索", True),
        "07": ("07_analysis_heatmap", "热力图生成", False),
        "08": ("08_analysis_full_pipeline", "完整分析流程", False),
        "09": ("09_end_to_end_rag", "端到端RAG示例", True),
    }

    # 显示帮助
    if args.example == "help":
        parser.print_help()
        print("\n" + "=" * 70)
        print("示例列表:")
        print("=" * 70)
        for num, (name, desc, need_model) in examples.items():
            model_flag = " [需要模型]" if need_model else ""
            print(f"  {num} - {desc}{model_flag}")
        return

    # 运行特定示例或所有示例
    if args.example == "all":
        print("\n" + "=" * 70)
        print("VERA Cookbook - 运行所有示例")
        print("=" * 70)

        results = {}
        for num, (name, desc, need_model) in examples.items():
            if args.skip_model and need_model:
                print(f"\n跳过示例 {num}（需要模型）: {desc}")
                continue

            success = run_example(num, desc)
            results[num] = success

        # 总结
        print("\n" + "=" * 70)
        print("运行总结")
        print("=" * 70)

        for num, success in results.items():
            status = "✓ 成功" if success else "❌ 失败"
            print(f"  示例 {num}: {status}")

    else:
        # 运行单个示例
        example_num = args.example.zfill(2)  # 确保是两位数，如 "01"
        if example_num not in examples:
            print(f"❌ 无效的示例编号: {args.example}")
            print(f"   可用的示例: {', '.join(examples.keys())}")
            return

        name, desc, need_model = examples[example_num]

        if args.skip_model and need_model:
            print(f"⚠ 示例 {example_num} 需要加载模型，但指定了 --skip-model")
            return

        run_example(example_num, desc)


if __name__ == "__main__":
    main()
