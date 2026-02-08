#!/usr/bin/env python3
"""
使用 VERA API 进行完整的三阶段分析
Phase 1: 扫描所有文件夹找到全局 Top 5 heads
Phase 2: 使用固定的 Top 5 heads 生成可视化
Phase 3: 生成全局热力图和统计信息
"""

from vera import analysis
import argparse

def main():
    parser = argparse.ArgumentParser(description='VERA 完整分析 - 使用 vera API')
    parser.add_argument('--root_dir', type=str, default='tem/qasper_qwen_img',
                        help='根目录，包含所有要分析的 question 文件夹 (默认: tem/qasper_qwen_img)')
    parser.add_argument('--output_folder', type=str, default='result',
                        help='输出文件夹名称 (默认: result)')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Top K patches (默认: 10)')
    parser.add_argument('--num_workers', type=int, default=100,
                        help='并行工作进程数 (默认: 100)')
    parser.add_argument('--mode', type=str, default='all', choices=['scan', 'viz', 'all'],
                        help='运行模式: scan(仅扫描), viz(仅可视化), all(全部,默认)')

    args = parser.parse_args()

    print("=" * 60)
    print("VERA 完整分析 - 使用 vera API")
    print("=" * 60)
    print(f"根目录: {args.root_dir}")
    print(f"输出文件夹: {args.output_folder}")
    print(f"Top K: {args.top_k}")
    print(f"工作进程数: {args.num_workers}")
    print(f"运行模式: {args.mode}")
    print("=" * 60)

    # 调用 vera API 运行完整分析
    stats = analysis.run_full_analysis(
        root_dir=args.root_dir,
        output_folder_name=args.output_folder,
        top_k_patches=args.top_k,
        num_workers=args.num_workers,
        mode=args.mode,
        # 可选参数 - 使用默认值
        # heatmap_color='red',
        # top_k_color='yellow',
        # kernel_size=(5, 5),
        # heatmap_alpha=0.5,
        # target_size=1024,
        # dpi=100
    )

    # 打印统计信息
    print("\n" + "=" * 60)
    print("分析完成！统计信息：")
    print("=" * 60)
    print(f"成功: {stats.get('success_count', 0)}")
    print(f"跳过: {stats.get('skipped_count', 0)}")
    print(f"错误: {stats.get('error_count', 0)}")
    if stats.get('errors'):
        print("\n错误详情:")
        for folder, error in stats['errors'][:5]:  # 只显示前5个错误
            print(f"  - {folder}: {error}")

    print("\n输出文件位置:")
    print(f"  - 单个文件夹结果: {args.root_dir}/*/{args.output_folder}/")
    if args.mode in ['all', 'scan']:
        print(f"  - 全局热力图: {args.root_dir}/{args.output_folder}/GLOBAL_attention_heatmap.png")
        print(f"  - 全局矩阵: {args.root_dir}/{args.output_folder}/GLOBAL_attention_matrix_normalized.json")
        print(f"  - 提取的证据: {args.root_dir}/{args.output_folder}/extracted_evidence.txt")
    print("=" * 60)

if __name__ == "__main__":
    main()
