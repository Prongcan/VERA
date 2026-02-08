"""
Retrieval Evaluation Script
评估不同检索方法的 extracted_evidence 与 gold_evidence.json 之间的检索效果

支持的检索方法:
- Vera: extracted_evidence.txt
- ColPali: extracted_evidence_colpali.txt
- Qwen Embedding: extracted_evidence_qwen_embedding.txt

Metrics:
- Precision: 提取文本中有多少比例与 gold evidence 重叠
- Recall: gold evidence 中有多少比例被提取到
- F1: Precision 和 Recall 的调和平均
"""

import os
import json
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import numpy as np

# ================= 配置区域 =================
ROOT_DIR = "tem/qasper_qwen_img"
# ===========================================


def tokenize(text):
    """简单的分词：转小写，按非字母数字字符分割"""
    if not text:
        return []
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def calculate_prf(gold_tokens, extracted_tokens):
    """
    计算 Precision, Recall, F1
    
    Args:
        gold_tokens: gold evidence 的 token 列表
        extracted_tokens: extracted evidence 的 token 列表
    
    Returns:
        precision, recall, f1
    """
    gold_counter = Counter(gold_tokens)
    extracted_counter = Counter(extracted_tokens)
    
    if not gold_counter or not extracted_counter:
        return 0.0, 0.0, 0.0
    
    # 计算重叠的 token 数量
    overlap = 0
    for token, count in extracted_counter.items():
        overlap += min(count, gold_counter.get(token, 0))
    
    # Precision = 重叠 / 提取的总数
    precision = overlap / sum(extracted_counter.values()) if sum(extracted_counter.values()) > 0 else 0
    
    # Recall = 重叠 / gold 的总数
    recall = overlap / sum(gold_counter.values()) if sum(gold_counter.values()) > 0 else 0
    
    # F1 = 2 * P * R / (P + R)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1


def find_question_folders(root_dir):
    """查找所有包含 gold_evidence.json 的文件夹"""
    folders = []
    for root, dirs, files in os.walk(root_dir):
        if "gold_evidence.json" in files:
            folders.append(root)
    return folders


def evaluate_single_sample(folder_path):
    """
    评估单个样本，同时评估 Vera、ColPali 和 Qwen Embedding 三种方法

    Returns:
        dict with metrics for all three methods or None if files missing
    """
    gold_path = os.path.join(folder_path, "gold_evidence.json")
    original_path = os.path.join(folder_path, "extracted_evidence.txt")
    colpali_path = os.path.join(folder_path, "extracted_evidence_colpali.txt")
    qwen_path = os.path.join(folder_path, "extracted_evidence_qwen_embedding.txt")

    # 检查 gold_evidence 是否存在
    if not os.path.exists(gold_path):
        return None

    # 检查至少有一种检索方法的结果
    has_original = os.path.exists(original_path)
    has_colpali = os.path.exists(colpali_path)
    has_qwen = os.path.exists(qwen_path)

    if not has_original and not has_colpali and not has_qwen:
        return {"folder": os.path.basename(folder_path), "status": "no_extracted_evidence"}

    try:
        # 加载 gold evidence
        with open(gold_path, 'r', encoding='utf-8') as f:
            gold_data = json.load(f)

        # 把所有 gold evidence 合并成一个字符串
        if isinstance(gold_data, list):
            gold_text = " ".join([str(item) for item in gold_data])
        else:
            gold_text = str(gold_data)

        if not gold_text.strip():
            return {"folder": os.path.basename(folder_path), "status": "empty_gold"}

        gold_tokens = tokenize(gold_text)
        result = {
            "folder": os.path.basename(folder_path),
            "status": "success",
            "gold_token_count": len(gold_tokens),
        }

        # 评估 Vera
        if has_original:
            with open(original_path, 'r', encoding='utf-8') as f:
                original_text = f.read().strip()

            if original_text:
                original_tokens = tokenize(original_text)
                precision, recall, f1 = calculate_prf(gold_tokens, original_tokens)
                result["vera"] = {
                    "extracted_token_count": len(original_tokens),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            else:
                result["vera"] = {"status": "empty_extracted"}
        else:
            result["vera"] = {"status": "file_not_found"}

        # 评估 ColPali
        if has_colpali:
            with open(colpali_path, 'r', encoding='utf-8') as f:
                colpali_text = f.read().strip()

            if colpali_text:
                colpali_tokens = tokenize(colpali_text)
                precision, recall, f1 = calculate_prf(gold_tokens, colpali_tokens)
                result["colpali"] = {
                    "extracted_token_count": len(colpali_tokens),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            else:
                result["colpali"] = {"status": "empty_extracted"}
        else:
            result["colpali"] = {"status": "file_not_found"}

        # 评估 Qwen Embedding
        if has_qwen:
            with open(qwen_path, 'r', encoding='utf-8') as f:
                qwen_text = f.read().strip()

            if qwen_text:
                qwen_tokens = tokenize(qwen_text)
                precision, recall, f1 = calculate_prf(gold_tokens, qwen_tokens)
                result["qwen_embedding"] = {
                    "extracted_token_count": len(qwen_tokens),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            else:
                result["qwen_embedding"] = {"status": "empty_extracted"}
        else:
            result["qwen_embedding"] = {"status": "file_not_found"}

        return result

    except Exception as e:
        return {"folder": os.path.basename(folder_path), "status": f"error: {str(e)}"}


def main():
    print("="*80)
    print("RETRIEVAL EVALUATION")
    print("Comparing Vera vs ColPali vs Qwen Embedding vs Gold Evidence")
    print("="*80)

    # 查找所有样本文件夹
    folders = find_question_folders(ROOT_DIR)
    print(f"\nFound {len(folders)} samples with gold_evidence.json")

    # 并行评估
    num_workers = 50
    results = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(evaluate_single_sample, folders),
            total=len(folders),
            desc="Evaluating"
        ))

    # 统计结果
    success_results = [r for r in results if r and r.get("status") == "success"]
    no_extracted = [r for r in results if r and r.get("status") == "no_extracted_evidence"]
    errors = [r for r in results if r and r.get("status", "").startswith("error")]

    # 统计各方法的有效结果
    vera_results = [r for r in success_results if "vera" in r and "f1" in r["vera"]]
    colpali_results = [r for r in success_results if "colpali" in r and "f1" in r["colpali"]]
    qwen_results = [r for r in success_results if "qwen_embedding" in r and "f1" in r["qwen_embedding"]]

    print(f"\n{'='*80}")
    print("STATISTICS")
    print(f"{'='*80}")
    print(f"Total samples: {len(folders)}")
    print(f"Successfully evaluated: {len(success_results)}")
    print(f"  - Vera: {len(vera_results)} samples")
    print(f"  - ColPali: {len(colpali_results)} samples")
    print(f"  - Qwen Embedding: {len(qwen_results)} samples")
    print(f"No extracted evidence: {len(no_extracted)}")
    print(f"Errors: {len(errors)}")

    if not success_results:
        print("\nNo successful evaluations. Please run retrieval scripts first to extract evidence.")
        return

    # 计算 Vera 平均分数
    if vera_results:
        vera_avg_precision = np.mean([r["vera"]["precision"] for r in vera_results])
        vera_avg_recall = np.mean([r["vera"]["recall"] for r in vera_results])
        vera_avg_f1 = np.mean([r["vera"]["f1"] for r in vera_results])
    else:
        vera_avg_precision = vera_avg_recall = vera_avg_f1 = 0.0

    # 计算 ColPali 平均分数
    if colpali_results:
        colpali_avg_precision = np.mean([r["colpali"]["precision"] for r in colpali_results])
        colpali_avg_recall = np.mean([r["colpali"]["recall"] for r in colpali_results])
        colpali_avg_f1 = np.mean([r["colpali"]["f1"] for r in colpali_results])
    else:
        colpali_avg_precision = colpali_avg_recall = colpali_avg_f1 = 0.0

    # 计算 Qwen Embedding 平均分数
    if qwen_results:
        qwen_avg_precision = np.mean([r["qwen_embedding"]["precision"] for r in qwen_results])
        qwen_avg_recall = np.mean([r["qwen_embedding"]["recall"] for r in qwen_results])
        qwen_avg_f1 = np.mean([r["qwen_embedding"]["f1"] for r in qwen_results])
    else:
        qwen_avg_precision = qwen_avg_recall = qwen_avg_f1 = 0.0

    # 打印对比结果表格
    print(f"\n{'='*80}")
    print("AVERAGE SCORES COMPARISON")
    print(f"{'='*80}")

    print(f"\n{'Method':<20} | {'Precision':<12} | {'Recall':<12} | {'F1':<12}")
    print("-" * 65)
    if vera_results:
        print(f"{'Vera':<20} | {vera_avg_precision:<12.4f} | {vera_avg_recall:<12.4f} | {vera_avg_f1:<12.4f}")
    if colpali_results:
        print(f"{'ColPali':<20} | {colpali_avg_precision:<12.4f} | {colpali_avg_recall:<12.4f} | {colpali_avg_f1:<12.4f}")
    if qwen_results:
        print(f"{'Qwen Embedding':<20} | {qwen_avg_precision:<12.4f} | {qwen_avg_recall:<12.4f} | {qwen_avg_f1:<12.4f}")

    # 打印解释
    print(f"\n{'='*80}")
    print("INTERPRETATION")
    print(f"{'='*80}")

    if vera_results:
        print(f"\nVera:")
        print(f"  - Precision: {vera_avg_precision*100:.1f}% 的提取文本与 gold evidence 重叠")
        print(f"  - Recall: {vera_avg_recall*100:.1f}% 的 gold evidence 被成功提取")
        print(f"  - F1: 综合指标为 {vera_avg_f1*100:.1f}%")

    if colpali_results:
        print(f"\nColPali:")
        print(f"  - Precision: {colpali_avg_precision*100:.1f}% 的提取文本与 gold evidence 重叠")
        print(f"  - Recall: {colpali_avg_recall*100:.1f}% 的 gold evidence 被成功提取")
        print(f"  - F1: 综合指标为 {colpali_avg_f1*100:.1f}%")

    if qwen_results:
        print(f"\nQwen Embedding:")
        print(f"  - Precision: {qwen_avg_precision*100:.1f}% 的提取文本与 gold evidence 重叠")
        print(f"  - Recall: {qwen_avg_recall*100:.1f}% 的 gold evidence 被成功提取")
        print(f"  - F1: 综合指标为 {qwen_avg_f1*100:.1f}%")

    # 对比分析
    if vera_results and colpali_results and qwen_results:
        print(f"\n{'='*80}")
        print("COMPARISON")
        print(f"{'='*80}")

        # 找出 F1 最高的方法
        methods = [
            ("Vera", vera_avg_f1),
            ("ColPali", colpali_avg_f1),
            ("Qwen Embedding", qwen_avg_f1)
        ]
        best_method = max(methods, key=lambda x: x[1])
        print(f"整体表现最佳：{best_method[0]} (F1={best_method[1]:.4f})")

        # 比较 ColPali 和 Vera
        f1_diff_col = colpali_avg_f1 - vera_avg_f1
        if abs(f1_diff_col) > 0.01:
            better = "ColPali" if f1_diff_col > 0 else "Vera"
            print(f"{better} vs Vera F1 差距：{abs(f1_diff_col)*100:.2f}%")

        # 比较 Qwen 和 Vera
        f1_diff_qwen = qwen_avg_f1 - vera_avg_f1
        if abs(f1_diff_qwen) > 0.01:
            better = "Qwen Embedding" if f1_diff_qwen > 0 else "Vera"
            print(f"{better} vs Vera F1 差距：{abs(f1_diff_qwen)*100:.2f}%")

    # 保存详细结果到 JSON
    output_path = os.path.join(ROOT_DIR, "retrieval_evaluation_results.json")
    output_data = {
        "summary": {
            "total_samples": len(folders),
            "successful": len(success_results),
            "vera_samples": len(vera_results),
            "colpali_samples": len(colpali_results),
            "qwen_embedding_samples": len(qwen_results),
            "no_extracted": len(no_extracted),
            "errors": len(errors),
        },
        "average_scores": {}
    }

    if vera_results:
        output_data["average_scores"]["vera"] = {
            "precision": vera_avg_precision,
            "recall": vera_avg_recall,
            "f1": vera_avg_f1,
        }

    if colpali_results:
        output_data["average_scores"]["colpali"] = {
            "precision": colpali_avg_precision,
            "recall": colpali_avg_recall,
            "f1": colpali_avg_f1,
        }

    if qwen_results:
        output_data["average_scores"]["qwen_embedding"] = {
            "precision": qwen_avg_precision,
            "recall": qwen_avg_recall,
            "f1": qwen_avg_f1,
        }

    output_data["detailed_results"] = success_results

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to: {output_path}")

    # 打印 Top 10 最佳和最差样本（按 F1 分数）
    if len(vera_results) >= 10:
        sorted_vera = sorted(vera_results, key=lambda x: x['vera']['f1'], reverse=True)

        print(f"\n{'='*80}")
        print("TOP 10 BEST Vera F1 SAMPLES")
        print(f"{'='*80}")
        for i, r in enumerate(sorted_vera[:10], 1):
            metrics = r['vera']
            print(f"{i}. {r['folder']}: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")

        print(f"\n{'='*80}")
        print("TOP 10 WORST Vera F1 SAMPLES")
        print(f"{'='*80}")
        for i, r in enumerate(sorted_vera[-10:], 1):
            metrics = r['vera']
            print(f"{i}. {r['folder']}: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")

    if len(colpali_results) >= 10:
        sorted_colpali = sorted(colpali_results, key=lambda x: x['colpali']['f1'], reverse=True)

        print(f"\n{'='*80}")
        print("TOP 10 BEST ColPali F1 SAMPLES")
        print(f"{'='*80}")
        for i, r in enumerate(sorted_colpali[:10], 1):
            metrics = r['colpali']
            print(f"{i}. {r['folder']}: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")

        print(f"\n{'='*80}")
        print("TOP 10 WORST ColPali F1 SAMPLES")
        print(f"{'='*80}")
        for i, r in enumerate(sorted_colpali[-10:], 1):
            metrics = r['colpali']
            print(f"{i}. {r['folder']}: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")

    if len(qwen_results) >= 10:
        sorted_qwen = sorted(qwen_results, key=lambda x: x['qwen_embedding']['f1'], reverse=True)

        print(f"\n{'='*80}")
        print("TOP 10 BEST Qwen Embedding F1 SAMPLES")
        print(f"{'='*80}")
        for i, r in enumerate(sorted_qwen[:10], 1):
            metrics = r['qwen_embedding']
            print(f"{i}. {r['folder']}: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")

        print(f"\n{'='*80}")
        print("TOP 10 WORST Qwen Embedding F1 SAMPLES")
        print(f"{'='*80}")
        for i, r in enumerate(sorted_qwen[-10:], 1):
            metrics = r['qwen_embedding']
            print(f"{i}. {r['folder']}: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")

    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
