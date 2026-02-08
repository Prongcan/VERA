import os
import sys
from pathlib import Path
import argparse
import json
import numpy as np
import torch
import gc
from tqdm import tqdm

# --- 路径设置 ---
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 假设 VLM_and_LLM 在 ROOT 下
if str(ROOT / "VLM_and_LLM") not in sys.path:
    sys.path.insert(0, str(ROOT / "VLM_and_LLM"))

# 添加PyTorch内存优化环境变量
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def is_question_processed(save_path, qid, processed_qids):
    """
    检查问题是否已经完全处理过
    Args:
        save_path: 保存路径
        qid: 问题ID
        processed_qids: 已处理的question_id集合
    Returns:
        bool: 是否已经处理过
    """
    # 检查是否在prediction.jsonl中
    if qid in processed_qids:
        return True

    # 检查保存目录是否存在且包含必要的文件
    if not os.path.exists(save_path):
        return False

    # 检查核心输出文件是否存在
    required_files = ["answer.txt", "context.txt", "gold_evidence.json", "rag_status.json"]
    return all(os.path.exists(os.path.join(save_path, f)) for f in required_files)

# --- 模块导入 ---
from models_thinking.wrapper_glm_thinking import GlmEngine_img_no_eager_entropy, InferenceConfig
from data.dataset_loader import DatasetUtils
# 导入文本转图片函数
from VLM_and_LLM.Text2img import process_single_text_evidence

# --- 加载模型配置 ---
MODEL_CONFIG_PATH = os.path.join(ROOT, "config/model_config.json")
if os.path.exists(MODEL_CONFIG_PATH):
    with open(MODEL_CONFIG_PATH, 'r', encoding='utf-8') as f:
        model_config = json.load(f)
    DEFAULT_MODEL_PATH = model_config.get("model_path", {}).get("glm")
    if not DEFAULT_MODEL_PATH:
        raise ValueError("model_path.glm not found in config/model_config.json")
else:
    raise FileNotFoundError(f"Model config file not found: {MODEL_CONFIG_PATH}")

def main():
    parser = argparse.ArgumentParser()
    # 模型路径
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    # 数据集路径
    parser.add_argument("--data_path", type=str, default="data/qasper/qasper-test-sample.json")
    # 结果保存路径
    parser.add_argument("--save_dir", type=str, default="tem/qasper_glm_img_RAG_entropy_new")

    # --- 渲染配置文件路径 ---
    parser.add_argument("--config_path", type=str,
                        default="config/config_en.json",
                        help="Path to the render configuration JSON file")
    
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of papers to process")
    parser.add_argument("--font_path", type=str, default=None, help="Override font path in config if needed")
    
    args = parser.parse_args()

    # --- 1. 加载渲染配置 ---
    print(f"Loading render config from: {args.config_path}")
    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f"Config file not found: {args.config_path}")
        
    with open(args.config_path, 'r', encoding='utf-8') as f:
        render_config = json.load(f)

    # 如果命令行指定了字体，覆盖配置文件中的设置
    if args.font_path:
        print(f"Overriding font path with: {args.font_path}")
        render_config["font_path"] = args.font_path

    # --- 2. 初始化引擎 ---
    config = InferenceConfig(
        model_path=args.model_path, 
        dataset_path=args.data_path, 
        save_base_dir=args.save_dir
    )
    
    print(f"Loading model from {args.model_path}...")
    engine = GlmEngine_img_no_eager_entropy(config)
    
    print("Loading dataset...")
    papers = DatasetUtils.load_qasper(args.data_path)
    print(f"Loaded {len(papers)} papers.")

    # --- 新增部分：初始化 prediction.jsonl 文件 ---
    # 确保保存目录存在
    os.makedirs(args.save_dir, exist_ok=True)
    prediction_file_path = os.path.join(args.save_dir, "prediction.jsonl")

    # 加载已有的预测结果，避免重复处理
    processed_qids = set()
    if os.path.exists(prediction_file_path):
        with open(prediction_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        pred_entry = json.loads(line.strip())
                        processed_qids.add(pred_entry.get("question_id", ""))
                    except json.JSONDecodeError:
                        continue
        print(f"Found {len(processed_qids)} already processed questions.")
    else:
        print("No existing prediction file found. Starting fresh.")

    print(f"Prediction file will be saved to: {prediction_file_path}")
    # --------------------------------------------

    # --- 3. 开始循环处理 ---
    processed_count = 0
    skipped_count = 0

    # 初始化待处理任务列表
    pending_tasks = []

    for paper in tqdm(papers[:args.limit]):
        pid, full_text_context = DatasetUtils.build_context(paper, config.max_context_chars)
        
        # 遍历该论文下的所有问题
        for qa in paper.get("qas", []):
            qid = qa.get("question_id", "nan")
            question_text = qa.get("question", "")

            # 提取 Golden Evidence
            gold_evidence = DatasetUtils.extract_evidence(qa)

            # 定义保存路径
            save_path = os.path.join(
                args.save_dir,
                DatasetUtils.safe_filename(pid),
                DatasetUtils.safe_filename(qid)
            )

            # 检查是否已经处理过
            if is_question_processed(save_path, qid, processed_qids):
                skipped_count += 1
                continue

            os.makedirs(save_path, exist_ok=True)
            processed_count += 1
            
            # 图片保存目录
            img_output_dir = os.path.join(save_path, "rendered_images")
            os.makedirs(img_output_dir, exist_ok=True)

            # --- 4. 文本转图片 ---
            # print(f"Rendering images for Question {qid}...")
            
            image_paths = process_single_text_evidence(
                txt=full_text_context,
                output_root=img_output_dir,
                config=render_config, 
                evidence_text=gold_evidence 
            )

            if not image_paths:
                print(f"Warning: Image generation failed for {pid}-{qid}. Skipping.")
                continue

            # --- 5. 运行推理 ---
            prompt_context = "Please answer the question based on the document images provided."
            
            try:
                res = engine.run_entropy_rag(prompt_context, question_text, image_paths, "qasper")

                # 不等待VLLM任务完成，直接保存占位符答案
                # VLLM任务信息会保存在文件中，后续可以异步获取结果

                # --- 6. 保存结果 ---

                # 原始文本
                with open(os.path.join(save_path, "context.txt"), "w") as f:
                    f.write(full_text_context)

                # 答案
                with open(os.path.join(save_path, "answer.txt"), "w") as f:
                    f.write(res["answer"])

                # 关键返回结果落盘到 tem 目录
                # 熵跟踪数据
                with open(os.path.join(save_path, "logits_entropy_trace.json"), "w", encoding="utf-8") as f:
                    json.dump(res.get("logits_entropy_trace", []), f, ensure_ascii=False, indent=2)
                # 输入tokens
                if res.get("input_tokens"):
                    with open(os.path.join(save_path, "input_tokens.json"), "w", encoding="utf-8") as f:
                        json.dump(res["input_tokens"], f, ensure_ascii=False, indent=2)

                # 生成的token IDs
                if res.get("generated_token_ids"):
                    with open(os.path.join(save_path, "generated_token_ids.json"), "w", encoding="utf-8") as f:
                        json.dump(res["generated_token_ids"], f, ensure_ascii=False, indent=2)

                # VLLM任务信息（用于异步处理）
                if res.get("vllm_task_info"):
                    with open(os.path.join(save_path, "vllm_task_info.json"), "w", encoding="utf-8") as f:
                        json.dump(res["vllm_task_info"], f, ensure_ascii=False, indent=2)

                    # 保存任务ID到全局任务列表，用于后续收集结果
                    pending_tasks.append({
                        "task_id": res["vllm_task_info"]["task_id"],
                        "save_path": save_path,
                        "qid": qid
                    })

                # RAG相关信息
                if res.get("rag_info"):
                    with open(os.path.join(save_path, "rag_info.txt"), "w", encoding="utf-8") as f:
                        f.write(res["rag_info"])

                # 高熵检测状态
                high_entropy_info = {}
                if "logits_entropy_trace" in res and res["logits_entropy_trace"]:
                    # 从熵轨迹中提取高熵信息
                    max_entropy = max(res["logits_entropy_trace"])
                    max_entropy_idx = res["logits_entropy_trace"].index(max_entropy)
                    high_entropy_info = {
                        "max_entropy": max_entropy,
                        "max_entropy_step": max_entropy_idx,
                        "entropy_threshold": 4.0,  # 基于_detect_first_high的逻辑
                        "total_steps": len(res["logits_entropy_trace"])
                    }

                if high_entropy_info:
                    with open(os.path.join(save_path, "high_entropy_info.json"), "w", encoding="utf-8") as f:
                        json.dump(high_entropy_info, f, ensure_ascii=False, indent=2)

                # RAG处理状态
                rag_status = {
                    "rag_applied": bool(res.get("rag_info", "").strip()),
                    "input_len": res.get("input_len", None),
                    "max_new_tokens": config.max_new_tokens,
                    "attention_data_released": res.get("attention_data") is None
                }
                with open(os.path.join(save_path, "rag_status.json"), "w", encoding="utf-8") as f:
                    json.dump(rag_status, f, ensure_ascii=False, indent=2)
                
                # Golden Evidence
                with open(os.path.join(save_path, "gold_evidence.json"), "w", encoding='utf-8') as f:
                    json.dump(gold_evidence, f, ensure_ascii=False, indent=2)

                # --- 新增部分：追加写入 prediction.jsonl ---
                # 注意：目前的 prompt 并没有要求模型显式输出 evidence，所以 predicted_evidence 暂时留空。
                # 如果你的模型输出了 evidence，请将 "" 替换为 res["evidence"] 或相应的变量。
                pred_entry = {
                    "question_id": qid,
                    "predicted_answer": res["answer"],
                    "predicted_evidence": ""  # Qasper 标准通常是列表，但这里按你要求设为字符串
                }
                
                with open(prediction_file_path, "a", encoding="utf-8") as f_jsonl:
                    f_jsonl.write(json.dumps(pred_entry, ensure_ascii=False) + "\n")
                # -----------------------------------------------

                # 仅保存当前返回里存在的字段，未返回的字段不写盘
            
            except Exception as e:
                print(f"Error during inference for {pid}-{qid}: {e}")
                import traceback
                traceback.print_exc()

            # 垃圾回收
            gc.collect()
            torch.cuda.empty_cache()

        # --- 每个循环结束时检查并更新已完成的异步任务 ---
        if pending_tasks:
            # 复制一份待处理任务列表，避免在遍历时修改
            tasks_to_check = pending_tasks.copy()

            for task_info in tasks_to_check:
                task_id = task_info["task_id"]
                save_path = task_info["save_path"]
                qid = task_info["qid"]

                try:
                    # 尝试获取最终结果，不阻塞（设置很短的超时时间）
                    final_answer = engine.get_vllm_result(task_id, timeout=0.1)  # 0.1秒超时，不阻塞

                    if final_answer:
                        # 更新答案文件
                        with open(os.path.join(save_path, "answer.txt"), "w") as f:
                            f.write(final_answer)

                        # 更新prediction.jsonl文件
                        # 首先读取现有的预测结果
                        predictions = []
                        if os.path.exists(prediction_file_path):
                            with open(prediction_file_path, 'r', encoding='utf-8') as f:
                                for line in f:
                                    if line.strip():
                                        try:
                                            pred_entry = json.loads(line.strip())
                                            predictions.append(pred_entry)
                                        except json.JSONDecodeError:
                                            continue

                        # 更新对应的预测结果
                        for pred in predictions:
                            if pred.get("question_id") == qid:
                                pred["predicted_answer"] = final_answer
                                break

                        # 重新写入prediction.jsonl
                        with open(prediction_file_path, "w", encoding="utf-8") as f_jsonl:
                            for pred in predictions:
                                f_jsonl.write(json.dumps(pred, ensure_ascii=False) + "\n")

                        # 从待处理列表中移除已完成的任务
                        pending_tasks.remove(task_info)
                        print(f"  ✓ 已更新任务 {task_id} (QID: {qid}) 的最终结果")

                except Exception as e:
                    # 不打印错误信息，避免干扰进度条
                    pass

    # --- 4. 最终收集剩余异步VLLM任务的最终结果 ---
    if pending_tasks:
        print(f"\n最终收集剩余 {len(pending_tasks)} 个异步VLLM任务的最终结果...")
        completed_tasks = 0

        for task_info in pending_tasks:
            task_id = task_info["task_id"]
            save_path = task_info["save_path"]
            qid = task_info["qid"]

            try:
                # 尝试获取最终结果，设置较长的超时时间
                final_answer = engine.get_vllm_result(task_id, timeout=120.0)  # 等待最多2分钟

                if final_answer:
                    # 更新答案文件
                    with open(os.path.join(save_path, "answer.txt"), "w") as f:
                        f.write(final_answer)

                    # 更新prediction.jsonl文件
                    # 首先读取现有的预测结果
                    predictions = []
                    if os.path.exists(prediction_file_path):
                        with open(prediction_file_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                if line.strip():
                                    try:
                                        pred_entry = json.loads(line.strip())
                                        predictions.append(pred_entry)
                                    except json.JSONDecodeError:
                                        continue

                    # 更新对应的预测结果
                    for pred in predictions:
                        if pred.get("question_id") == qid:
                            pred["predicted_answer"] = final_answer
                            break

                    # 重新写入prediction.jsonl
                    with open(prediction_file_path, "w", encoding="utf-8") as f_jsonl:
                        for pred in predictions:
                            f_jsonl.write(json.dumps(pred, ensure_ascii=False) + "\n")

                    completed_tasks += 1
                    print(f"  ✓ 任务 {task_id} (QID: {qid}) 完成")
                else:
                    print(f"  ✗ 任务 {task_id} (QID: {qid}) 未完成，继续使用占位符")

            except Exception as e:
                print(f"  ✗ 收集任务 {task_id} (QID: {qid}) 结果时出错: {e}")

        print(f"最终异步任务收集完成: {completed_tasks}/{len(pending_tasks)} 成功")

    # --- 最终统计 ---
    print(f"\n{'='*50}")
    print("处理完成统计:")
    print(f"  总论文数: {len(papers[:args.limit])}")
    print(f"  已跳过: {skipped_count}")
    print(f"  新处理: {processed_count}")
    print(f"  预测文件: {prediction_file_path}")
    print(f"{'='*50}")

    # --- 清理VLLM异步任务 ---
    print("正在清理VLLM异步任务...")
    engine.cleanup_completed_futures()
    engine.shutdown_vllm_executor()
    print("VLLM线程池已关闭")

if __name__ == "__main__":
    main()