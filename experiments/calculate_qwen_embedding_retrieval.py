"""
Qwen3-VL Embedding 检索脚本 (qasper 版本)
使用 Qwen3-VL-8B-Instruct 的 embedding 模型进行检索
检索出与 question 相关度排名前 K 的句子，提取对应文字

注意：qasper 数据集没有 word_mapping.json，使用 context.txt 进行文本检索
"""

import os
import sys
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# ================= 配置区域 =================
# 路径设置
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ROOT_DIR = "tem/qasper_qwen_img"

# 加载模型配置
MODEL_CONFIG_PATH = os.path.join(ROOT, "config/model_config.json")
if os.path.exists(MODEL_CONFIG_PATH):
    with open(MODEL_CONFIG_PATH, 'r', encoding='utf-8') as f:
        model_config = json.load(f)
    MODEL_PATH = model_config.get("model_path", {}).get("qwen")
    if not MODEL_PATH:
        raise ValueError("model_path.qwen not found in config/model_config.json")
else:
    raise FileNotFoundError(f"Model config file not found: {MODEL_CONFIG_PATH}")

# Top K 句子数量
TOP_K_SENTENCES = 10
# ===========================================


def load_model_and_processor(model_path: str, device: str = "cuda"):
    """加载 Qwen3-VL 模型和处理器"""
    print(f"正在加载模型: {model_path}")
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    ).eval()
    
    processor = AutoProcessor.from_pretrained(
        model_path, 
        trust_remote_code=True
    )
    
    print(f"模型加载完成，设备: {model.device}")
    return model, processor


def get_text_embedding(model, processor, text: str, device: str = "cuda"):
    """
    获取文本的 embedding
    使用 language model 的 embedding layer + mean pooling
    """
    tokenizer = processor.tokenizer
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)
    
    with torch.no_grad():
        embed_layer = model.get_input_embeddings()
        text_embeds = embed_layer(input_ids)  # (1, seq_len, hidden_dim)
        
        # Mean pooling over tokens
        mask_expanded = attention_mask.unsqueeze(-1).expand(text_embeds.size()).float()
        sum_embeddings = torch.sum(text_embeds * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        text_embedding = sum_embeddings / sum_mask  # (1, hidden_dim)
    
    return text_embedding


def get_batch_text_embeddings(model, processor, texts: list, device: str = "cuda", batch_size: int = 32):
    """
    批量获取文本的 embedding
    """
    tokenizer = processor.tokenizer
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        tokens = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)
        
        with torch.no_grad():
            embed_layer = model.get_input_embeddings()
            text_embeds = embed_layer(input_ids)  # (batch, seq_len, hidden_dim)
            
            # Mean pooling over tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(text_embeds.size()).float()
            sum_embeddings = torch.sum(text_embeds * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            batch_embeddings = sum_embeddings / sum_mask  # (batch, hidden_dim)
        
        all_embeddings.append(batch_embeddings)
    
    return torch.cat(all_embeddings, dim=0)


def split_into_sentences(text: str) -> list:
    """
    将文本分割成句子
    """
    # 使用正则表达式分割句子
    # 保留句子结尾的标点符号
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # 过滤空句子和过短的句子
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    return sentences


def find_question_folders(root_dir):
    """
    查找所有包含 input_tokens.json 和 context.txt 的文件夹
    qasper 结构: paper_id/hash_id/ (包含 input_tokens.json, context.txt, gold_evidence.json)
    """
    folders = []
    for root, dirs, files in os.walk(root_dir):
        # 跳过 result 文件夹
        if "result" in root:
            continue
        # 检查必要的文件
        if "input_tokens.json" in files and "context.txt" in files:
            folders.append(root)
    return folders


def extract_question_from_tokens(input_tokens_path, tokenizer=None):
    """
    从 input_tokens.json 中提取问题文本
    """
    if not os.path.exists(input_tokens_path):
        return None
    
    try:
        with open(input_tokens_path, 'r', encoding='utf-8') as f:
            tokens = json.load(f)
        
        # 找到 <|vision_end|> 的位置
        vision_end_idx = None
        for i, token in enumerate(tokens):
            if token == '<|vision_end|>':
                vision_end_idx = i
                break
        
        if vision_end_idx is None:
            return None
        
        # 提取 <|vision_end|> 到 <|im_end|> 之间的 tokens
        question_tokens = []
        for i in range(vision_end_idx + 1, len(tokens)):
            token = tokens[i]
            if token == '<|im_end|>':
                break
            question_tokens.append(token)
        
        # 使用 tokenizer 解码
        if tokenizer is not None:
            token_ids = tokenizer.convert_tokens_to_ids(question_tokens)
            question_text = tokenizer.decode(token_ids, skip_special_tokens=True)
            return question_text.strip()
        
        # 备用方案：手动解析
        question_text = ""
        for token in question_tokens:
            if token.startswith('<|') and token.endswith('|>'):
                continue
            if token.startswith('Ġ'):
                question_text += ' ' + token[1:]
            elif token == 'Ċ':
                question_text += '\n'
            else:
                question_text += token
        
        return question_text.strip()
    
    except Exception as e:
        print(f"Error extracting question: {e}")
        return None


def process_single_question(folder_path, model, processor, device="cuda"):
    """
    处理单个问题文件夹
    
    Args:
        folder_path: 包含 input_tokens.json 和 context.txt 的文件夹路径
        model: Qwen3-VL 模型
        processor: Qwen3-VL 处理器
        device: 设备
    
    Returns:
        (folder_name, success_flag, message)
    """
    try:
        folder_name = os.path.basename(folder_path)
        
        # 获取文件路径
        input_tokens_path = os.path.join(folder_path, "input_tokens.json")
        context_path = os.path.join(folder_path, "context.txt")
        output_path = os.path.join(folder_path, "extracted_evidence_qwen_embedding.txt")
        
        # 检查文件是否存在
        if not os.path.exists(input_tokens_path) or not os.path.exists(context_path):
            return (folder_name, False, "Missing input_tokens.json or context.txt")
        
        # 提取问题文本
        question = extract_question_from_tokens(input_tokens_path, tokenizer=processor.tokenizer)
        if question is None:
            return (folder_name, False, "Failed to extract question")
        
        # 清理问题文本（去掉指令部分）
        display_question = question
        prefix_marker = "Please answer the question based on the document images provided."
        if display_question.startswith(prefix_marker):
            display_question = display_question[len(prefix_marker):].strip()
        cutoff_marker = "Please output your answer **directly**"
        if cutoff_marker in display_question:
            display_question = display_question.split(cutoff_marker)[0].strip()
        
        print(f"\n[{folder_name}] Question: {display_question[:100]}...")
        question = display_question
        
        # 读取 context.txt
        with open(context_path, 'r', encoding='utf-8') as f:
            context = f.read().strip()
        
        if not context:
            return (folder_name, False, "Empty context.txt")
        
        # 将 context 分割成句子
        sentences = split_into_sentences(context)
        
        if len(sentences) == 0:
            return (folder_name, False, "No valid sentences in context")
        
        # 获取问题的 embedding
        question_embedding = get_text_embedding(model, processor, question, device)
        
        # 批量获取所有句子的 embedding
        sentence_embeddings = get_batch_text_embeddings(model, processor, sentences, device)
        
        # 计算相似度
        question_embedding = question_embedding.float()
        sentence_embeddings = sentence_embeddings.float()
        
        # 归一化
        question_norm = F.normalize(question_embedding, p=2, dim=-1)
        sentence_norm = F.normalize(sentence_embeddings, p=2, dim=-1)
        
        # 计算余弦相似度
        similarity = torch.matmul(sentence_norm, question_norm.squeeze(0))  # (num_sentences,)
        similarity = similarity.cpu().numpy()
        
        # 获取 Top K 句子
        k = min(TOP_K_SENTENCES, len(sentences))
        top_k_indices = np.argsort(similarity)[-k:][::-1]  # 降序排列
        
        # 提取 Top K 句子
        top_sentences = [sentences[idx] for idx in top_k_indices]
        extracted_text = "\n".join(top_sentences)
        
        # 保存到文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(extracted_text)
        
        return (folder_name, True, f"Extracted {k} sentences from {len(sentences)} total")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return (os.path.basename(folder_path), False, str(e))


def main():
    """主函数"""
    print("=" * 80)
    print("Qwen3-VL Embedding 文本检索脚本 (qasper 版本)")
    print("=" * 80)
    
    # 查找所有问题文件夹
    folders = find_question_folders(ROOT_DIR)
    print(f"\n找到 {len(folders)} 个待处理的文件夹")
    
    if len(folders) == 0:
        print("未找到任何待处理的文件夹")
        return
    
    # 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = load_model_and_processor(MODEL_PATH, device)
    
    # 统计
    success_count = 0
    fail_count = 0
    
    # 处理每个文件夹
    print(f"\n开始处理...")
    for folder_path in tqdm(folders, desc="Processing"):
        folder_name, success, message = process_single_question(
            folder_path, model, processor, device
        )
        
        if success:
            success_count += 1
        else:
            fail_count += 1
            print(f"  Failed: {folder_name} - {message}")
    
    # 输出统计
    print("\n" + "=" * 80)
    print("处理完成!")
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")
    print(f"  总计: {len(folders)}")
    print("=" * 80)
    print(f"证据文本已保存到各问题目录下的 extracted_evidence_qwen_embedding.txt")


if __name__ == "__main__":
    main()
