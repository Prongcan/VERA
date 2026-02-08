"""
Qwen embedding-based retrieval for VERA
使用 Qwen3-VL 模型的 embedding 进行文本检索
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def qwen_embedding(
    model_path: str,
    data_path: str,
    save_dir: str,
    top_k: int = 10
) -> dict:
    """
    使用 Qwen embedding 进行检索

    Args:
        model_path: Qwen 模型路径
        data_path: 数据集路径（包含 input_tokens.json 和 context.txt 的文件夹）
        save_dir: 保存目录
        top_k: 返回 Top K 个句子

    Returns:
        dict: 检索结果统计信息

    Examples:
        >>> stats = retrieval.qwen_embedding(
        ...     model_path="/path/to/Qwen3-VL-8B-Instruct",
        ...     data_path="tem/qasper_qwen_img",
        ...     save_dir="tem/qasper_qwen_img",
        ...     top_k=10
        ... )
        >>> print(f"Processed: {stats['total']}, Success: {stats['success']}")
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and processor
    print(f"Loading model: {model_path}")
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

    # Find all question folders
    folders = []
    for root, dirs, files in os.walk(data_path):
        if "result" in root:
            continue
        if "input_tokens.json" in files and "context.txt" in files:
            folders.append(root)

    print(f"Found {len(folders)} question folders")

    # Process each folder
    stats = {"total": len(folders), "success": 0, "failed": 0, "errors": []}

    for folder_path in tqdm(folders, desc="Processing questions"):
        try:
            folder_name = os.path.basename(folder_path)

            # Get file paths
            input_tokens_path = os.path.join(folder_path, "input_tokens.json")
            context_path = os.path.join(folder_path, "context.txt")
            output_path = os.path.join(folder_path, "extracted_evidence_qwen_embedding.txt")

            # Extract question
            question = _extract_question_from_tokens(input_tokens_path, processor.tokenizer)
            if question is None:
                stats["failed"] += 1
                stats["errors"].append((folder_name, "Failed to extract question"))
                continue

            # Clean question text
            prefix_marker = "Please answer the question based on the document images provided."
            if question.startswith(prefix_marker):
                question = question[len(prefix_marker):].strip()
            cutoff_marker = "Please output your answer **directly**"
            if cutoff_marker in question:
                question = question.split(cutoff_marker)[0].strip()

            # Read context
            with open(context_path, 'r', encoding='utf-8') as f:
                context = f.read().strip()

            if not context:
                stats["failed"] += 1
                stats["errors"].append((folder_name, "Empty context.txt"))
                continue

            # Split context into sentences
            sentences = _split_into_sentences(context)

            if len(sentences) == 0:
                stats["failed"] += 1
                stats["errors"].append((folder_name, "No valid sentences"))
                continue

            # Get embeddings
            question_embedding = _get_text_embedding(model, processor, question, device)
            sentence_embeddings = _get_batch_text_embeddings(model, processor, sentences, device)

            # Calculate similarity
            question_embedding = question_embedding.float()
            sentence_embeddings = sentence_embeddings.float()

            question_norm = F.normalize(question_embedding, p=2, dim=-1)
            sentence_norm = F.normalize(sentence_embeddings, p=2, dim=-1)

            similarity = torch.matmul(sentence_norm, question_norm.squeeze(0))
            similarity = similarity.cpu().numpy()

            # Get Top K
            k = min(top_k, len(sentences))
            top_k_indices = np.argsort(similarity)[-k:][::-1]

            top_sentences = [sentences[idx] for idx in top_k_indices]
            extracted_text = "\n".join(top_sentences)

            # Save results
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(extracted_text)

            stats["success"] += 1

        except Exception as e:
            stats["failed"] += 1
            stats["errors"].append((folder_name, str(e)))

    return stats


def _extract_question_from_tokens(input_tokens_path: str, tokenizer) -> Optional[str]:
    """Extract question text from input_tokens.json"""
    if not os.path.exists(input_tokens_path):
        return None

    try:
        with open(input_tokens_path, 'r', encoding='utf-8') as f:
            tokens = json.load(f)

        vision_end_idx = None
        for i, token in enumerate(tokens):
            if token == '<|vision_end|>':
                vision_end_idx = i
                break

        if vision_end_idx is None:
            return None

        question_tokens = []
        for i in range(vision_end_idx + 1, len(tokens)):
            token = tokens[i]
            if token == '<|im_end|>':
                break
            question_tokens.append(token)

        if tokenizer is not None:
            token_ids = tokenizer.convert_tokens_to_ids(question_tokens)
            question_text = tokenizer.decode(token_ids, skip_special_tokens=True)
            return question_text.strip()

        return None

    except Exception:
        return None


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    return sentences


def _get_text_embedding(model, processor, text: str, device: str) -> torch.Tensor:
    """Get text embedding"""
    tokenizer = processor.tokenizer
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    with torch.no_grad():
        embed_layer = model.get_input_embeddings()
        text_embeds = embed_layer(input_ids)

        mask_expanded = attention_mask.unsqueeze(-1).expand(text_embeds.size()).float()
        sum_embeddings = torch.sum(text_embeds * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        text_embedding = sum_embeddings / sum_mask

    return text_embedding


def _get_batch_text_embeddings(model, processor, texts: List[str], device: str, batch_size: int = 32) -> torch.Tensor:
    """Get batch text embeddings"""
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
            text_embeds = embed_layer(input_ids)

            mask_expanded = attention_mask.unsqueeze(-1).expand(text_embeds.size()).float()
            sum_embeddings = torch.sum(text_embeds * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            batch_embeddings = sum_embeddings / sum_mask

        all_embeddings.append(batch_embeddings)

    return torch.cat(all_embeddings, dim=0)
