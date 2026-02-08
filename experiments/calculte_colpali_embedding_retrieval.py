"""
ColPali Embedding 检索脚本
使用 ColPali 模型进行 embedding 检索
检索出与 question 相关度排名前 K 的 patch，提取对应文字

主要流程：
1. 将长图切分成若干正方形图片（不足部分用白色填充）
2. 对每个正方形图片计算 patch embeddings
3. 计算 question embedding 与所有 patch 的相似度
4. 选择得分最高的 K 个 patch
5. 从 word_mapping.json 中提取对应文本
6. 保存到 extracted_evidence_colpali.txt
"""

import os
import sys
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.utils.torch_utils import get_torch_device
from colpali_engine.interpretability import get_similarity_maps_from_embeddings

# ================= 配置区域 =================
# 路径设置
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ROOT_DIR = "tem/qasper_qwen_img"
# word_mapping.json 所在的目录（用于证据提取）
WORD_MAPPING_ROOT_DIR = "tem/qasper_qwen_img"

# 加载模型配置
MODEL_CONFIG_PATH = os.path.join(ROOT, "config/model_config.json")
if os.path.exists(MODEL_CONFIG_PATH):
    with open(MODEL_CONFIG_PATH, 'r', encoding='utf-8') as f:
        model_config = json.load(f)
    MODEL_NAME = model_config.get("model_path", {}).get("colpali")
    QWEN_MODEL_PATH = model_config.get("model_path", {}).get("qwen")
    if not MODEL_NAME:
        raise ValueError("model_path.colpali not found in config/model_config.json")
    if not QWEN_MODEL_PATH:
        raise ValueError("model_path.qwen not found in config/model_config.json")
else:
    raise FileNotFoundError(f"Model config file not found: {MODEL_CONFIG_PATH}")

OUTPUT_FOLDER_NAME = "result"  # 跳过此文件夹

# Top K patches 数量
TOP_K_PATCHES = 20

# 切分图片时的重叠像素数
OVERLAP = 0

# 临时文件夹名称（存放切分后的图片和JSON）
TEMP_FOLDER_NAME = "colpali_temp"
# ===========================================


def split_image_into_squares(image: Image.Image, overlap: int = 0) -> list:
    """
    将图片按宽度切分成若干个正方形图片。
    最后一块如果不够正方形大小，用白色填充。
    
    Args:
        image: PIL Image 对象
        overlap: 相邻块之间的重叠像素数（可选）
    
    Returns:
        list of tuples: [(cropped_image, start_x, start_y, actual_height), ...]
        每个元素包含切分后的图片、在原图中的起始位置、以及该块在原图中的实际高度
    """
    width, height = image.size
    square_size = width  # 使用宽度作为正方形边长
    
    crops = []
    
    if height <= width:
        # 图片是横向或正方形，不需要切分
        # 如果高度小于宽度，用白色填充到正方形
        if height < width:
            new_image = Image.new(image.mode, (width, width), (255, 255, 255))
            new_image.paste(image, (0, 0))
            crops.append((new_image, 0, 0, height))
        else:
            crops.append((image.copy(), 0, 0, height))
    else:
        # 图片是纵向的，需要切分
        step = square_size - overlap  # 每次移动的步长
        num_crops = math.ceil(height / step)
        
        for i in range(num_crops):
            start_y = i * step
            end_y = min(start_y + square_size, height)
            actual_height = end_y - start_y  # 该块在原图中的实际高度
            
            # 切出这一块
            crop = image.crop((0, start_y, width, end_y))
            
            # 如果最后一块不足正方形大小，用白色填充
            if actual_height < square_size:
                # 创建白色背景的正方形图片
                padded_crop = Image.new(image.mode, (square_size, square_size), (255, 255, 255))
                # 将切出的内容粘贴到左上角
                padded_crop.paste(crop, (0, 0))
                crops.append((padded_crop, 0, start_y, actual_height))
            else:
                crops.append((crop, 0, start_y, actual_height))
    
    return crops


def load_model_and_processor(model_name: str, device):
    """加载 ColPali 模型和处理器"""
    print(f"正在加载模型: {model_name}")
    
    model = ColPali.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ).eval()
    
    processor = ColPaliProcessor.from_pretrained(model_name)
    
    print(f"模型加载完成，设备: {device}")
    print(f"Patch 大小: {model.patch_size}")
    return model, processor


def get_image_patch_embeddings(model, processor, image: Image.Image, device):
    """
    获取图像的 patch embeddings
    
    Returns:
        image_embeddings: 图像的 patch embeddings
        n_patches: (n_patches_x, n_patches_y)
    """
    batch_images = processor.process_images([image]).to(device)
    
    with torch.no_grad():
        image_embeddings = model.forward(**batch_images)
    
    n_patches = processor.get_n_patches(image_size=image.size, patch_size=model.patch_size)
    
    return image_embeddings, n_patches, batch_images


def get_query_embedding(model, processor, query: str, device):
    """
    获取查询文本的 embedding
    """
    batch_queries = processor.process_queries([query]).to(device)
    
    with torch.no_grad():
        query_embeddings = model.forward(**batch_queries)
    
    return query_embeddings


def compute_patch_scores(
    image_embeddings, 
    query_embeddings, 
    n_patches, 
    image_mask,
    crop_start_y: int,
    crop_height: int,
    patch_pixel_width: float,
    patch_pixel_height: float
) -> list:
    """
    计算每个 patch 的相似度得分，并返回 patch 信息列表
    
    Args:
        image_embeddings: 图像 embeddings
        query_embeddings: 查询 embeddings
        n_patches: (n_patches_x, n_patches_y)
        image_mask: 图像 mask
        crop_start_y: 该 crop 在原图中的起始 Y 坐标
        crop_height: crop 的高度
        patch_pixel_width: 每个 patch 对应的像素宽度
        patch_pixel_height: 每个 patch 对应的像素高度
    
    Returns:
        list of patch info dicts
    """
    # 获取 similarity maps
    batched_similarity_maps = get_similarity_maps_from_embeddings(
        image_embeddings=image_embeddings,
        query_embeddings=query_embeddings,
        n_patches=n_patches,
        image_mask=image_mask,
    )
    
    similarity_maps = batched_similarity_maps[0]  # shape: (num_tokens, n_patches_y, n_patches_x)
    
    n_patches_x, n_patches_y = n_patches
    num_tokens = similarity_maps.shape[0]
    
    patches = []
    
    for patch_y in range(n_patches_y):
        for patch_x in range(n_patches_x):
            # 计算 patch 在 crop 中的像素坐标
            local_x_start = int(patch_x * patch_pixel_width)
            local_x_end = int((patch_x + 1) * patch_pixel_width)
            local_y_start = int(patch_y * patch_pixel_height)
            local_y_end = int((patch_y + 1) * patch_pixel_height)
            
            # 计算 patch 在原图中的像素坐标
            global_x_start = local_x_start
            global_x_end = local_x_end
            global_y_start = crop_start_y + local_y_start
            global_y_end = crop_start_y + local_y_end
            
            # 获取每个 token 的得分
            token_scores = {}
            for token_idx in range(num_tokens):
                score = similarity_maps[token_idx, patch_y, patch_x].item()
                token_scores[token_idx] = score
            
            # 计算所有 token 的平均得分和最大得分
            avg_score = sum(token_scores.values()) / len(token_scores)
            max_score = max(token_scores.values())
            
            patch_info = {
                'patch_coords': {
                    'x_start': global_x_start,
                    'y_start': global_y_start,
                    'x_end': global_x_end,
                    'y_end': global_y_end,
                },
                'scores': {
                    'avg_score': avg_score,
                    'max_score': max_score,
                }
            }
            patches.append(patch_info)
    
    return patches


def get_top_k_patch_bounds(patches: list, k: int = 10) -> list:
    """
    获取得分最高的 K 个 patch 的像素边界
    
    Args:
        patches: patch 信息列表
        k: Top K 数量
    
    Returns:
        List of (x1, y1, x2, y2) 像素边界列表
    """
    # 按 max_score 降序排序
    sorted_patches = sorted(patches, key=lambda p: p['scores']['max_score'], reverse=True)
    
    # 去重（基于坐标）
    seen_coords = set()
    unique_patches = []
    for patch in sorted_patches:
        coords = patch['patch_coords']
        key = (coords['x_start'], coords['y_start'], coords['x_end'], coords['y_end'])
        if key not in seen_coords:
            seen_coords.add(key)
            unique_patches.append(patch)
    
    # 取 Top K
    top_k_patches = unique_patches[:k]
    
    bounds_list = []
    for patch in top_k_patches:
        coords = patch['patch_coords']
        bounds_list.append((
            coords['x_start'],
            coords['y_start'],
            coords['x_end'],
            coords['y_end']
        ))
    
    return bounds_list


def extract_evidence_from_patches(patch_bounds, word_mapping_path, output_path):
    """
    根据 patch 像素位置，从 word_mapping.json 中提取有交集的行文本
    
    Args:
        patch_bounds: List of (x1, y1, x2, y2) patch 像素边界列表
        word_mapping_path: word_mapping.json 文件路径
        output_path: 输出文件路径
    
    Returns:
        提取的文本内容
    """
    if not os.path.exists(word_mapping_path):
        return ""
    
    try:
        with open(word_mapping_path, 'r', encoding='utf-8') as f:
            word_data = json.load(f)
    except Exception as e:
        print(f"Error loading word_mapping.json: {e}")
        return ""
    
    # 收集与 patch 有交集的行号
    involved_lines = set()
    
    for (px1, py1, px2, py2) in patch_bounds:
        for word_info in word_data.get("words", []):
            word_bbox = word_info.get("bbox", [])
            if len(word_bbox) < 4:
                continue
            
            wx1, wy1, wx2, wy2 = word_bbox
            
            # AABB 相交检测：两个矩形是否有重叠
            if not (px2 < wx1 or wx2 < px1 or py2 < wy1 or wy2 < py1):
                involved_lines.add(word_info.get("line", -1))
    
    # 按行号收集文本
    line_texts = {}
    for word_info in word_data.get("words", []):
        line_num = word_info.get("line", -1)
        if line_num in involved_lines:
            if line_num not in line_texts:
                line_texts[line_num] = word_info.get("word", "")
    
    # 按行号排序并合并
    sorted_lines = sorted(line_texts.keys())
    extracted_lines = [line_texts[ln] for ln in sorted_lines]
    extracted_text = "\n".join(extracted_lines)
    
    # 保存到文件
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(extracted_text)
    except Exception as e:
        print(f"Error saving extracted evidence: {e}")
    
    return extracted_text


def find_leaf_folders(root_dir):
    """查找所有包含 merged.png 的叶子文件夹"""
    leaf_folders = []
    for root, dirs, files in os.walk(root_dir):
        if OUTPUT_FOLDER_NAME in root:
            continue
        if TEMP_FOLDER_NAME in root:
            continue
        # 只检查 merged.png，word_mapping.json 在另一个目录
        if "merged.png" in files:
            leaf_folders.append(root)
    return leaf_folders


def find_word_mapping_path(folder_path):
    """
    从 WORD_MAPPING_ROOT_DIR 中查找对应的 word_mapping.json
    
    匹配逻辑：
    - folder_path: ROOT_DIR/<paper_id>/<question_hash>/rendered_images/<image_hash>/
    - 在 WORD_MAPPING_ROOT_DIR/<paper_id>/<question_hash>/rendered_images/*/word_mapping.json 中查找
    
    Args:
        folder_path: 包含 merged.png 的文件夹路径 (在 ROOT_DIR 中)
    
    Returns:
        word_mapping.json 的完整路径，如果不存在返回 None
    """
    # 首先尝试本地目录
    local_path = os.path.join(folder_path, "word_mapping.json")
    if os.path.exists(local_path):
        return local_path
    
    # 从 folder_path 提取相对路径信息
    # folder_path: .../qasper_qwen_img/<paper_id>/<question_hash>/rendered_images/<image_hash>/
    try:
        # 获取 rendered_images 的父目录 (question_hash 所在目录)
        rendered_images_dir = os.path.dirname(folder_path.rstrip('/'))  # .../rendered_images
        question_folder = os.path.dirname(rendered_images_dir)  # .../<question_hash>
        paper_folder = os.path.dirname(question_folder)  # .../<paper_id>
        
        question_hash = os.path.basename(question_folder)
        paper_id = os.path.basename(paper_folder)
        
        # 在 WORD_MAPPING_ROOT_DIR 中查找对应的 word_mapping.json
        # 路径: WORD_MAPPING_ROOT_DIR/<paper_id>/<question_hash>/rendered_images/*/word_mapping.json
        target_rendered_images = os.path.join(WORD_MAPPING_ROOT_DIR, paper_id, question_hash, "rendered_images")
        
        if os.path.exists(target_rendered_images):
            # 遍历子目录查找 word_mapping.json
            for subdir in os.listdir(target_rendered_images):
                word_mapping_path = os.path.join(target_rendered_images, subdir, "word_mapping.json")
                if os.path.exists(word_mapping_path):
                    return word_mapping_path
    except Exception as e:
        print(f"Error finding word_mapping.json: {e}")
    
    return None


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
        
        # 如果有 tokenizer，使用标准方法解码
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


def process_single_question(folder_path, model, processor, device, qwen_tokenizer=None):
    """
    处理单个问题文件夹
    
    Args:
        folder_path: rendered_images 下的叶子文件夹路径
        model: ColPali 模型
        processor: ColPali 处理器
        device: 设备
        qwen_tokenizer: Qwen tokenizer（用于解码 input_tokens.json）
    
    Returns:
        (folder_name, success_flag, message)
    """
    try:
        folder_name = os.path.basename(folder_path)
        
        # 定位 question_folder（包含 input_tokens.json 的父目录）
        question_folder = os.path.dirname(os.path.dirname(folder_path))
        
        # 获取路径
        merged_path = os.path.join(folder_path, "merged.png")
        # 使用 find_word_mapping_path 从另一个目录查找 word_mapping.json
        word_mapping_path = find_word_mapping_path(folder_path)
        input_tokens_path = os.path.join(question_folder, "input_tokens.json")
        output_path = os.path.join(question_folder, "extracted_evidence_colpali.txt")
        
        # 创建临时文件夹
        temp_folder = os.path.join(question_folder, TEMP_FOLDER_NAME)
        os.makedirs(temp_folder, exist_ok=True)
        
        # 检查文件是否存在
        if not os.path.exists(merged_path):
            return (folder_name, False, "Missing merged.png")
        
        if word_mapping_path is None:
            return (folder_name, False, "word_mapping.json not found in WORD_MAPPING_ROOT_DIR")
        
        # 提取问题文本
        question = extract_question_from_tokens(input_tokens_path, tokenizer=qwen_tokenizer)
        if question is None:
            return (folder_name, False, "Failed to extract question")
        
        # 处理问题文本（去掉前后指令部分）
        display_question = question
        prefix_marker = "Please answer the question based on the document images provided."
        if display_question.startswith(prefix_marker):
            display_question = display_question[len(prefix_marker):].strip()
        cutoff_marker = "Please output your answer **directly**"
        if cutoff_marker in display_question:
            display_question = display_question.split(cutoff_marker)[0].strip()
        
        print(f"\n[{folder_name}] Question: {display_question[:100]}...")
        question = display_question
        
        # 加载图像
        image = Image.open(merged_path).convert("RGB")
        img_w, img_h = image.size
        
        # 切分图片
        crops = split_image_into_squares(image, overlap=OVERLAP)
        
        # 保存切分后的图片到临时文件夹
        for i, (crop_img, start_x, start_y, actual_height) in enumerate(crops):
            crop_save_path = os.path.join(temp_folder, f"crop_{i}_y{start_y}.png")
            crop_img.save(crop_save_path)
        
        # 获取 query embedding
        query_embeddings = get_query_embedding(model, processor, question, device)
        
        # 存储所有 patch 的信息
        all_patches = []
        
        # 处理每个 crop
        for crop_idx, (crop_img, start_x, start_y, actual_height) in enumerate(crops):
            # 获取图像 patch embeddings
            image_embeddings, n_patches, batch_images = get_image_patch_embeddings(
                model, processor, crop_img, device
            )
            
            # 获取 image mask
            image_mask = processor.get_image_mask(batch_images)
            
            # 计算 patch 像素尺寸
            crop_width, crop_height = crop_img.size
            n_patches_x, n_patches_y = n_patches
            patch_pixel_width = crop_width / n_patches_x
            patch_pixel_height = crop_height / n_patches_y
            
            # 计算 patch 得分
            patches = compute_patch_scores(
                image_embeddings=image_embeddings,
                query_embeddings=query_embeddings,
                n_patches=n_patches,
                image_mask=image_mask,
                crop_start_y=start_y,
                crop_height=crop_height,
                patch_pixel_width=patch_pixel_width,
                patch_pixel_height=patch_pixel_height
            )
            
            all_patches.extend(patches)
        
        # 保存 patch 数据到 JSON（临时文件夹）
        patch_data = {
            'metadata': {
                'original_image_size': (img_w, img_h),
                'num_crops': len(crops),
                'overlap': OVERLAP,
            },
            'patches': all_patches
        }
        json_path = os.path.join(temp_folder, "patch_scores.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(patch_data, f, ensure_ascii=False, indent=2)
        
        # 获取 Top K patch 的像素边界
        patch_bounds = get_top_k_patch_bounds(all_patches, k=TOP_K_PATCHES)
        
        # 提取证据文本
        extracted_text = extract_evidence_from_patches(
            patch_bounds, word_mapping_path, output_path
        )
        
        if extracted_text:
            line_count = len(extracted_text.strip().split('\n')) if extracted_text.strip() else 0
            return (folder_name, True, f"Extracted {line_count} lines")
        else:
            return (folder_name, False, "No text extracted")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return (os.path.basename(folder_path), False, str(e))


def main():
    """主函数"""
    print("=" * 80)
    print("ColPali Embedding 检索脚本")
    print("=" * 80)
    
    # 查找所有叶子文件夹
    folders = find_leaf_folders(ROOT_DIR)
    print(f"\n找到 {len(folders)} 个待处理的文件夹")
    
    if len(folders) == 0:
        print("未找到任何待处理的文件夹")
        return
    
    # 加载 ColPali 模型
    device = get_torch_device("auto")
    model, processor = load_model_and_processor(MODEL_NAME, device)
    
    # 尝试加载 Qwen tokenizer（用于解码 input_tokens.json）
    qwen_tokenizer = None
    try:
        from transformers import AutoTokenizer
        if os.path.exists(QWEN_MODEL_PATH):
            qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_PATH, trust_remote_code=True)
            print(f"已加载 Qwen tokenizer 用于解码问题文本")
    except Exception as e:
        print(f"无法加载 Qwen tokenizer，将使用备用解码方法: {e}")
    
    # 统计
    success_count = 0
    fail_count = 0
    
    # 处理每个文件夹
    print(f"\n开始处理...")
    for folder_path in tqdm(folders, desc="Processing"):
        folder_name, success, message = process_single_question(
            folder_path, model, processor, device, qwen_tokenizer
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
    print(f"证据文本已保存到各问题目录下的 extracted_evidence_colpali.txt")
    print(f"临时文件（切分图片、patch 得分 JSON）保存在各问题目录下的 {TEMP_FOLDER_NAME}/ 文件夹中")


if __name__ == "__main__":
    main()

