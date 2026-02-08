import os
import json
import glob
import cv2
import numpy as np
import matplotlib
# 设置后端为 'Agg' 以支持多进程绘图
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import math
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm
from functools import partial

# ================= 配置区域 =================
ROOT_DIR = "tem/hotpot_qwen_img"
# word_mapping.json 所在的目录（用于证据提取）
WORD_MAPPING_ROOT_DIR = "tem/hotpot_qwen_img"
OUTPUT_FOLDER_NAME = "result"

# === 颜色设置 ===
RED_LOWER = np.array([0, 0, 150])
RED_UPPER = np.array([100, 100, 255])

# === 膨胀参数 ===
DILATION_KERNEL_SIZE = (12, 3)
DILATION_ITERATIONS = 1

# === Debug 边框设置 ===
DEBUG_BOX_COLOR = (255, 0, 0)
DEBUG_BOX_THICKNESS = 2

# === 可视化参数 ===
DEBUG_HEATMAP_ALPHA = 0.6 
TOP_K_PATCHES = 10
TOP_K_COLOR = (0, 0, 255)

# === 开关 ===
SAVE_DEBUG_MASK = True 
SAVE_PLOTS = True
# ===========================================

OUTPUT_DIR = os.path.join(ROOT_DIR, OUTPUT_FOLDER_NAME)

def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def find_leaf_folders(root_dir):
    leaf_folders = []
    for root, dirs, files in os.walk(root_dir):
        if OUTPUT_FOLDER_NAME in root:
            continue
        if "merged_evidence.png" in files:
            leaf_folders.append(root)
    return leaf_folders

def calculate_patch_distribution(img_path, total_visual_tokens):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    h_img, w_img = img.shape[:2]
    aspect_ratio = h_img / w_img 
    num_patches = total_visual_tokens

    factors = []
    for i in range(1, int(math.sqrt(num_patches)) + 1):
        if num_patches % i == 0:
            factors.append((i, num_patches // i))
            if i != num_patches // i:
                factors.append((num_patches // i, i))

    best_fit = None
    best_ratio_diff = float('inf')
    for w, h in factors:
        current_ratio = h / w
        ratio_diff = abs(current_ratio - aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_fit = (w, h)

    if best_fit is None:
        grid_size = int(math.sqrt(num_patches))
        best_fit = (grid_size, grid_size)
    return best_fit

def get_precise_bbox_mask_and_debug_img(image_path, grid_w, grid_h, folder_name, save_debug=False):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    h_img, w_img = img.shape[:2]
    mask_color = cv2.inRange(img, RED_LOWER, RED_UPPER)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, DILATION_KERNEL_SIZE)
    mask_dilated = cv2.dilate(mask_color, kernel, iterations=DILATION_ITERATIONS)
    contours, _ = cv2.findContours(mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask_boxes = np.zeros((h_img, w_img), dtype=np.uint8)
    debug_base_img = img.copy() if save_debug else None

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 4 and h > 4:
            cv2.rectangle(mask_boxes, (x, y), (x + w, y + h), 255, thickness=-1)
            if save_debug:
                cv2.rectangle(debug_base_img, (x, y), (x + w, y + h), DEBUG_BOX_COLOR, thickness=DEBUG_BOX_THICKNESS)

    if save_debug:
        debug_path = os.path.join(OUTPUT_DIR, f"{folder_name}_debug_boxes.png")
        if not os.path.exists(debug_path):
            cv2.imwrite(debug_path, debug_base_img)

    # 创建patch级别的二值mask：如果patch与evidence区域有交集则为1
    patch_mask = np.zeros((grid_h, grid_w), dtype=np.uint8)

    # 计算每个patch在原图上的坐标范围
    patch_h = h_img // grid_h
    patch_w = w_img // grid_w

    for i in range(grid_h):
        for j in range(grid_w):
            # patch在原图上的坐标范围
            y1 = i * patch_h
            y2 = min((i + 1) * patch_h, h_img)
            x1 = j * patch_w
            x2 = min((j + 1) * patch_w, w_img)

            # 检查这个patch区域是否与mask_boxes有交集
            patch_region = mask_boxes[y1:y2, x1:x2]
            if np.any(patch_region > 0):  # 如果有任何像素在evidence区域内
                patch_mask[i, j] = 1

    patch_weights = patch_mask.flatten().astype(np.float32)
    return patch_weights, debug_base_img

def create_attention_overlay(base_img, attn_array, grid_h, grid_w, alpha=0.5):
    h_img, w_img = base_img.shape[:2]
    num_patches = grid_h * grid_w
    if len(attn_array) < num_patches:
        attn_array = np.pad(attn_array, (0, num_patches - len(attn_array)))
    attn_array = attn_array[:num_patches]
    
    try:
        attn_grid = attn_array.reshape((grid_h, grid_w))
    except ValueError:
        return base_img

    attn_norm = cv2.normalize(attn_grid, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    attn_resized = cv2.resize(attn_norm, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
    heatmap = cv2.applyColorMap(attn_resized, cv2.COLORMAP_JET)
    overlay_img = cv2.addWeighted(base_img, 1.0 - alpha, heatmap, alpha, 0)
    return overlay_img

def create_top_k_overlay(base_img, attn_array, grid_h, grid_w, k=10):
    h_img, w_img = base_img.shape[:2]
    num_patches = grid_h * grid_w
    if len(attn_array) < num_patches:
        attn_array = np.pad(attn_array, (0, num_patches - len(attn_array)))
    attn_array = attn_array[:num_patches]

    top_k_indices = np.argsort(attn_array)[-k:]
    mask_grid = np.zeros((grid_h * grid_w), dtype=np.uint8)
    mask_grid[top_k_indices] = 1 
    
    try:
        mask_grid_2d = mask_grid.reshape((grid_h, grid_w))
    except ValueError:
        return base_img

    mask_resized = cv2.resize(mask_grid_2d, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
    color_layer = np.zeros_like(base_img)
    color_layer[:] = TOP_K_COLOR
    output_img = base_img.copy()
    
    alpha = 0.5
    roi = output_img[mask_resized == 1]
    overlay_roi = color_layer[mask_resized == 1]
    if len(roi) > 0:
        blended = cv2.addWeighted(roi, 1.0 - alpha, overlay_roi, alpha, 0)
        output_img[mask_resized == 1] = blended

    return output_img


def get_top_k_patch_pixel_bounds(attn_array, grid_h, grid_w, img_h, img_w, k=10):
    """
    计算 Top K patch 的像素边界
    
    Args:
        attn_array: 注意力数组
        grid_h, grid_w: 网格尺寸
        img_h, img_w: 原图尺寸
        k: Top K 的数量
    
    Returns:
        List of (x1, y1, x2, y2) 像素边界列表
    """
    num_patches = grid_h * grid_w
    if len(attn_array) < num_patches:
        attn_array = np.pad(attn_array, (0, num_patches - len(attn_array)))
    attn_array = attn_array[:num_patches]
    
    top_k_indices = np.argsort(attn_array)[-k:]
    
    patch_width = img_w / grid_w
    patch_height = img_h / grid_h
    
    bounds_list = []
    for patch_idx in top_k_indices:
        row = patch_idx // grid_w
        col = patch_idx % grid_w
        
        x1 = int(col * patch_width)
        y1 = int(row * patch_height)
        x2 = int((col + 1) * patch_width)
        y2 = int((row + 1) * patch_height)
        
        bounds_list.append((x1, y1, x2, y2))
    
    return bounds_list


def extract_evidence_from_patches(patch_bounds, word_mapping_path, output_path):
    """
    根据 patch 像素位置，从 word_mapping.json 中提取有交集的行文本
    
    Args:
        patch_bounds: List of (x1, y1, x2, y2) patch 像素边界列表
        word_mapping_path: word_mapping.json 文件路径
        output_path: 输出文件路径 (extracted_evidence.txt)
    
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
            # 注意：根据 word_mapping.json 结构，每个 "word" 实际上是整行文本
    
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
    # folder_path 类似: ROOT_DIR/<paper_id>/<question_hash>/rendered_images/<image_hash>/
    # 需要提取 paper_id 和 question_hash
    
    # 首先尝试本地目录
    local_path = os.path.join(folder_path, "word_mapping.json")
    if os.path.exists(local_path):
        return local_path
    
    # 从 folder_path 提取相对路径信息
    # folder_path: .../qasper_qwen_img/1601.03313/e6204.../rendered_images/69322.../
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
        pass
    
    return None


def get_visual_token_indices(question_folder):
    input_tokens_path = os.path.join(question_folder, "input_tokens.json")
    if not os.path.exists(input_tokens_path): return None
    try:
        with open(input_tokens_path, 'r') as f: tokens = json.load(f)
        vision_start_idx, vision_end_idx = None, None
        for i, token in enumerate(tokens):
            if token == '<|vision_start|>': vision_start_idx = i
            elif token == '<|vision_end|>': vision_end_idx = i
        if vision_start_idx is not None and vision_end_idx is not None:
            return vision_start_idx + 1, vision_end_idx
    except Exception: pass
    return None

def load_attention_data(question_folder):
    attn_path = os.path.join(question_folder, "attn_first_token.json")
    if not os.path.exists(attn_path): return None
    try:
        with open(attn_path, 'r') as f: return json.load(f)
    except Exception: return None

def calculate_entropy_confidence(prob_dist):
    """
    计算基于归一化熵的置信度。
    Definition: Confidence = 1 - (Shannon_Entropy / Max_Possible_Entropy)
    Result: [0, 1]. 1 means extremely focused (spiky), 0 means uniform (flat).
    """
    probs = prob_dist + 1e-12
    probs = probs / np.sum(probs)
    entropy = -np.sum(probs * np.log2(probs))
    n = len(probs)
    if n <= 1: return 1.0
    max_entropy = np.log2(n)
    confidence = 1.0 - (entropy / max_entropy)
    return max(0.0, min(1.0, confidence))

def calculate_iou(attention_vector, mask_vector):
    """
    计算Attention向量A与Mask向量M的IoU。
    IoU = ∑(A_i · M_i) / (∑A_i + ∑M_i - ∑(A_i · M_i))
    """
    # 确保两个向量长度相同
    min_len = min(len(attention_vector), len(mask_vector))
    A = attention_vector[:min_len]
    M = mask_vector[:min_len]

    # 计算分子：交集
    intersection = np.sum(A * M)

    # 计算分母：并集
    union = np.sum(A) + np.sum(M) - intersection

    # 避免除零
    if union <= 1e-12:
        return 0.0

    iou = intersection / union
    return max(0.0, min(1.0, iou))

def calculate_mse(attention_vector, mask_vector):
    """
    计算Attention向量A与Mask向量M的MSE。
    MSE = (1/n) * Σ(A_i - M_i)^2
    """
    # 确保两个向量长度相同
    min_len = min(len(attention_vector), len(mask_vector))
    A = attention_vector[:min_len]
    M = mask_vector[:min_len]

    # 计算MSE
    mse = np.mean((A - M) ** 2)
    return mse

def process_folder(folder_path, candidate_heads=None, mode='scan'):
    """
    mode='scan': 计算矩阵，返回给 Phase 1 用于统计全局得分最高的 Top 5
    mode='viz': 
        1. 接收 candidate_heads (全局得分最高的 Top 5)
        2. 直接使用这固定的 5 个头进行渲染
        3. 绘图
    """
    try:
        folder_name = os.path.basename(folder_path)
        question_folder = os.path.dirname(os.path.dirname(folder_path))
        evidence_path = os.path.join(folder_path, "merged_evidence.png")
        merged_path = os.path.join(folder_path, "merged.png")

        attn_data = load_attention_data(question_folder)
        if attn_data is None: return None
        visual_indices = get_visual_token_indices(question_folder)
        if visual_indices is None: return None

        visual_start, visual_end = visual_indices
        visual_token_count = visual_end - visual_start
        grid_w, grid_h = calculate_patch_distribution(merged_path, visual_token_count)

        # Viz 模式才需要生成 debug 图
        save_debug = (mode == 'viz' and SAVE_DEBUG_MASK)
        patch_weights, debug_base_img = get_precise_bbox_mask_and_debug_img(
            evidence_path, grid_w, grid_h, folder_name, save_debug=save_debug
        )

        num_layers = len(attn_data)
        num_heads = len(attn_data[0])
        score_matrix = np.zeros((num_layers, num_heads))
        confidence_matrix = np.zeros((num_layers, num_heads)) 

        # === 1. Calculate Score & Confidence Matrix ===
        for layer_idx in range(num_layers):
            heads_data = attn_data[layer_idx]
            for head_idx, attn_list in enumerate(heads_data):
                full_attn_array = np.array(attn_list[0])
                visual_attn = full_attn_array[visual_start:visual_end]

                if len(visual_attn) != len(patch_weights):
                    min_len = min(len(visual_attn), len(patch_weights))
                    attn_array = visual_attn[:min_len]
                    current_weights = patch_weights[:min_len]
                else:
                    attn_array = visual_attn
                    current_weights = patch_weights
                
                weighted_score = np.sum(attn_array * current_weights)
                total_attn = np.sum(attn_array) + 1e-9
                score_matrix[layer_idx, head_idx] = weighted_score / total_attn
                confidence_matrix[layer_idx, head_idx] = calculate_entropy_confidence(attn_array)

        # Scan Mode: 直接返回
        if mode == 'scan':
            return (folder_name, score_matrix, confidence_matrix)

        # === 2. 'viz' Mode: 直接使用全局得分最高的 Top 5 头进行渲染 ===
        if mode == 'viz' and debug_base_img is not None and candidate_heads is not None:
            
            # 直接使用传入的全局 Top 5 头（已按全局平均得分排序）
            # candidate_heads is List[(layer, head)]
            best_5_heads = [(l, h) for (l, h) in candidate_heads if l < num_layers and h < num_heads]
            
            # 绘图
            avg_attn_vector = np.zeros(visual_token_count)
            valid_heads_count = 0

            for layer_idx, head_idx in best_5_heads:
                full_target_attn = np.array(attn_data[layer_idx][head_idx][0])
                visual_attn_part = full_target_attn[visual_start:visual_end]
                
                if len(visual_attn_part) < visual_token_count:
                    visual_attn_part = np.pad(visual_attn_part, (0, visual_token_count - len(visual_attn_part)))
                elif len(visual_attn_part) > visual_token_count:
                    visual_attn_part = visual_attn_part[:visual_token_count]
                
                avg_attn_vector += visual_attn_part
                valid_heads_count += 1
            
            if valid_heads_count > 0:
                avg_attn_vector /= valid_heads_count

                final_debug_img = create_attention_overlay(
                    debug_base_img, avg_attn_vector, grid_h, grid_w, alpha=DEBUG_HEATMAP_ALPHA
                )
                debug_path = os.path.join(OUTPUT_DIR, f"{folder_name}_GlobalTop5_heatmap.png")
                cv2.imwrite(debug_path, final_debug_img)

                top10_debug_img = create_top_k_overlay(
                    debug_base_img, avg_attn_vector, grid_h, grid_w, k=TOP_K_PATCHES
                )
                top10_path = os.path.join(OUTPUT_DIR, f"{folder_name}_GlobalTop5_Top{TOP_K_PATCHES}Patch.png")
                cv2.imwrite(top10_path, top10_debug_img)
                
                # === 新增：提取证据文本 ===
                # 获取图片尺寸
                img_for_bounds = cv2.imread(merged_path)
                if img_for_bounds is not None:
                    h_img, w_img = img_for_bounds.shape[:2]
                    
                    # 计算 Top K patch 的像素边界
                    patch_bounds = get_top_k_patch_pixel_bounds(
                        avg_attn_vector, grid_h, grid_w, h_img, w_img, k=TOP_K_PATCHES
                    )
                    
                    # 查找 word_mapping.json 路径
                    word_mapping_path = find_word_mapping_path(folder_path)
                    
                    if word_mapping_path:
                        # 输出路径：与 gold_evidence.json 同目录
                        # question_folder 是 gold_evidence.json 所在目录
                        output_evidence_path = os.path.join(question_folder, "extracted_evidence.txt")
                        
                        extracted_text = extract_evidence_from_patches(
                            patch_bounds, word_mapping_path, output_evidence_path
                        )
                        
                        if extracted_text:
                            # 可选：打印提取的行数
                            line_count = len(extracted_text.strip().split('\n')) if extracted_text.strip() else 0
                            # print(f"  Extracted {line_count} lines to {output_evidence_path}")

        return (folder_name, score_matrix, confidence_matrix)

    except Exception as e:
        print(f"Error processing {folder_path}: {e}")
        return None

def main():
    ensure_output_dir()
    folders = find_leaf_folders(ROOT_DIR)
    print(f"Found {len(folders)} leaf folders.")
    
    num_workers = 100 
    
    # ================= Phase 1: Scan (Determine Global Top 5 by Average Score) =================
    print(f"\n=== Phase 1: Scanning to determine Global Top 5 Heads by Average Score (Workers: {num_workers}) ===")
    
    sum_matrix = None
    count_valid_samples = 0

    scan_func = partial(process_folder, mode='scan')

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(scan_func, folders), total=len(folders), desc="Scanning"))

    for res in results:
        if res is None: continue
        
        folder_name, matrix, conf_matrix = res
        
        # 累加全局得分矩阵
        if sum_matrix is None:
            sum_matrix = matrix
        else:
            if sum_matrix.shape == matrix.shape:
                sum_matrix += matrix
            else:
                continue
        count_valid_samples += 1

    # === 计算全局平均得分矩阵，提取 Top 5 ===
    if count_valid_samples > 0 and sum_matrix is not None:
        avg_matrix_phase1 = sum_matrix / count_valid_samples
        flattened_avg_scores = avg_matrix_phase1.flatten()
        sorted_indices = np.argsort(flattened_avg_scores)[::-1]  # 降序
        top_5_indices = sorted_indices[:5]
        top_5_coords = [np.unravel_index(idx, avg_matrix_phase1.shape) for idx in top_5_indices]
        top_5_scores = [flattened_avg_scores[idx] for idx in top_5_indices]
        
        global_top_5_heads = top_5_coords  # List[(layer, head)]
    else:
        global_top_5_heads = []
        top_5_scores = []

    # ================= 计算Top Heads的平均Attention分布IoU =================
    print("\n" + "="*80)
    print("CALCULATING TOP HEADS AVERAGE ATTENTION DISTRIBUTION IOU")
    print("="*80)

    # 使用 Phase 1 已计算的全局平均矩阵
    avg_matrix = sum_matrix / count_valid_samples if count_valid_samples > 0 and sum_matrix is not None else None

    # 基于全局平均矩阵找到Top5, Top20, Top50 heads的坐标
    top_5_heads = []
    top_20_heads = []
    top_50_heads = []
    if avg_matrix is not None:
        flattened_scores = avg_matrix.flatten()
        sorted_indices = np.argsort(flattened_scores)[::-1]
        top_5_coords = [np.unravel_index(i, avg_matrix.shape) for i in sorted_indices[:5]]
        top_20_coords = [np.unravel_index(i, avg_matrix.shape) for i in sorted_indices[:20]]
        top_50_coords = [np.unravel_index(i, avg_matrix.shape) for i in sorted_indices[:50]]
        top_5_heads = top_5_coords
        top_20_heads = top_20_coords
        top_50_heads = top_50_coords

    print(f"Top 5 Heads: {len(top_5_heads)}")
    print(f"Top 20 Heads: {len(top_20_heads)}")
    print(f"Top 50 Heads: {len(top_50_heads)}")

    # 只计算前5个样本的IoU
    sample_count = min(10, len(folders))
    print(f"Processing first {sample_count} samples...")

    print(f"{'Sample':<15} | {'Top5 IoU':<10} | {'Top5 MSE':<10} | {'Top20 IoU':<10} | {'Top20 MSE':<10} | {'Top50 IoU':<10} | {'Top50 MSE':<10}")
    print("-" * 100)

    for i, folder_path in enumerate(folders[:sample_count]):
        try:
            folder_name = os.path.basename(folder_path)
            question_folder = os.path.dirname(os.path.dirname(folder_path))
            evidence_path = os.path.join(folder_path, "merged_evidence.png")
            merged_path = os.path.join(folder_path, "merged.png")

            attn_data = load_attention_data(question_folder)
            if attn_data is None:
                print(f"{folder_name:<15} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10}")
                continue
            visual_indices = get_visual_token_indices(question_folder)
            if visual_indices is None:
                print(f"{folder_name:<15} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10}")
                continue

            visual_start, visual_end = visual_indices
            visual_token_count = visual_end - visual_start
            grid_w, grid_h = calculate_patch_distribution(merged_path, visual_token_count)

            # 获取mask
            patch_weights, _ = get_precise_bbox_mask_and_debug_img(
                evidence_path, grid_w, grid_h, folder_name, save_debug=False
            )

            num_layers = len(attn_data)
            num_heads = len(attn_data[0])

            iou_results = {'top5': 'N/A', 'top20': 'N/A', 'top50': 'N/A'}
            mse_results = {'top5': 'N/A', 'top20': 'N/A', 'top50': 'N/A'}

            # 计算Top5平均attention分布
            avg_attn_top5 = np.zeros(visual_token_count)
            valid_count_5 = 0
            for layer_idx, head_idx in top_5_heads:
                if layer_idx < num_layers and head_idx < num_heads:
                    full_attn = np.array(attn_data[layer_idx][head_idx][0])
                    visual_attn = full_attn[visual_start:visual_end]
                    if len(visual_attn) < visual_token_count:
                        visual_attn = np.pad(visual_attn, (0, visual_token_count - len(visual_attn)))
                    elif len(visual_attn) > visual_token_count:
                        visual_attn = visual_attn[:visual_token_count]
                    avg_attn_top5 += visual_attn
                    valid_count_5 += 1

            if valid_count_5 > 0:
                avg_attn_top5 /= valid_count_5
                iou_5 = calculate_iou(avg_attn_top5, patch_weights)
                mse_5 = calculate_mse(avg_attn_top5, patch_weights)
                iou_results['top5'] = f"{iou_5:.4f}"
                mse_results['top5'] = f"{mse_5:.4f}"

            # 计算Top20平均attention分布
            avg_attn_top20 = np.zeros(visual_token_count)
            valid_count_20 = 0
            for layer_idx, head_idx in top_20_heads:
                if layer_idx < num_layers and head_idx < num_heads:
                    full_attn = np.array(attn_data[layer_idx][head_idx][0])
                    visual_attn = full_attn[visual_start:visual_end]
                    if len(visual_attn) < visual_token_count:
                        visual_attn = np.pad(visual_attn, (0, visual_token_count - len(visual_attn)))
                    elif len(visual_attn) > visual_token_count:
                        visual_attn = visual_attn[:visual_token_count]
                    avg_attn_top20 += visual_attn
                    valid_count_20 += 1

            if valid_count_20 > 0:
                avg_attn_top20 /= valid_count_20
                iou_20 = calculate_iou(avg_attn_top20, patch_weights)
                mse_20 = calculate_mse(avg_attn_top20, patch_weights)
                iou_results['top20'] = f"{iou_20:.4f}"
                mse_results['top20'] = f"{mse_20:.4f}"

            # 计算Top50平均attention分布
            avg_attn_top50 = np.zeros(visual_token_count)
            valid_count_50 = 0
            for layer_idx, head_idx in top_50_heads:
                if layer_idx < num_layers and head_idx < num_heads:
                    full_attn = np.array(attn_data[layer_idx][head_idx][0])
                    visual_attn = full_attn[visual_start:visual_end]
                    if len(visual_attn) < visual_token_count:
                        visual_attn = np.pad(visual_attn, (0, visual_token_count - len(visual_attn)))
                    elif len(visual_attn) > visual_token_count:
                        visual_attn = visual_attn[:visual_token_count]
                    avg_attn_top50 += visual_attn
                    valid_count_50 += 1

            if valid_count_50 > 0:
                avg_attn_top50 /= valid_count_50
                iou_50 = calculate_iou(avg_attn_top50, patch_weights)
                mse_50 = calculate_mse(avg_attn_top50, patch_weights)
                iou_results['top50'] = f"{iou_50:.4f}"
                mse_results['top50'] = f"{mse_50:.4f}"

            print(f"{folder_name:<15} | {iou_results['top5']:<10} | {mse_results['top5']:<10} | {iou_results['top20']:<10} | {mse_results['top20']:<10} | {iou_results['top50']:<10} | {mse_results['top50']:<10}")

        except Exception as e:
            print(f"{os.path.basename(folder_path):<15} | {'ERROR':<10} | {'ERROR':<10} | {'ERROR':<10} | {'ERROR':<10} | {'ERROR':<10} | {'ERROR':<10}")
            continue

    print("="*80)

    # ================= 结果输出区域 =================
    print("\n" + "="*80)
    print("TABLE 1: GLOBAL TOP 5 HEADS BY AVERAGE ATTENTION SCORE")
    print("Phase 2 will use THESE FIXED 5 heads for all cases.")
    print("="*80)
    print(f"{'Rank':<5} | {'Layer':<10} | {'Head':<10} | {'Avg Score':<20}")
    print("-" * 80)
    for i, ((layer, head), score) in enumerate(zip(global_top_5_heads, top_5_scores), 1):
        print(f"{i:<5} | {layer:<10} | {head:<10} | {score:<20.6f}")
    print("="*80)
    
    print(f"Total Samples Processed: {count_valid_samples}")
    print("="*80)

    # ================= Phase 2: Visualization & Evidence Extraction =================
    print(f"\n=== Phase 2: Generating Visualizations & Extracting Evidence ===")
    print(f"Strategy: Use FIXED Global Top 5 Heads (by average score) for all cases.")
    print(f"Evidence extraction: Extract text lines that overlap with Top {TOP_K_PATCHES} attention patches.")
    
    # 传入 global_top_5_heads 作为固定的 5 个头
    viz_func = partial(process_folder, candidate_heads=global_top_5_heads, mode='viz')

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        _ = list(tqdm(executor.map(viz_func, folders), total=len(folders), desc="Visualizing & Extracting"))

    # ================= Phase 3: Global Heatmap =================
    if count_valid_samples > 0 and sum_matrix is not None:
        avg_matrix = sum_matrix / count_valid_samples
        print(f"\nGenerating Global Average Heatmap...")
        
        # 找到全局平均注意力得分最高的前20名
        flattened_scores = avg_matrix.flatten()
        sorted_indices = np.argsort(flattened_scores)[::-1]  # 降序排列
        top_20_indices = sorted_indices[:20]
        top_20_coords = [np.unravel_index(idx, avg_matrix.shape) for idx in top_20_indices]
        top_20_scores = [flattened_scores[idx] for idx in top_20_indices]
        
        # 计算并输出全局所有头的平均注意力得分（整体平均）
        overall_avg_score = np.mean(avg_matrix)
        print("\n" + "="*80)
        print("GLOBAL AVERAGE ATTENTION SCORE (All Heads Average)")
        print("="*80)
        print(f"Overall Average Score: {overall_avg_score:.6f}")
        print("="*80)
        
        # 输出前20个头的各自的平均注意力得分
        print("\n" + "="*80)
        print("TOP 20 HEADS BY GLOBAL AVERAGE ATTENTION SCORE")
        print("="*80)
        print(f"{'Rank':<5} | {'Layer':<10} | {'Head':<10} | {'Avg Attention Score':<20}")
        print("-" * 80)
        for rank, ((layer, head), score) in enumerate(zip(top_20_coords, top_20_scores), 1):
            print(f"{rank:<5} | {layer:<10} | {head:<10} | {score:<20.6f}")
        print("="*80)
        
        # 生成热图并标记前20名
        plt.figure(figsize=(14, 10))
        sns.heatmap(avg_matrix, cmap="inferno", vmin=0, cbar_kws={'label': 'Avg Attention Score'})
        plt.title(f"GLOBAL Attention Distribution\n(Marked: Top 20 Heads by Global Average Score)")
        plt.xlabel("Head Index")
        plt.ylabel("Layer Index")
        # 标记全局平均注意力得分最高的前20名
        for (layer, head) in top_20_coords:
            plt.gca().add_patch(plt.Rectangle((head, layer), 1, 1, fill=False, edgecolor='cyan', lw=3))
        avg_save_path = os.path.join(ROOT_DIR, "GLOBAL_attention_heatmap.png")
        plt.savefig(avg_save_path, dpi=150)
        plt.close()
        print(f"\nSaved Global Heatmap to {avg_save_path}")

        # 计算并输出前20个头的平均注意力得分的平均分
        top_20_avg = sum(top_20_scores) / len(top_20_scores)
        print(f"\nTop 20 Heads Average Score: {top_20_avg:.6f}")
        print("="*80)

        
        # 生成归一化的注意力矩阵数据 (JSON格式)
        print(f"\nGenerating Normalized Attention Matrix JSON...")
        min_val = np.min(avg_matrix)
        max_val = np.max(avg_matrix)
        if max_val > min_val:
            normalized_matrix = (avg_matrix - min_val) / (max_val - min_val)
        else:
            normalized_matrix = np.zeros_like(avg_matrix)

        # 转换为列表格式以便JSON序列化
        normalized_data = {
            "normalized_attention_matrix": normalized_matrix.tolist(),
            "shape": normalized_matrix.shape,
            "min_original_value": float(min_val),
            "max_original_value": float(max_val),
            "normalization_method": "min_max_scaling",
            "description": "Global average attention matrix normalized to [0,1] range"
        }

        json_save_path = os.path.join(ROOT_DIR, "GLOBAL_attention_matrix_normalized.json")
        with open(json_save_path, 'w', encoding='utf-8') as f:
            json.dump(normalized_data, f, indent=2, ensure_ascii=False)
        print(f"Saved Normalized Attention Matrix to {json_save_path}")
        
                # ================= 输出指定头的注意力分数 =================
        specified_heads = [(29, 24), (20, 13), (16, 28), (11, 14), (7, 3)]
        print("\n" + "="*80)
        print("SPECIFIED HEADS ATTENTION SCORES")
        print("="*80)
        print(f"{'Head':<10} | {'Layer':<10} | {'Avg Attention Score':<20}")
        print("-" * 80)
        valid_scores = []
        for head_idx, layer_idx in specified_heads:
            if layer_idx < avg_matrix.shape[0] and head_idx < avg_matrix.shape[1]:
                score = avg_matrix[layer_idx, head_idx]
                print(f"{head_idx:<10} | {layer_idx:<10} | {score:<20.6f}")
                valid_scores.append(score)
            else:
                print(f"{head_idx:<10} | {layer_idx:<10} | N/A (out of range)")

        # 计算并输出指定头的有效平均注意力得分
        if valid_scores:
            specified_avg = sum(valid_scores) / len(valid_scores)
            print(f"\nSpecified Heads Average Score (valid heads only): {specified_avg:.6f}")
        print("="*80)

        # ============================================================
        # NEW: 分别绘制三张分布图 (Top 1/3 阈值, 稀疏占比, 特别标注指定Head)
        # ============================================================
        print(f"\nGenerating 3 Separate Distribution Plots with Annotations...")
        
        # --- 1. 数据准备 ---
        flattened_scores = avg_matrix.flatten()
        total_heads = len(flattened_scores)
        
        sorted_indices_all = np.argsort(flattened_scores)[::-1] # 降序
        sorted_scores_all = flattened_scores[sorted_indices_all]
        sorted_coords_all = [np.unravel_index(i, avg_matrix.shape) for i in sorted_indices_all]
        
        # --- 2. 计算阈值 (Top 1/3) ---
        min_score = np.min(flattened_scores)
        max_score = np.max(flattened_scores)
        threshold_value = min_score + (max_score - min_score) / 1.5
        print(f"Threshold (Top 1/3 Range): {threshold_value:.6f}")

        # 筛选大于阈值的 Heads
        high_score_heads = []
        for i, score in enumerate(sorted_scores_all):
            if score > threshold_value:
                high_score_heads.append({
                    'rank': i + 1,
                    'score': score,
                    'layer': sorted_coords_all[i][0],
                    'head': sorted_coords_all[i][1]
                })
        
        # --- 3. 计算占比信息 ---
        count_above = len(high_score_heads)
        sparsity_percentage = (count_above / total_heads) * 100
        sparsity_info_text = f"Above Threshold: {count_above}/{total_heads} Heads ({sparsity_percentage:.2f}%)"

        # ============================================================
        # 图 1: 全局分数长尾分布图 (含特定 Head 标注)
        # ============================================================
        plt.figure(figsize=(10, 6))
        
        x_all = range(1, len(sorted_scores_all) + 1)
        plt.plot(x_all, sorted_scores_all, color='#1f77b4', linewidth=2, label='Score Curve')
        plt.fill_between(x_all, sorted_scores_all, color='#1f77b4', alpha=0.1)
        
        # 绘制 Top 1/3 阈值线
        plt.axhline(y=threshold_value, color='green', linestyle='--', linewidth=2, label='Top 1/3 Threshold')
        # 标注 Top 20 分界线
        plt.axvline(x=20, color='red', linestyle=':', alpha=0.8, label='Top 20 Cutoff')
        
        # -------------------------------------------------------
        # [NEW] 特别标注 (24, 29) 和 (21, 11) - 按分数自动调整角度
        # -------------------------------------------------------
        target_markers = [(24, 29), (21, 11)]
        
        # 1. 先把所有目标点的数据找出来
        found_targets = []
        for target_l, target_h in target_markers:
            for idx, (l, h) in enumerate(sorted_coords_all):
                if l == target_l and h == target_h:
                    found_targets.append({
                        'l': target_l,
                        'h': target_h,
                        'rank': idx + 1,
                        'score': sorted_scores_all[idx]
                    })
                    break
        
        # 2. 按分数降序排列
        found_targets.sort(key=lambda x: x['score'], reverse=True)

        # 3. 绘图并根据排序应用样式
        for i, item in enumerate(found_targets):
            rank = item['rank']
            score = item['score']
            l = item['l']
            h = item['h']

            # 绘制高亮星号
            plt.scatter(rank, score, color='orange', s=150, marker='*', zorder=10, edgecolors='black')
            
            # --- 样式控制逻辑 ---
            base_x_offset = total_heads * 0.04
            base_y_offset = (max_score - min_score) * 0.06

            if i == 0:
                # === 最高分的 Head === 陡峭，夹角大
                xy_text_pos = (rank + base_x_offset * 2.0, score + base_y_offset * (-1.0))
                conn_style = "arc3,rad=0.4" 
            else:
                # === 分数较低的 Head === 平缓，夹角小
                xy_text_pos = (rank + base_x_offset * 3.0, score + base_y_offset * (-2.5))
                conn_style = "arc3,rad=0.1"

            # 添加标注
            plt.annotate(f"L{l}-H{h}\n(Rank {rank})", 
                         xy=(rank, score), 
                         xytext=xy_text_pos,
                         arrowprops=dict(facecolor='orange', shrink=0.05, width=1, headwidth=6, 
                                         connectionstyle=conn_style), 
                         fontsize=10, fontweight='bold', color='darkorange')
            
            print(f"Highlighted L{l}-H{h} (Rank {rank}, Score {score:.4f})")

        # --- 补全：保存图1的代码 ---
        plt.gca().text(0.95, 0.85, sparsity_info_text, transform=plt.gca().transAxes, 
                       fontsize=12, fontweight='bold', color='darkgreen', 
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8))

        plt.xlabel("Head Rank (Sorted by Score)", fontsize=14)
        plt.ylabel("Visual Retrieval Score", fontsize=14)
        plt.legend(fontsize=12, loc='center right')
        plt.grid(True, linestyle=':', alpha=0.6)
        
        path_1 = os.path.join(ROOT_DIR, "Dist_1_Global_Curve.png")
        plt.tight_layout()
        plt.savefig(path_1, dpi=300)
        plt.close()
        print(f"Saved Plot 1 to {path_1}")

        # ============================================================
        # 图 2: Top 50 放大特写
        # ============================================================
        plt.figure(figsize=(10, 6))
        
        top_k_zoom = 50
        ranks_zoom = range(1, top_k_zoom + 1)
        scores_zoom = sorted_scores_all[:top_k_zoom]
        
        plt.scatter(ranks_zoom, scores_zoom, color='#d62728', s=60, alpha=0.9, edgecolors='black', zorder=5)
        plt.plot(ranks_zoom, scores_zoom, color='#d62728', alpha=0.3, zorder=1)
        plt.axhline(y=threshold_value, color='green', linestyle='--', linewidth=2, label='Top 1/3 Threshold')

        for item in high_score_heads:
            if item['rank'] > top_k_zoom: continue
            offset_y = (max_score - min_score) * 0.05
            if item['rank'] % 2 == 0:
                text_y = item['score'] + offset_y
                conn_style = "arc3,rad=.2"
            else:
                text_y = item['score'] + offset_y * 1.5
                conn_style = "arc3,rad=-.2"

            plt.annotate(f"L{item['layer']}-H{item['head']}", 
                         xy=(item['rank'], item['score']), 
                         xytext=(item['rank'] + 2, text_y),
                         arrowprops=dict(facecolor='black', arrowstyle='->', connectionstyle=conn_style, alpha=0.6),
                         fontsize=10, fontweight='bold', color='black')

        plt.gca().text(0.95, 0.90, sparsity_info_text, transform=plt.gca().transAxes, 
                       fontsize=12, fontweight='bold', color='darkgreen', 
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8))

        plt.xlabel("Head Rank", fontsize=14)
        plt.ylabel("Visual Retrieval Score", fontsize=14)
        plt.legend(loc='lower right', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        
        path_2 = os.path.join(ROOT_DIR, "Dist_2_Top50_Zoom.png")
        plt.tight_layout()
        plt.savefig(path_2, dpi=300)
        plt.close()
        print(f"Saved Plot 2 to {path_2}")

        # ============================================================
        # 图 3: 层级分布散点图
        # ============================================================
        plt.figure(figsize=(12, 6))
        
        layers = [c[0] for c in sorted_coords_all]
        scores = sorted_scores_all
        
        sc = plt.scatter(layers, scores, c=scores, cmap='viridis', alpha=0.7, s=50, edgecolors='grey')
        cbar = plt.colorbar(sc)
        cbar.set_label("Visual Retrieval Score", fontsize=12)
        plt.axhline(y=threshold_value, color='green', linestyle='--', linewidth=2, label='Top 1/3 Threshold')

        for item in high_score_heads:
            l, h, s = item['layer'], item['head'], item['score']
            plt.scatter([l], [s], s=200, facecolors='none', edgecolors='red', linewidth=2)
            plt.text(l, s, f" H{h}", color='red', fontweight='bold', fontsize=11, verticalalignment='bottom')

        plt.gca().text(0.02, 0.95, sparsity_info_text, transform=plt.gca().transAxes, 
                       fontsize=12, fontweight='bold', color='darkgreen', 
                       verticalalignment='top', horizontalalignment='left',
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8))

        plt.xlabel("Layer Index", fontsize=14)
        plt.ylabel("Visual Retrieval Score", fontsize=14)
        plt.legend(['Top 1/3 Threshold'], loc='lower right', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.4)
        
        path_3 = os.path.join(ROOT_DIR, "Dist_3_Layer_Scatter.png")
        plt.tight_layout()
        plt.savefig(path_3, dpi=300)
        plt.close()
        print(f"Saved Plot 3 to {path_3}")
    print(f"Done.")

if __name__ == "__main__":
    main()