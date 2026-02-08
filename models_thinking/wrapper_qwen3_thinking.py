import os
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import subprocess
import gc
import torch
import json
import numpy as np
import argparse
import math
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Sequence
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor
from modeling_qwen3_vl import Qwen3VLForConditionalGeneration_masked, Qwen3VLForConditionalGeneration
from transformers import AutoProcessor
from intervener import AttentionMonitor, AttentionMasker
import torch
import gc
from abc import ABC, abstractmethod
from PIL import Image
import requests

QASPER_FEW_SHOT_TARGET_HEADS = [
        (22, 4), # Rank 1
        (20, 15), # Rank 2
        (24, 13), # Rank 2
        (23, 30), # Rank 2
        (21, 10), # Rank 2
    ]#few shot

MUSIQUE_FEW_SHOT_TARGET_HEADS = [
        (22, 4), # Rank 1
        (23, 10), # Rank 2
        (20, 15), # Rank 2
        (24, 23), # Rank 2
        (21, 10), # Rank 2
    ]#few shot

def setup_gpu_environment(threshold_mb=2000):
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'], 
            encoding='utf-8'
        )
        gpu_memory = [int(x) for x in result.strip().split('\n')]
        free_ids = [str(i) for i, mem in enumerate(gpu_memory) if mem < threshold_mb]
        
        if not free_ids:
            print("⚠️ Warning: No free GPUs found! Using default.")
            return
            
        print(f"🚀 Found free GPUs: {free_ids}. Combining for Max VRAM...")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(free_ids)
    except Exception as e:
        print(f"[Warning] GPU auto-detect failed: {e}")

# setup_gpu_environment()  # 注释掉自动检测，使用手动设置的GPU

@dataclass
class InferenceConfig:
    model_path: str
    dataset_path: str
    save_base_dir: str
    max_input_tokens: int = 256000  
    max_new_tokens: int = 3000
    max_context_chars: int = 100000000 
    save_only_last_token_query: bool = True

class BaseEngine(ABC):
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading model from {config.model_path}...")
        self.model = self._load_model()
        self.processor = self._load_processor()
        
        # 初始化注意力监控器 (Monitor)
        self.monitor = AttentionMonitor(self.model, config)
        
    def _load_model(self):
        """加载模型，通用配置"""
        # GPU设置已在文件顶部完成

        return Qwen3VLForConditionalGeneration.from_pretrained(
            self.config.model_path,
            device_map="auto",  # auto会在可见的GPU上自动分配
            trust_remote_code=True,
            attn_implementation="eager",
            torch_dtype="auto"
        ).eval()

    def _load_processor(self):
        """加载处理器"""
        return AutoProcessor.from_pretrained(
            self.config.model_path, 
            trust_remote_code=True
        )

    def clear_memory(self, *objs):
        """统一的显存清理方法"""
        for obj in objs:
            if obj is not None:
                del obj
        gc.collect()
        torch.cuda.empty_cache()

    @abstractmethod
    def run(self, *args, **kwargs):
        """子类必须实现具体的推理逻辑"""
        pass

    @abstractmethod
    def run_no_attention(self, *args, **kwargs):
        """子类必须实现的简化推理逻辑（不捕获注意力）"""
        pass

    def capture_attention(self, input_ids, attention_mask):
        """
        通用的注意力捕获逻辑 (Prefill + Probe)
        返回: attn_data (list), error (str/None)
        """
        print("  [Step 1] Capturing Attention...")
        attn_data = []
        error = None
        past_key_values = None
        prefill_out = None

        try:
            # 1. Prefill (处理 N-1 个 token)
            with torch.no_grad():
                prefill_out = self.model(
                    input_ids=input_ids[:, :-1],
                    attention_mask=attention_mask[:, :-1],
                    use_cache=True,
                    output_attentions=False
                )
            past_key_values = prefill_out.past_key_values

            # 2. Probe (处理最后一个 token，开启 Hook)
            with self.monitor:
                with torch.no_grad():
                    self.model(
                        input_ids=input_ids[:, -1:],
                        past_key_values=past_key_values,
                        attention_mask=attention_mask,
                        use_cache=True,
                        output_attentions=True
                    )
            attn_data = self.monitor.get_results()
            print(f"    -> Attention captured. Layers: {len(attn_data)}")

        except Exception as e:
            if "out of memory" in str(e).lower():
                error = "OOM during Attention Capture"
            else:
                error = str(e)
            print(f"  [Error] Task A failed: {error}")
        
        # 无论成功失败，都清理中间变量
        self.clear_memory(past_key_values, prefill_out)
        
        return attn_data, error

    def generate_answer(self, input_ids, attention_mask, seq_len):
        """
        通用的生成逻辑
        返回: answer (str)
        """
        print("  [Step 2] Generating Answer (Fresh Start)...")
        answer = ""
        try:
            pad_token_id = self.processor.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.processor.tokenizer.eos_token_id

            with torch.no_grad():
                gen_out = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pad_token_id=pad_token_id,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                    output_attentions=False,
                    use_cache=True
                )
            
            if gen_out.shape[1] > seq_len:
                new_ids = gen_out[:, seq_len:]
                answer = self.processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
            print(f"    -> Answer generated. Length: {len(answer)}")
            
        except Exception as e:
            print(f"  [Error] Task B failed: {str(e)}")
            answer = f"ERROR_GEN: {str(e)}"
            
        return answer
    
class QwenEngine_txt(BaseEngine):
    def __init__(self, config):
        super().__init__(config)  # 调用父类初始化

    def run(self, context: str, question: str):
        # 1. 构建 Prompt 和 Input
        user_text = f"{context}\n\n{question}\n Please output your answer **directly** based on the article."
        messages = [{"role": "user", "content": user_text}]
        
        text_fmt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=text_fmt, 
            return_tensors="pt", 
            padding=True, 
            truncation=False
        ).to(self.model.device)
        
        full_input_ids = inputs["input_ids"]
        full_mask = inputs["attention_mask"]
        seq_len = full_input_ids.shape[1]
        
        print(f"  Input Length: {seq_len} tokens")

        # 2. 调用父类方法获取 Attention
        attn_data, attn_error = self.capture_attention(full_input_ids, full_mask)

        # 3. 调用父类方法生成答案
        # 注意：这里我们重新传入 input_ids，确保生成过程不受之前显存状态影响
        answer = self.generate_answer(full_input_ids, full_mask, seq_len)

        # 4. 获取 Token 列表 (用于可视化或分析)
        tokens = []
        try:
            tokenizer = getattr(self.processor, "tokenizer", None)
            if tokenizer:
                tokens = tokenizer.convert_ids_to_tokens(full_input_ids[0].tolist())
        except Exception:
            pass

        # 5. 返回结果字典
        return {
            "answer": answer,
            "input_tokens": tokens,
            "attn_data": attn_data,
            "attn_error": attn_error,  # 可选：返回错误信息
            "input_len": seq_len
        }

    def run_no_attention(self, context: str, question: str):
        """
        简化推理版本：只生成答案，不捕获注意力
        """
        # 1. 构建 Prompt 和 Input (与 run 方法相同)
        user_text = f"{context}\n\n{question}\n Please output your answer **directly** based on the article."
        messages = [{"role": "user", "content": user_text}]

        text_fmt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=text_fmt,
            return_tensors="pt",
            padding=True,
            truncation=False
        ).to(self.model.device)

        full_input_ids = inputs["input_ids"]
        full_mask = inputs["attention_mask"]
        seq_len = full_input_ids.shape[1]

        print(f"  [No Attention Mode] Input Length: {seq_len} tokens")

        # 2. 直接生成答案（跳过注意力捕获）
        answer = self.generate_answer(full_input_ids, full_mask, seq_len)

        # 3. 获取 Token 列表 (用于可视化或分析)
        tokens = []
        try:
            tokenizer = getattr(self.processor, "tokenizer", None)
            if tokenizer:
                tokens = tokenizer.convert_ids_to_tokens(full_input_ids[0].tolist())
        except Exception:
            pass

        # 4. 返回简化结果字典（不包含注意力相关数据）
        return {
            "answer": answer,
            "input_tokens": tokens,
            "input_len": seq_len
        }

  
class QwenEngine_img(BaseEngine):
    def __init__(self, config):
        super().__init__(config)

    def _load_image(self, image_path):
        """辅助函数：加载图片"""
        if image_path.startswith("http"):
            return Image.open(requests.get(image_path, stream=True).raw)
        else:
            return Image.open(image_path)

    def run(self, context: str, question: str, image_paths: List[str]):
        # 1. 加载图片为 PIL 对象
        # 我们手动加载图片，而不依赖 process_vision_info
        images = [self._load_image(p) for p in image_paths]
        
        # 2. 构建多模态 Message
        # Qwen2-VL 的 chat template 能够处理这种结构
        content = []
        for _ in image_paths:
            content.append({"type": "image"}) # 这里只占位，实际图片数据通过 processor 的 images 参数传入
        
        user_text = f"{context}\n\n{question}\n Please output your answer **directly** based on the provided images and text."
        content.append({"type": "text", "text": user_text})
        
        messages = [{"role": "user", "content": content}]

        # 3. 预处理 (Processor)
        # 直接把 PIL image 列表传给 processor
        text_fmt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text_fmt],
            images=images,  # <--- 直接传 PIL Image 对象列表
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        
        # 提取参数
        full_input_ids = inputs["input_ids"]
        full_mask = inputs["attention_mask"]
        seq_len = full_input_ids.shape[1]
        
        # 提取视觉参数 (pixel_values 等)
        vision_kwargs = {k: v for k, v in inputs.items() if k not in ["input_ids", "attention_mask"]}

        print(f"  [Image Mode] Input Length: {seq_len} tokens | Images: {len(image_paths)}")

        # 4. Attention Capture
        attn_data, attn_error = self.capture_attention(full_input_ids, full_mask, **vision_kwargs)

        # 5. Generate Answer
        answer = self.generate_answer(full_input_ids, full_mask, seq_len, **vision_kwargs)

        # 6. 获取 Tokens
        tokens = []
        try:
            tokenizer = getattr(self.processor, "tokenizer", None)
            if tokenizer:
                tokens = tokenizer.convert_ids_to_tokens(full_input_ids[0].tolist())
        except Exception:
            pass

        return {
            "answer": answer,
            "input_tokens": tokens,
            "attn_data": attn_data,
            "attn_error": attn_error,
            "input_len": seq_len
        }

    def get_first_attention(self, context: str, question: str, image_paths: List[str]):
        # 1. 加载图片为 PIL 对象
        # 我们手动加载图片，而不依赖 process_vision_info
        images = [self._load_image(p) for p in image_paths]

        # 2. 构建多模态 Message
        # Qwen2-VL 的 chat template 能够处理这种结构
        content = []
        for _ in image_paths:
            content.append({"type": "image"}) # 这里只占位，实际图片数据通过 processor 的 images 参数传入

        user_text = f"{context}\n\n{question}\n Please output your answer **directly** based on the provided images and text."
        content.append({"type": "text", "text": user_text})

        messages = [{"role": "user", "content": content}]

        # 3. 预处理 (Processor)
        # 直接把 PIL image 列表传给 processor
        text_fmt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text_fmt],
            images=images,  # <--- 直接传 PIL Image 对象列表
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # 提取参数
        full_input_ids = inputs["input_ids"]
        full_mask = inputs["attention_mask"]
        seq_len = full_input_ids.shape[1]

        # 提取视觉参数 (pixel_values 等)
        vision_kwargs = {k: v for k, v in inputs.items() if k not in ["input_ids", "attention_mask"]}

        print(f"  [Image Mode] Input Length: {seq_len} tokens | Images: {len(image_paths)}")

        # 4. Attention Capture
        attn_data, attn_error = self.capture_attention(full_input_ids, full_mask, **vision_kwargs)

        # 5. 获取 Tokens
        tokens = []
        try:
            tokenizer = getattr(self.processor, "tokenizer", None)
            if tokenizer:
                tokens = tokenizer.convert_ids_to_tokens(full_input_ids[0].tolist())
        except Exception:
            pass

        return {
            "attn_data": attn_data,
            "input_tokens": tokens,
            "input_len": seq_len
        }

    def run_no_attention(self, context: str, question: str, image_paths: List[str]):
        """
        简化推理版本：只生成答案，不捕获注意力
        """
        # 1. 加载图片为 PIL 对象 (与 run 方法相同)
        images = [self._load_image(p) for p in image_paths]

        # 2. 构建多模态 Message (与 run 方法相同)
        content = []
        for _ in image_paths:
            content.append({"type": "image"})

        user_text = f"{context}\n\n{question}\n Please output your answer **directly** based on the provided images and text."
        content.append({"type": "text", "text": user_text})

        messages = [{"role": "user", "content": content}]

        # 3. 预处理 (Processor) (与 run 方法相同)
        text_fmt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text_fmt],
            images=images,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # 提取参数
        full_input_ids = inputs["input_ids"]
        full_mask = inputs["attention_mask"]
        seq_len = full_input_ids.shape[1]

        # 提取视觉参数 (pixel_values 等)
        vision_kwargs = {k: v for k, v in inputs.items() if k not in ["input_ids", "attention_mask"]}

        print(f"  [No Attention Mode] Input Length: {seq_len} tokens | Images: {len(image_paths)}")

        # 4. 直接生成答案（跳过注意力捕获）
        answer = self.generate_answer(full_input_ids, full_mask, seq_len, **vision_kwargs)

        # 5. 获取 Tokens
        tokens = []
        try:
            tokenizer = getattr(self.processor, "tokenizer", None)
            if tokenizer:
                tokens = tokenizer.convert_ids_to_tokens(full_input_ids[0].tolist())
        except Exception:
            pass

        # 6. 返回简化结果字典（不包含注意力相关数据）
        return {
            "answer": answer,
            "input_tokens": tokens,
            "input_len": seq_len
        }
        
    def capture_attention(self, input_ids, attention_mask, **kwargs):
        """
        重写以支持传入 pixel_values 等视觉参数
        """
        print("  [Step 1] Capturing Attention (Multimodal)...")
        attn_data = []
        error = None
        past_key_values = None
        prefill_out = None

        try:
            # 1. Prefill (处理 N-1 个 token)
            # 注意：kwargs 包含了 pixel_values，必须传入，否则模型不知道如何处理图片 token
            with torch.no_grad():
                prefill_out = self.model(
                    input_ids=input_ids[:, :-1],
                    attention_mask=attention_mask[:, :-1],
                    use_cache=True,
                    output_attentions=False,
                    **kwargs  # <--- 关键：传入 pixel_values, image_grid_thw
                )
            past_key_values = prefill_out.past_key_values

            # 2. Probe (处理最后一个 token)
            with self.monitor:
                with torch.no_grad():
                    self.model(
                        input_ids=input_ids[:, -1:],
                        past_key_values=past_key_values,
                        attention_mask=attention_mask, # Mask 需要是完整的
                        use_cache=True,
                        output_attentions=True,
                        # Probe 阶段通常不需要传入 pixel_values，因为视觉信息已经在 past_key_values 里了
                        # 但为了防止某些版本报错，如果 kwargs 里有且形状匹配，可以传入。
                        # Qwen2-VL 通常只在 prefill 阶段处理 vision encoder。
                    )
            attn_data = self.monitor.get_results()
            print(f"    -> Attention captured. Layers: {len(attn_data)}")

        except Exception as e:
            if "out of memory" in str(e).lower():
                error = "OOM during Attention Capture"
            else:
                error = str(e)
            print(f"  [Error] Task A failed: {error}")
            import traceback
            traceback.print_exc() # 打印详细错误堆栈方便调试
        
        self.clear_memory(past_key_values, prefill_out)
        return attn_data, error

    def generate_answer(self, input_ids, attention_mask, seq_len, **kwargs):
        """
        重写以支持传入 pixel_values 等视觉参数
        """
        print("  [Step 2] Generating Answer (Multimodal Fresh Start)...")
        answer = ""
        try:
            pad_token_id = self.processor.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.processor.tokenizer.eos_token_id

            with torch.no_grad():
                gen_out = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pad_token_id=pad_token_id,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                    output_attentions=False,
                    use_cache=True,
                    **kwargs # <--- 关键：传入 pixel_values 等
                )

            if gen_out.shape[1] > seq_len:
                new_ids = gen_out[:, seq_len:]
                answer = self.processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
            print(f"    -> Answer generated. Length: {len(answer)}")

        except Exception as e:
            print(f"  [Error] Task B failed: {str(e)}")
            answer = f"ERROR_GEN: {str(e)}"

        return answer

class QwenEngine_txt_no_eager(QwenEngine_txt):
    """QwenEngine_txt 的双模型版本：flash_attention_2推理 + eager attention捕获"""

    def __init__(self, config):
        # 不调用父类的__init__，因为我们要重写模型加载
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading dual models from {config.model_path}...")
        self.fast_model, self.attention_model = self._load_dual_models()
        self.processor = self._load_processor()

        # 初始化注意力监控器 (使用attention模型)
        self.monitor = AttentionMonitor(self.attention_model, config)

    def _load_dual_models(self):
        """加载两个模型：fast_model用于推理，attention_model用于attention捕获"""
        # GPU设置已在文件顶部完成

        # 1. 快速推理模型（flash_attention_2）
        print("  Loading fast model (flash_attention_2)...")
        fast_model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.config.model_path,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",  # 快速推理
            torch_dtype="auto"
        ).eval()

        # 2. Attention捕获模型（eager）
        print("  Loading attention model (eager)...")
        attention_model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.config.model_path,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",  # 支持attention捕获
            torch_dtype="auto"
        ).eval()

        return fast_model, attention_model

    def capture_attention(self, input_ids, attention_mask):
        """
        双模型注意力捕获逻辑：
        1. 使用fast_model（flash_attention_2）产生KV cache
        2. 使用attention_model（eager）捕获attention
        返回: attn_data (list), error (str/None)
        """
        print("  [Step 1] Capturing Attention (Dual Model)...")
        attn_data = []
        error = None
        past_key_values = None
        prefill_out = None

        try:
            # 1. 使用fast_model进行prefill，获得高效的KV cache
            print("    -> Fast model prefill...")
            with torch.no_grad():
                prefill_out = self.fast_model(
                    input_ids=input_ids[:, :-1],
                    attention_mask=attention_mask[:, :-1],
                    use_cache=True,
                    return_dict=True
                )
            past_key_values = prefill_out.past_key_values

            # 2. 使用attention_model进行probe，捕获attention
            print("    -> Attention model probe...")
            with self.monitor:
                with torch.no_grad():
                    self.attention_model(
                        input_ids=input_ids[:, -1:],
                        past_key_values=past_key_values,
                        #attention_mask=attention_mask[:, :-1],  # 使用与prefill相同的attention_mask长度
                        use_cache=True,
                        output_attentions=True
                    )
            attn_data = self.monitor.get_results()
            print(f"    -> Attention captured. Layers: {len(attn_data)}")

        except Exception as e:
            if "out of memory" in str(e).lower():
                error = "OOM during Attention Capture"
            else:
                error = str(e)
            print(f"  [Error] Dual model attention capture failed: {error}")

        # 清理显存
        self.clear_memory(past_key_values, prefill_out)

        return attn_data, error

    def generate_answer(self, input_ids, attention_mask, seq_len):
        """
        双模型答案生成：使用fast_model进行高效生成
        返回: answer (str)
        """
        print("  [Step 2] Generating Answer (Fast Model)...")
        answer = ""
        try:
            pad_token_id = self.processor.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.processor.tokenizer.eos_token_id

            # 使用fast_model进行高效生成
            with torch.no_grad():
                gen_out = self.fast_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pad_token_id=pad_token_id,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                    output_attentions=False,
                    use_cache=True
                )

            if gen_out.shape[1] > seq_len:
                new_ids = gen_out[:, seq_len:]
                answer = self.processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
            print(f"    -> Answer generated. Length: {len(answer)}")

        except Exception as e:
            print(f"  [Error] Task B failed: {str(e)}")
            answer = f"ERROR_GEN: {str(e)}"

        return answer

    def run(self, context: str, question: str):
        """
        双模型run方法：使用fast_model生成答案，attention_model捕获attention
        """
        # 1. 构建 Prompt 和 Input
        user_text = f"{context}\n\n{question}\n Please output your answer **directly** based on the article."
        messages = [{"role": "user", "content": user_text}]

        text_fmt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=text_fmt,
            return_tensors="pt",
            padding=True,
            truncation=False
        ).to(self.device)

        full_input_ids = inputs["input_ids"]
        full_mask = inputs["attention_mask"]
        seq_len = full_input_ids.shape[1]

        print(f"  Input Length: {seq_len} tokens")

        # 2. 调用双模型attention捕获
        attn_data, attn_error = self.capture_attention(full_input_ids, full_mask)

        # 3. 调用双模型答案生成
        answer = self.generate_answer(full_input_ids, full_mask, seq_len)

        # 4. 获取 Token 列表 (用于可视化或分析)
        tokens = []
        try:
            tokenizer = getattr(self.processor, "tokenizer", None)
            if tokenizer:
                tokens = tokenizer.convert_ids_to_tokens(full_input_ids[0].tolist())
        except Exception:
            pass

        # 5. 返回结果字典
        return {
            "answer": answer,
            "input_tokens": tokens,
            "attn_data": attn_data,
            "attn_error": attn_error,
            "input_len": seq_len
        }

class QwenEngine_img_no_eager(QwenEngine_img):
    """
    QwenEngine_img 的双模型版本：
    1. 继承 QwenEngine_img 的图片处理和输入构建逻辑。
    2. 使用 dual-model (flash_attn2 + eager) 架构来加速推理并捕获注意力。
    """

    def __init__(self, config):
        # 不调用父类的 __init__，因为我们需要加载双模型而不是单模型
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading dual models (Multimodal) from {config.model_path}...")
        self.fast_model, self.attention_model = self._load_dual_models()
        self.processor = self._load_processor()
        
        # 初始化注意力监控器 (绑定到 eager 模式的 attention_model)
        self.monitor = AttentionMonitor(self.attention_model, config)

    def _load_dual_models(self):
        """加载两个模型：fast_model用于推理/Prefill，attention_model用于Attention捕获"""
        # 1. 快速推理模型（flash_attention_2）
        print("  Loading fast model (flash_attention_2)...")
        fast_model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.config.model_path,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",  # 快速推理 & 高效 Prefill
            torch_dtype="auto"
        ).eval()

        # 2. Attention捕获模型（eager）
        print("  Loading attention model (eager)...")
        attention_model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.config.model_path,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",  # 支持 Hook 捕获 Attention
            torch_dtype="auto"
        ).eval()

        return fast_model, attention_model

    def capture_attention(self, input_ids, attention_mask, **kwargs):
        """
        双模型注意力捕获逻辑 (多模态版)：
        1. Fast Model: Prefill (处理图片 + 文本前缀)，生成 KV Cache。
        2. Attention Model: Probe (使用 KV Cache 处理最后一个 token)，捕获 Attention。
        """
        print("  [Step 1] Capturing Attention (Dual Model + Multimodal)...")
        attn_data = []
        error = None
        past_key_values = None
        prefill_out = None

        try:
            # --- 1. 使用 fast_model 进行 Prefill ---
            # 这一步非常关键，它负责处理庞大的视觉 Token 和长文本
            print("    -> Fast model prefill (processing images & text)...")
            with torch.no_grad():
                prefill_out = self.fast_model(
                    input_ids=input_ids[:, :-1],
                    attention_mask=attention_mask[:, :-1],
                    use_cache=True,
                    return_dict=True,
                    **kwargs  # <--- 关键：传入 pixel_values, image_grid_thw 等视觉参数
                )
            past_key_values = prefill_out.past_key_values

            # --- 2. 使用 attention_model 进行 Probe ---
            print("    -> Attention model probe...")
            with self.monitor:
                with torch.no_grad():
                    self.attention_model(
                        input_ids=input_ids[:, -1:],
                        past_key_values=past_key_values, # 传入 fast_model 计算好的 KV Cache
                        #attention_mask=attention_mask,   # 使用完整的 Mask
                        use_cache=True,
                        output_attentions=True,
                        # Probe 阶段不传入视觉参数，因为视觉信息已经在 past_key_values 中
                        # Qwen3-VL 的视觉编码只在 prefill 阶段处理
                    )
            attn_data = self.monitor.get_results()
            print(f"    -> Attention captured. Layers: {len(attn_data)}")

        except Exception as e:
            if "out of memory" in str(e).lower():
                error = "OOM during Attention Capture"
            else:
                error = str(e)
            print(f"  [Error] Dual model attention capture failed: {error}")
            import traceback
            traceback.print_exc()

        # 清理显存
        self.clear_memory(past_key_values, prefill_out)
        
        return attn_data, error

    def generate_answer(self, input_ids, attention_mask, seq_len, **kwargs):
        """
        双模型答案生成：使用 fast_model 进行高效生成，同时支持多模态参数
        """
        print("  [Step 2] Generating Answer (Fast Model + Multimodal)...")
        answer = ""
        try:
            pad_token_id = self.processor.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.processor.tokenizer.eos_token_id

            # 使用 fast_model (Flash Attn 2) 进行生成
            with torch.no_grad():
                gen_out = self.fast_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pad_token_id=pad_token_id,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                    output_attentions=False,
                    use_cache=True,
                    **kwargs # <--- 关键：传入 pixel_values 等
                )
            
            if gen_out.shape[1] > seq_len:
                new_ids = gen_out[:, seq_len:]
                answer = self.processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
            print(f"    -> Answer generated. Length: {len(answer)}")
            
        except Exception as e:
            print(f"  [Error] Task B failed: {str(e)}")
            answer = f"ERROR_GEN: {str(e)}"

        return answer

    def run(self, context: str, question: str, image_paths: List[str]):
        """
        双模型run方法：使用fast_model生成答案，attention_model捕获attention（多模态版本）
        """
        # 1. 加载图片为 PIL 对象
        images = [self._load_image(p) for p in image_paths]

        # 2. 构建多模态 Message
        content = []
        for _ in image_paths:
            content.append({"type": "image"}) # 这里只占位，实际图片数据通过 processor 的 images 参数传入

        user_text = f"{context}\n\n{question}\n Please output your answer **directly** based on the provided images and text. Don't think too much at most 5000 words."
        content.append({"type": "text", "text": user_text})

        messages = [{"role": "user", "content": content}]

        # 3. 预处理 (Processor)
        # 直接把 PIL image 列表传给 processor
        text_fmt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text_fmt],
            images=images,  # <--- 直接传 PIL Image 对象列表
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # 提取参数
        full_input_ids = inputs["input_ids"]
        full_mask = inputs["attention_mask"]
        seq_len = full_input_ids.shape[1]

        # 提取视觉参数 (pixel_values 等)
        vision_kwargs = {k: v for k, v in inputs.items() if k not in ["input_ids", "attention_mask"]}

        print(f"  [Dual Model Image Mode] Input Length: {seq_len} tokens | Images: {len(image_paths)}")

        # 4. 调用双模型attention捕获
        attn_data, attn_error = self.capture_attention(full_input_ids, full_mask, **vision_kwargs)

        # 5. 调用双模型答案生成
        answer = self.generate_answer(full_input_ids, full_mask, seq_len, **vision_kwargs)

        # 6. 获取 Token 列表 (用于可视化或分析)
        tokens = []
        try:
            tokenizer = getattr(self.processor, "tokenizer", None)
            if tokenizer:
                tokens = tokenizer.convert_ids_to_tokens(full_input_ids[0].tolist())
        except Exception:
            pass

        # 7. 返回结果字典
        return {
            "answer": answer,
            "input_tokens": tokens,
            "attn_data": attn_data,
            "attn_error": attn_error,
            "input_len": seq_len
        }

    def get_first_attention(self, context: str, question: str, image_paths: List[str]):
        """
        双模型run方法：使用fast_model生成答案，attention_model捕获attention（多模态版本）
        """
        # 1. 加载图片为 PIL 对象
        images = [self._load_image(p) for p in image_paths]

        # 2. 构建多模态 Message
        content = []
        for _ in image_paths:
            content.append({"type": "image"}) # 这里只占位，实际图片数据通过 processor 的 images 参数传入

        user_text = f"{context}\n\n{question}\n Please output your answer **directly** based on the provided images and text."
        content.append({"type": "text", "text": user_text})

        messages = [{"role": "user", "content": content}]

        # 3. 预处理 (Processor)
        # 直接把 PIL image 列表传给 processor
        text_fmt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text_fmt],
            images=images,  # <--- 直接传 PIL Image 对象列表
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # 提取参数
        full_input_ids = inputs["input_ids"]
        full_mask = inputs["attention_mask"]
        seq_len = full_input_ids.shape[1]

        # 提取视觉参数 (pixel_values 等)
        vision_kwargs = {k: v for k, v in inputs.items() if k not in ["input_ids", "attention_mask"]}

        print(f"  [Dual Model Image Mode] Input Length: {seq_len} tokens | Images: {len(image_paths)}")

        # 4. 调用双模型attention捕获
        attn_data, attn_error = self.capture_attention(full_input_ids, full_mask, **vision_kwargs)

        # 5. 获取 Tokens
        tokens = []
        try:
            tokenizer = getattr(self.processor, "tokenizer", None)
            if tokenizer:
                tokens = tokenizer.convert_ids_to_tokens(full_input_ids[0].tolist())
        except Exception:
            pass

        return {
            "attn_data": attn_data,
            "input_tokens": tokens,
            "attn_error": attn_error,
            "input_len": seq_len
        }

    def run_no_attention(self, context: str, question: str, image_paths: List[str]):
        """
        双模型版本的run_no_attention方法：只生成答案，不捕获注意力
        """
        # 1. 加载图片为 PIL 对象 (与 run 方法相同)
        images = [self._load_image(p) for p in image_paths]

        # 2. 构建多模态 Message (与 run 方法相同)
        content = []
        for _ in image_paths:
            content.append({"type": "image"})

        user_text = f"{context}\n\n{question}\n Please output your answer **directly** based on the provided images and text, don't add any other text."
        content.append({"type": "text", "text": user_text})

        messages = [{"role": "user", "content": content}]

        # 3. 预处理 (Processor) (与 run 方法相同)
        text_fmt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text_fmt],
            images=images,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)  # 使用双模型的device

        # 提取参数
        full_input_ids = inputs["input_ids"]
        full_mask = inputs["attention_mask"]
        seq_len = full_input_ids.shape[1]

        # 提取视觉参数 (pixel_values 等)
        vision_kwargs = {k: v for k, v in inputs.items() if k not in ["input_ids", "attention_mask"]}

        print(f"  [Dual Model No Attention Mode] Input Length: {seq_len} tokens | Images: {len(image_paths)}")

        # 4. 直接生成答案（跳过注意力捕获）
        answer = self.generate_answer(full_input_ids, full_mask, seq_len, **vision_kwargs)

        # 5. 获取 Tokens
        tokens = []
        try:
            tokenizer = getattr(self.processor, "tokenizer", None)
            if tokenizer:
                tokens = tokenizer.convert_ids_to_tokens(full_input_ids[0].tolist())
        except Exception:
            pass

        # 6. 返回简化结果字典（不包含注意力相关数据）
        return {
            "answer": answer,
            "input_tokens": tokens,
            "input_len": seq_len
        }


class QwenEngine_img_no_eager_entropy(QwenEngine_img_no_eager):
    """
    多模态双模型版本：逐token计算注意力分布的香农熵。
    - 每生成一个token，从 attention_model 的 Hook 拿到该步注意力并求熵
    - 实时释放注意力张量，减小显存
    """

    def __init__(self, config):
        super().__init__(config)

    @staticmethod
    def _calc_entropy(attn_data, image_token_range):
        """
        计算每个头的熵 + 平均熵，仅对图像 token（排除第一个）计算
        
        Args:
            attn_data: List[layer] -> [heads, query_len(=1), kv_len]
            image_token_range: (start_idx, end_idx) 图像 token 范围（已排除第一个）
        
        Returns:
            {
                "per_head": [[layer0_head0, ...head31], ...],  # shape: [num_layers, num_heads]
                "avg": float  # 所有头的平均熵
            }
        """
        if not attn_data:
            return {"per_head": None, "avg": None}
        
        start_idx, end_idx = image_token_range
        if start_idx >= end_idx:
            return {"per_head": None, "avg": None}
        
        entropies_per_head = []  # [num_layers, num_heads]
        all_entropies = []
        device = None  # 尽量在GPU上计算
        
        for layer_attn in attn_data:
            if layer_attn is None:
                entropies_per_head.append([None] * 32)  # 假设 32 个头
                continue

            # 确定使用的device（优先复用attn张量所在的device）
            if device is None:
                if isinstance(layer_attn, torch.Tensor):
                    device = layer_attn.device
                elif torch.cuda.is_available():
                    device = torch.device("cuda")
                else:
                    device = torch.device("cpu")

            attn_tensor = torch.as_tensor(layer_attn, dtype=torch.float32, device=device)
            if attn_tensor.ndim == 4:
                attn_tensor = attn_tensor[0]  # 去掉 batch 维 -> [heads, query_len, kv_len]
            if attn_tensor.ndim == 3 and attn_tensor.shape[1] == 1:
                attn_tensor = attn_tensor.squeeze(1)  # [heads, kv_len]
            
            if attn_tensor.numel() == 0:
                entropies_per_head.append([None] * attn_tensor.shape[0])
                continue
            
            # 只保留图像 token 的注意力（排除第一个）
            attn_tensor = attn_tensor[:, start_idx:end_idx]  # [heads, image_token_len]
            
            # 归一化（在图像 token 维度上）
            attn_sum = attn_tensor.sum(dim=-1, keepdim=True)  # [heads, 1]
            attn_tensor = attn_tensor / (attn_sum + 1e-12)
            
            # 计算每个头的熵
            probs = torch.clamp(attn_tensor, min=1e-12)
            head_entropies = -(probs * probs.log()).sum(dim=-1)  # [heads]
            
            head_entropies_list = head_entropies.detach().to("cpu").tolist()
            entropies_per_head.append(head_entropies_list)
            all_entropies.extend(head_entropies_list)
            del attn_tensor
        
        avg_entropy = float(np.mean(all_entropies)) if all_entropies else None
        
        return {
            "per_head": entropies_per_head,
            "avg": avg_entropy
        }

    @staticmethod
    def _detect_first_high(vals: Sequence[float], top_ratio: float = 0.1):
        """
        逐个token检测高熵值：
        1. 优先找第一个熵值 > 4.0 的token
        2. 如果找不到，找熵值 > 3.5 的token
        3. 以此类推，每次降低0.5，直到0为止
        返回 (idx, val, delta_vs_prev) 或 None。
        """
        if not vals:
            return None

        # 设定阈值优先级：4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0
        thresholds = [4.0, 3.5, 3.0, 2.5, 2.0]

        for thresh in thresholds:
            for i, v in enumerate(vals):
                if v > thresh:  # 注意：使用 > 而不是 >=，避免边界问题
                    prev = vals[i - 1] if i > 0 else v
                    return i, v, v - prev

        return None

    @staticmethod
    def _to_cpu_attn(attn_data):
        """将单步 attention 数据搬到 CPU，便于后续保存与返回。"""
        if attn_data is None:
            return None
        safe = []
        for layer_attn in attn_data:
            if layer_attn is None:
                safe.append(None)
            elif isinstance(layer_attn, torch.Tensor):
                safe.append(layer_attn.detach().to("cpu").tolist())
            else:
                try:
                    safe.append(torch.as_tensor(layer_attn).detach().to("cpu").tolist())
                except Exception:
                    safe.append(layer_attn)
        return safe

    def _compute_attention_at_step(self, full_input_ids, vision_kwargs, saved_generated_ids, saved_kv_cache, target_step):
        """
        修正版：只计算单步 token 的 attention，避免 OOM。
        """
        # 1. 确定当前这一步的 input_token 是什么
        # 如果 target_step 为 0，说明是 prefill 之后的第一个生成步，输入是 prompt 的最后一个 token
        # 如果 target_step > 0，输入是上一步生成的 token
        if target_step == 0:
            current_token_id = full_input_ids[:, -1:]
        else:
            # 取 saved_generated_ids 中的第 (target_step - 1) 个元素作为输入
            # 注意：saved_generated_ids[:target_step] 是当前步之前的历史，
            # 所以输入应该是这个历史的最后一个，即 saved_generated_ids[target_step - 1]
            prev_token = saved_generated_ids[target_step - 1]
            current_token_id = torch.tensor([[prev_token]], device=self.device)

        # 2. 显存清理（可选，保险起见）
        torch.cuda.empty_cache()

        # 3. 使用 attention_model 进行单步计算
        # input_ids 长度为 1，配合 saved_kv_cache，这样只计算 1 x SeqLen 的矩阵
        with self.monitor: 
            with torch.no_grad():
                outputs = self.attention_model(
                    input_ids=current_token_id, # <--- 关键修改：只传 1 个 token
                    attention_mask=None,
                    past_key_values=saved_kv_cache, 
                    output_attentions=True,
                    use_cache=False, 
                    # **vision_kwargs # 通常有了 KV cache 就不需要再传 vision_kwargs 了，除非模型特殊
                )

        # 获取attention数据
        attn_data = self.monitor.get_results()
        attention_result = self._to_cpu_attn(attn_data)
        self.monitor.clear()

        # 清理
        del outputs, attn_data
        gc.collect()

        return attention_result
    
    def run_detect_high_entropy(self, context: str, question: str, image_paths: List[str]):
        """
        优化版：双模型流式熵检测，不再需要保存所有attention。
        使用fast_model进行快速生成，实时检测熵，当发现高熵token时，
        用attention_model在特定位置计算attention，然后继续生成。

        返回字段：
            - answer
            - attention_high_logits: {idx, token, logits_entropy, delta_vs_prev, attn_data}
            - generated_tokens / generated_token_ids
            - logits_entropy_trace
            - input_len
        """
        images = [self._load_image(p) for p in image_paths]

        content = []
        for _ in image_paths:
            content.append({"type": "image"})
        user_text = f"{context}\n\n{question}\n Please output your answer **directly** based on the provided images and text."
        content.append({"type": "text", "text": user_text})
        messages = [{"role": "user", "content": content}]

        text_fmt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text_fmt],
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        full_input_ids = inputs["input_ids"]
        full_mask = inputs["attention_mask"]
        seq_len = full_input_ids.shape[1]
        vision_kwargs = {k: v for k, v in inputs.items() if k not in ["input_ids", "attention_mask"]}

        # 图像token范围
        image_token_id = self.attention_model.config.image_token_id
        input_ids_np = full_input_ids[0].cpu().numpy()
        image_token_indices = np.where(input_ids_np == image_token_id)[0]
        if len(image_token_indices) > 1:
            image_start = int(image_token_indices[1])
            image_end = int(image_token_indices[-1] + 1)
        else:
            image_start = 0
            image_end = 100
        image_token_range = (image_start, image_end)

        logits_entropy_trace = []
        generated_ids = []
        eos_token_id = self.processor.tokenizer.eos_token_id

        # Phase 1: Prefill with fast model
        print("Prefill with fast model...")
        with torch.no_grad():
            prefill_out = self.fast_model(  # 使用fast_model进行prefill
                input_ids=full_input_ids[:, :-1],
                attention_mask=full_mask[:, :-1],
                use_cache=True,
                output_attentions=False,
                return_dict=True,
                **vision_kwargs,
            )
        past_key_values = prefill_out.past_key_values
        current_input_ids = full_input_ids[:, -1:]

        print("Decoding with real-time entropy detection...")

        # Phase 2: 实时生成和熵检测
        high_entropy_detected = False
        high_entropy_idx = None
        high_entropy_val = None
        high_entropy_delta = None
        saved_kv_cache = None
        saved_generated_ids = None

        for step in range(self.config.max_new_tokens):
            cache_position = None
            if past_key_values is not None and hasattr(past_key_values, "get_seq_length"):
                past_len = past_key_values.get_seq_length()
                cache_position = torch.arange(
                    past_len, past_len + current_input_ids.shape[1],
                    device=current_input_ids.device
                )

            # 使用fast_model进行快速推理
            with torch.no_grad():
                outputs = self.fast_model(
                    input_ids=current_input_ids,
                    attention_mask=None,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    output_attentions=False,  # 不计算attention，节省显存
                    use_cache=True,
                )

            logits = outputs.logits

            # 计算logits熵
            with torch.no_grad():
                probs = torch.softmax(logits[:, -1, :], dim=-1)
                logits_entropy = -(probs * (probs + 1e-12).log()).sum(dim=-1)
                current_entropy = logits_entropy.item()
                logits_entropy_trace.append(current_entropy)

            # 实时检测高熵（使用新的逐token检测逻辑）
            if not high_entropy_detected:
                high_info = self._detect_first_high(logits_entropy_trace)
                if high_info:
                    hi_idx, hi_val, hi_delta = high_info
                    high_entropy_detected = True
                    high_entropy_idx = hi_idx
                    high_entropy_val = hi_val
                    high_entropy_delta = hi_delta

                    # 保存当前状态：高熵token之前的KV cache和已生成的token
                    saved_kv_cache = past_key_values  # 保存产生高熵token之前的KV cache
                    saved_generated_ids = generated_ids.copy()  # 保存之前生成的token

                    # 获取高熵token的信息（hi_idx对应的是已经生成的token）
                    high_entropy_token_str = None
                    try:
                        if hi_idx < len(generated_ids):
                            high_entropy_token_id = generated_ids[hi_idx]
                            high_entropy_token_str = self.processor.tokenizer.decode([high_entropy_token_id])
                        else:
                            high_entropy_token_str = f"<pending_token_at_step_{hi_idx}>"
                    except:
                        high_entropy_token_str = f"<id:{generated_ids[hi_idx] if hi_idx < len(generated_ids) else 'unknown'}>"

                    print(f"[HighLogitsEntropy] Detected at step {hi_idx}: entropy={hi_val:.6f}, delta_vs_prev={hi_delta:.6f}, token={repr(high_entropy_token_str)}")
                    break

            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            generated_ids.append(next_token.item())

            past_key_values = outputs.past_key_values
            current_input_ids = next_token.unsqueeze(1)

            # 清理中间变量
            del logits, outputs
            if step % 10 == 0:  # 每10步清理一次
                gc.collect()

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        # Phase 3: 如果检测到高熵token，用attention_model计算attention
        attention_high_logits = None
        if high_entropy_detected and saved_kv_cache is not None:
            print(f"Computing attention for high-entropy token at step {high_entropy_idx}...")
            attention_high_logits = self._compute_attention_at_step(
                full_input_ids, vision_kwargs, saved_generated_ids, saved_kv_cache, high_entropy_idx
            )

                # 获取输入的token（不包含生成的token）
        input_tokens = []
        try:
            tokenizer = getattr(self.processor, "tokenizer", None)
            if tokenizer:
                input_tokens = tokenizer.convert_ids_to_tokens(full_input_ids[0].tolist())
        except Exception:
            pass
        '''
        answer = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # 获取生成的token
        generated_tokens = []
        try:
            tokenizer = getattr(self.processor, "tokenizer", None)
            if tokenizer:
                generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids)
        except Exception:
            pass

        # 打印高熵信息
        if high_entropy_detected:
            hi_tok = generated_tokens[high_entropy_idx] if high_entropy_idx < len(generated_tokens) else None
            print(
                f"[HighLogitsEntropy] Final: entropy={high_entropy_val:.6f}, delta_vs_prev={high_entropy_delta:.6f}, token={repr(hi_tok)}"
            )'''

        return {
            "answer": "",
            "input_tokens": input_tokens,
            "generated_token_ids": generated_ids,
            "logits_entropy_trace": logits_entropy_trace,
            "attention_high_logits": attention_high_logits,
            "input_len": seq_len,
        }

    @staticmethod
    def qasper_rag(attention_data, evidence_image_path, tokens=None) -> str:
        """
        QASPER RAG：基于注意力数据从证据图片中提取相关文本信息

        Args:
            attention_data: 注意力数据（CPU上的list结构）
            evidence_image_path: 证据图片路径（merged_evidence.png）
            tokens: 输入的tokens序列，用于确定视觉token范围

        Returns:
            str: 提取的相关文本信息
        """
        if attention_data is None:
            print("attention is empty")
            return ""

        try:
            import cv2
            import math
            import json
            import numpy as np
        except ImportError as e:
            print(f"Import error in qasper_rag: {e}")
            return ""

        # 获取视觉token索引范围（类似VER_pipeline_qasper.py中的get_visual_token_indices）
        visual_indices = None
        if tokens is not None:
            vision_start_idx = None
            vision_end_idx = None
            for i, token in enumerate(tokens):
                if token == '<|vision_start|>':
                    vision_start_idx = i
                elif token == '<|vision_end|>':
                    vision_end_idx = i
            if vision_start_idx is not None and vision_end_idx is not None:
                visual_indices = (vision_start_idx + 1, vision_end_idx)
                print(f"RAG visual_indices: {visual_indices}")

        # 目标头定义 - 使用QASPER的few-shot heads
        TARGET_HEADS = QASPER_FEW_SHOT_TARGET_HEADS

        # 获取word_mapping路径（与evidence_image在同一目录）
        word_mapping_path = os.path.join(os.path.dirname(evidence_image_path), "word_mapping.json")
        if not os.path.exists(word_mapping_path):
            print(f"Warning: word_mapping.json not found at {word_mapping_path}")
            return ""

        try:
            # 加载word_mapping
            with open(word_mapping_path, 'r', encoding='utf-8') as f:
                word_data = json.load(f)

            # 获取图片尺寸
            img_info = word_data["images"][0]
            img_width = img_info["width"]
            img_height = img_info["height"]

            # 计算patch分布
            img = cv2.imread(evidence_image_path)
            if img is None:
                print(f"Warning: Could not load evidence image: {evidence_image_path}")
                return ""

            h_img, w_img = img.shape[:2]
            aspect_ratio = h_img / w_img

            # 获取视觉token索引范围和数量
            if visual_indices is not None:
                visual_start_idx, visual_end_idx = visual_indices
                visual_token_count = visual_end_idx - visual_start_idx
                print(f"RAG using visual_indices: {visual_indices}, count: {visual_token_count}")
            else:
                # 回退到原来的逻辑，使用word_mapping中的单词数量
                total_visual_tokens = len(word_data.get("words", []))
                visual_start_idx = 0
                visual_end_idx = total_visual_tokens
                visual_token_count = total_visual_tokens
                print(f"RAG fallback: visual_start_idx=0, visual_end_idx={total_visual_tokens}")

            # 使用正确的视觉token数量计算最合适的patch网格分布
            num_patches = visual_token_count
            factors = []
            for i in range(1, int(math.sqrt(num_patches)) + 1):
                if num_patches % i == 0:
                    factors.append((i, num_patches // i))
                    if i != num_patches // i:
                        factors.append((num_patches // i, i))

            best_fit = None
            best_ratio_diff = float('inf')
            for w, h in factors:
                if w > 0 and h > 0:
                    current_ratio = h / w
                    ratio_diff = abs(current_ratio - aspect_ratio)
                    if ratio_diff < best_ratio_diff:
                        best_ratio_diff = ratio_diff
                        best_fit = (w, h)

            if best_fit is None:
                grid_size = int(math.sqrt(num_patches))
                best_fit = (grid_size, grid_size)

            grid_w, grid_h = best_fit

            # 打印调试信息
            ratio = grid_h / grid_w if grid_w > 0 else 0
            print(f"RAG best_fit: {best_fit}, ratio: {ratio:.3f}, total_patches: {num_patches}, img_ratio: {aspect_ratio:.3f}")

            # 计算平均注意力向量
            avg_attn_vector = np.zeros(visual_token_count)
            valid_heads_count = 0

            for layer_idx, head_idx in TARGET_HEADS:
                if layer_idx < len(attention_data) and head_idx < len(attention_data[layer_idx]):
                    full_target_attn = np.array(attention_data[layer_idx][head_idx][0])
                    visual_attn_part = full_target_attn[visual_start_idx:visual_end_idx]

                    # 对齐长度
                    if len(visual_attn_part) < visual_token_count:
                        visual_attn_part = np.pad(visual_attn_part, (0, visual_token_count - len(visual_attn_part)))
                    elif len(visual_attn_part) > visual_token_count:
                        visual_attn_part = visual_attn_part[:visual_token_count]

                    avg_attn_vector += visual_attn_part
                    valid_heads_count += 1

            if valid_heads_count > 0:
                avg_attn_vector /= valid_heads_count

            # 获取top-10 patches的索引
            top_k = 10
            if top_k > len(avg_attn_vector):
                top_k = len(avg_attn_vector)

            top_k_indices = np.argsort(avg_attn_vector)[-top_k:]

            # 计算每个patch的像素坐标
            patch_height = img_height / grid_h
            patch_width = img_width / grid_w

            # 收集涉及的行
            involved_lines = set()

            for patch_idx in top_k_indices:
                # 计算patch在网格中的位置
                patch_row = patch_idx // grid_w
                patch_col = patch_idx % grid_w

                # 计算patch的像素坐标
                y1 = int(patch_row * patch_height)
                x1 = int(patch_col * patch_width)
                y2 = int((patch_row + 1) * patch_height)
                x2 = int((patch_col + 1) * patch_width)

                # 找到与这个patch区域相交的所有单词行
                for word_info in word_data["words"]:
                    word_bbox = word_info["bbox"]  # [x1, y1, x2, y2]

                    # 检查patch与单词bbox是否有重叠（相交）
                    # 使用AABB（轴对齐包围盒）相交检测
                    if not (x2 < word_bbox[0] or word_bbox[2] < x1 or
                            y2 < word_bbox[1] or word_bbox[3] < y1):
                        involved_lines.add(word_info["line"])

            # 提取涉及行的文本
            line_texts = {}
            for word_info in word_data["words"]:
                line_num = word_info["line"]
                if line_num in involved_lines:
                    if line_num not in line_texts:
                        line_texts[line_num] = []
                    line_texts[line_num].append(word_info["word"])

            # 合并每行的文本
            extracted_lines = []
            for line_num in sorted(line_texts.keys()):
                line_text = " ".join(line_texts[line_num])
                extracted_lines.append(line_text)

            extracted_text = "\n".join(extracted_lines)

            # 生成debug图像
            try:
                # 在merged_evidence.png上绘制top10 patches
                debug_img = cv2.imread(evidence_image_path)

                # 计算top patch坐标用于绘制
                top_patch_coords = []
                for patch_idx in top_k_indices:
                    patch_row = patch_idx // grid_w
                    patch_col = patch_idx % grid_w
                    y1 = int(patch_row * patch_height)
                    x1 = int(patch_col * patch_width)
                    y2 = int((patch_row + 1) * patch_height)
                    x2 = int((patch_col + 1) * patch_width)
                    top_patch_coords.append((x1, y1, x2, y2))

                # 在debug图像上绘制top10 patches
                for i, (x1, y1, x2, y2) in enumerate(top_patch_coords):
                    # 使用不同的颜色区分不同的patch
                    color = (0, 255 - i * 25, i * 25)  # 从蓝到红的渐变色
                    cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)

                    # 添加patch编号
                    cv2.putText(debug_img, f"{i+1}", (x1 + 5, y1 + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # 保存debug图像到merged.png同文件夹
                debug_output_path = os.path.join(os.path.dirname(evidence_image_path), "debug_top10_patches.png")
                cv2.imwrite(debug_output_path, debug_img)
                print(f"RAG debug image saved to: {debug_output_path}")
            except Exception as e:
                print(f"Warning: Failed to generate debug image: {e}")

            print(f"RAG extracted {len(involved_lines)} lines of text from {len(top_k_indices)} patches")
            return extracted_text

        except Exception as e:
            print(f"Error in qasper_rag: {e}")
            import traceback
            traceback.print_exc()
            return ""
    @staticmethod
    def musique_rag(attention_data, evidence_image_path, tokens=None) -> str:
        """
        QASPER RAG：基于注意力数据从证据图片中提取相关文本信息

        Args:
            attention_data: 注意力数据（CPU上的list结构）
            evidence_image_path: 证据图片路径（merged_evidence.png）
            tokens: 输入的tokens序列，用于确定视觉token范围

        Returns:
            str: 提取的相关文本信息
        """
        if attention_data is None:
            print("attention is empty")
            return ""

        try:
            import cv2
            import math
            import json
            import numpy as np
        except ImportError as e:
            print(f"Import error in qasper_rag: {e}")
            return ""

        # 获取视觉token索引范围（类似VER_pipeline_qasper.py中的get_visual_token_indices）
        visual_indices = None
        if tokens is not None:
            vision_start_idx = None
            vision_end_idx = None
            for i, token in enumerate(tokens):
                if token == '<|vision_start|>':
                    vision_start_idx = i
                elif token == '<|vision_end|>':
                    vision_end_idx = i
            if vision_start_idx is not None and vision_end_idx is not None:
                visual_indices = (vision_start_idx + 1, vision_end_idx)
                print(f"RAG visual_indices: {visual_indices}")

        # 目标头定义 - 使用QASPER的few-shot heads
        TARGET_HEADS = MUSIQUE_FEW_SHOT_TARGET_HEADS

        # 获取word_mapping路径（与evidence_image在同一目录）
        word_mapping_path = os.path.join(os.path.dirname(evidence_image_path), "word_mapping.json")
        if not os.path.exists(word_mapping_path):
            print(f"Warning: word_mapping.json not found at {word_mapping_path}")
            return ""

        try:
            # 加载word_mapping
            with open(word_mapping_path, 'r', encoding='utf-8') as f:
                word_data = json.load(f)

            # 获取图片尺寸
            img_info = word_data["images"][0]
            img_width = img_info["width"]
            img_height = img_info["height"]

            # 计算patch分布
            img = cv2.imread(evidence_image_path)
            if img is None:
                print(f"Warning: Could not load evidence image: {evidence_image_path}")
                return ""

            h_img, w_img = img.shape[:2]
            aspect_ratio = h_img / w_img

            # 获取视觉token索引范围和数量
            if visual_indices is not None:
                visual_start_idx, visual_end_idx = visual_indices
                visual_token_count = visual_end_idx - visual_start_idx
                print(f"RAG using visual_indices: {visual_indices}, count: {visual_token_count}")
            else:
                # 回退到原来的逻辑，使用word_mapping中的单词数量
                total_visual_tokens = len(word_data.get("words", []))
                visual_start_idx = 0
                visual_end_idx = total_visual_tokens
                visual_token_count = total_visual_tokens
                print(f"RAG fallback: visual_start_idx=0, visual_end_idx={total_visual_tokens}")

            # 使用正确的视觉token数量计算最合适的patch网格分布
            num_patches = visual_token_count
            factors = []
            for i in range(1, int(math.sqrt(num_patches)) + 1):
                if num_patches % i == 0:
                    factors.append((i, num_patches // i))
                    if i != num_patches // i:
                        factors.append((num_patches // i, i))

            best_fit = None
            best_ratio_diff = float('inf')
            for w, h in factors:
                if w > 0 and h > 0:
                    current_ratio = h / w
                    ratio_diff = abs(current_ratio - aspect_ratio)
                    if ratio_diff < best_ratio_diff:
                        best_ratio_diff = ratio_diff
                        best_fit = (w, h)

            if best_fit is None:
                grid_size = int(math.sqrt(num_patches))
                best_fit = (grid_size, grid_size)

            grid_w, grid_h = best_fit

            # 打印调试信息
            ratio = grid_h / grid_w if grid_w > 0 else 0
            print(f"RAG best_fit: {best_fit}, ratio: {ratio:.3f}, total_patches: {num_patches}, img_ratio: {aspect_ratio:.3f}")

            # 计算平均注意力向量
            avg_attn_vector = np.zeros(visual_token_count)
            valid_heads_count = 0

            for layer_idx, head_idx in TARGET_HEADS:
                if layer_idx < len(attention_data) and head_idx < len(attention_data[layer_idx]):
                    full_target_attn = np.array(attention_data[layer_idx][head_idx][0])
                    visual_attn_part = full_target_attn[visual_start_idx:visual_end_idx]

                    # 对齐长度
                    if len(visual_attn_part) < visual_token_count:
                        visual_attn_part = np.pad(visual_attn_part, (0, visual_token_count - len(visual_attn_part)))
                    elif len(visual_attn_part) > visual_token_count:
                        visual_attn_part = visual_attn_part[:visual_token_count]

                    avg_attn_vector += visual_attn_part
                    valid_heads_count += 1

            if valid_heads_count > 0:
                avg_attn_vector /= valid_heads_count

            # 获取top-10 patches的索引
            top_k = 10
            if top_k > len(avg_attn_vector):
                top_k = len(avg_attn_vector)

            top_k_indices = np.argsort(avg_attn_vector)[-top_k:]

            # 计算每个patch的像素坐标
            patch_height = img_height / grid_h
            patch_width = img_width / grid_w

            # 收集涉及的行
            involved_lines = set()

            for patch_idx in top_k_indices:
                # 计算patch在网格中的位置
                patch_row = patch_idx // grid_w
                patch_col = patch_idx % grid_w

                # 计算patch的像素坐标
                y1 = int(patch_row * patch_height)
                x1 = int(patch_col * patch_width)
                y2 = int((patch_row + 1) * patch_height)
                x2 = int((patch_col + 1) * patch_width)

                # 找到与这个patch区域相交的所有单词行
                for word_info in word_data["words"]:
                    word_bbox = word_info["bbox"]  # [x1, y1, x2, y2]

                    # 检查patch与单词bbox是否有重叠（相交）
                    # 使用AABB（轴对齐包围盒）相交检测
                    if not (x2 < word_bbox[0] or word_bbox[2] < x1 or
                            y2 < word_bbox[1] or word_bbox[3] < y1):
                        involved_lines.add(word_info["line"])

            # 提取涉及行的文本
            line_texts = {}
            for word_info in word_data["words"]:
                line_num = word_info["line"]
                if line_num in involved_lines:
                    if line_num not in line_texts:
                        line_texts[line_num] = []
                    line_texts[line_num].append(word_info["word"])

            # 合并每行的文本
            extracted_lines = []
            for line_num in sorted(line_texts.keys()):
                line_text = " ".join(line_texts[line_num])
                extracted_lines.append(line_text)

            extracted_text = "\n".join(extracted_lines)

            # 生成debug图像
            try:
                # 在merged_evidence.png上绘制top10 patches
                debug_img = cv2.imread(evidence_image_path)

                # 计算top patch坐标用于绘制
                top_patch_coords = []
                for patch_idx in top_k_indices:
                    patch_row = patch_idx // grid_w
                    patch_col = patch_idx % grid_w
                    y1 = int(patch_row * patch_height)
                    x1 = int(patch_col * patch_width)
                    y2 = int((patch_row + 1) * patch_height)
                    x2 = int((patch_col + 1) * patch_width)
                    top_patch_coords.append((x1, y1, x2, y2))

                # 在debug图像上绘制top10 patches
                for i, (x1, y1, x2, y2) in enumerate(top_patch_coords):
                    # 使用不同的颜色区分不同的patch
                    color = (0, 255 - i * 25, i * 25)  # 从蓝到红的渐变色
                    cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)

                    # 添加patch编号
                    cv2.putText(debug_img, f"{i+1}", (x1 + 5, y1 + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # 保存debug图像到merged.png同文件夹
                debug_output_path = os.path.join(os.path.dirname(evidence_image_path), "debug_top10_patches.png")
                cv2.imwrite(debug_output_path, debug_img)
                print(f"RAG debug image saved to: {debug_output_path}")
            except Exception as e:
                print(f"Warning: Failed to generate debug image: {e}")

            print(f"RAG extracted {len(involved_lines)} lines of text from {len(top_k_indices)} patches")
            return extracted_text

        except Exception as e:
            print(f"Error in qasper_rag: {e}")
            import traceback
            traceback.print_exc()
            return ""
    def run_entropy_rag(self, context: str, question: str, image_paths: List[str], dataset_name: str):
        """
        基于熵检测的RAG方法：
        1. 检测高熵token并获取其注意力数据
        2. 调用dummy_rag获取检索信息
        3. 拼接原始上下文+问题+RAG结果，重新prefill
        4. 使用fast model生成最终答案

        返回字段：
            - answer: 最终生成的答案
            - attention_data: 高熵token的注意力数据（已释放）
            - rag_info: RAG返回的信息
            - generated_tokens / generated_token_ids
            - logits_entropy_trace
            - input_len
        """
        images = [self._load_image(p) for p in image_paths]

        content = []
        for _ in image_paths:
            content.append({"type": "image"})
        user_text = f"{context}\n\n{question}\n Please output your answer **directly** based on the provided images and text."
        content.append({"type": "text", "text": user_text})
        messages = [{"role": "user", "content": content}]

        text_fmt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text_fmt],
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        full_input_ids = inputs["input_ids"]
        full_mask = inputs["attention_mask"]
        seq_len = full_input_ids.shape[1]
        vision_kwargs = {k: v for k, v in inputs.items() if k not in ["input_ids", "attention_mask"]}

        # 图像token范围
        image_token_id = self.attention_model.config.image_token_id
        input_ids_np = full_input_ids[0].cpu().numpy()
        image_token_indices = np.where(input_ids_np == image_token_id)[0]
        if len(image_token_indices) > 1:
            image_start = int(image_token_indices[1])
            image_end = int(image_token_indices[-1] + 1)
        else:
            image_start = 0
            image_end = 100
        image_token_range = (image_start, image_end)

        logits_entropy_trace = []
        generated_ids = []
        eos_token_id = self.processor.tokenizer.eos_token_id

        # Phase 1: Prefill with fast model
        print("Prefill with fast model...")
        with torch.no_grad():
            prefill_out = self.fast_model(  # 使用fast_model进行prefill
                input_ids=full_input_ids[:, :-1],
                attention_mask=full_mask[:, :-1],
                use_cache=True,
                output_attentions=False,
                return_dict=True,
                **vision_kwargs,
            )
        past_key_values = prefill_out.past_key_values
        current_input_ids = full_input_ids[:, -1:]

        print("Decoding with entropy detection for RAG...")

        # Phase 2: 实时生成和熵检测，直到检测到高熵token
        high_entropy_detected = False
        high_entropy_idx = None
        high_entropy_val = None
        high_entropy_delta = None
        saved_kv_cache = None
        saved_generated_ids = None
        attention_data = None
        rag_info = ""

        for step in range(self.config.max_new_tokens):
            cache_position = None
            if past_key_values is not None and hasattr(past_key_values, "get_seq_length"):
                past_len = past_key_values.get_seq_length()
                cache_position = torch.arange(
                    past_len, past_len + current_input_ids.shape[1],
                    device=current_input_ids.device
                )

            # 使用fast_model进行快速推理
            with torch.no_grad():
                outputs = self.fast_model(
                    input_ids=current_input_ids,
                    attention_mask=None,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    output_attentions=False,  # 不计算attention，节省显存
                    use_cache=True,
                )

            logits = outputs.logits

            # 计算logits熵
            with torch.no_grad():
                probs = torch.softmax(logits[:, -1, :], dim=-1)
                logits_entropy = -(probs * (probs + 1e-12).log()).sum(dim=-1)
                current_entropy = logits_entropy.item()
                logits_entropy_trace.append(current_entropy)

            # 实时检测高熵（使用新的逐token检测逻辑）
            if not high_entropy_detected:
                high_info = self._detect_first_high(logits_entropy_trace)
                if high_info:
                    hi_idx, hi_val, hi_delta = high_info
                    high_entropy_detected = True
                    high_entropy_idx = hi_idx
                    high_entropy_val = hi_val
                    high_entropy_delta = hi_delta

                    # 保存当前状态：高熵token之前的KV cache和已生成的token
                    saved_kv_cache = past_key_values  # 保存产生高熵token之前的KV cache
                    saved_generated_ids = generated_ids.copy()  # 保存之前生成的token

                    # 获取高熵token的信息
                    high_entropy_token_str = None
                    try:
                        if hi_idx < len(generated_ids):
                            high_entropy_token_id = generated_ids[hi_idx]
                            high_entropy_token_str = self.processor.tokenizer.decode([high_entropy_token_id])
                        else:
                            high_entropy_token_str = f"<pending_token_at_step_{hi_idx}>"
                    except:
                        high_entropy_token_str = f"<id:{generated_ids[hi_idx] if hi_idx < len(generated_ids) else 'unknown'}>"

                    print(f"[HighLogitsEntropy] Detected at step {hi_idx}: entropy={hi_val:.6f}, delta_vs_prev={hi_delta:.6f}, token={repr(high_entropy_token_str)}")

                    # 立即计算注意力数据并调用RAG
                    print(f"Computing attention for RAG at step {high_entropy_idx}...")
                    attention_data = self._compute_attention_at_step(
                        full_input_ids, vision_kwargs, saved_generated_ids, saved_kv_cache, high_entropy_idx
                    )

                    # 获取证据图片路径（将merged.png替换为merged_evidence.png）
                    evidence_image_path = None
                    for img_path in image_paths:
                        if "merged.png" in img_path:
                            evidence_image_path = img_path.replace("merged.png", "merged_evidence.png")
                            break

                    if evidence_image_path is None:
                        print("Warning: Could not find merged.png in image_paths for evidence extraction")
                        rag_info = ""
                    else:
                        # 获取完整的tokens序列（类似VER_pipeline_qasper.py中的逻辑）
                        tokens = self.processor.tokenizer.convert_ids_to_tokens(full_input_ids[0].cpu().numpy())
                        tokens = [str(token) for token in tokens]  # 确保都是字符串

                        # 根据数据集名称动态调用对应的RAG方法
                        rag_method_name = f"{dataset_name}_rag"
                        if hasattr(self, rag_method_name):
                            rag_method = getattr(self, rag_method_name)
                            rag_info = rag_method(attention_data, evidence_image_path, tokens)
                        else:
                            print(f"Warning: RAG method {rag_method_name} not found, using empty info")
                            rag_info = ""

                    # 释放注意力数据
                    attention_data = None
                    # del attention_data
                    # gc.collect()

                    print(f"RAG info: {rag_info}")

                    # 停止当前生成循环，准备重新prefill
                    break

            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            generated_ids.append(next_token.item())

            past_key_values = outputs.past_key_values
            current_input_ids = next_token.unsqueeze(1)

            # 清理中间变量
            del logits, outputs
            if step % 10 == 0:  # 每10步清理一次
                gc.collect()

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        # Phase 3: 拼接新的上下文并使用VLLM生成
        if high_entropy_detected and rag_info:
            print("Preparing new context with RAG info for VLLM generation...")
            # 拼接新的上下文：原始context + question + RAG信息
            new_context = f"{context}\n\n{question}\n\nSome text Information (Maybe useful, extracted from image): {rag_info}\n\nPlease output your answer **directly** based on the provided images and text. The answer shouldn't include reason (if not reqiured)."
        else:
            # 如果没有检测到高熵或者没有RAG信息，使用原始上下文
            new_context = f"{context}\n\n{question}\n\nPlease output your answer **directly** based on the provided images and text. The answer shouldn't include reason (if not reqiured)."

        # 创建VLLM引擎实例
        vllm_engine = VllmEngine_img(self.config)

        # 使用VLLM进行生成
        print("Generating final answer with RAG-enhanced context via VLLM...")
        vllm_result = vllm_engine.run(new_context, "", image_paths)

        # 获取最终答案
        final_answer = vllm_result["answer"]

        # 获取输入的token
        input_tokens = []
        try:
            tokenizer = getattr(self.processor, "tokenizer", None)
            if tokenizer:
                input_tokens = tokenizer.convert_ids_to_tokens(full_input_ids[0].tolist())
        except Exception:
            pass


        return {
            "answer": final_answer,
            "input_tokens": input_tokens,
            "logits_entropy_trace": logits_entropy_trace,
            "attention_data": None,  # 注意力数据已被释放
            "rag_info": rag_info,
            "input_len": seq_len,
        }
    
class VllmEngine_img():
    def __init__(self, config):
        # 不调用父类__init__，避免加载模型和占用GPU
        self.config = config
        # VLLM引擎不需要加载模型，只需要配置vLLM服务器连接
        self.vllm_url = "http://localhost:5666/v1/chat/completions"  # 使用OpenAI兼容的端点
        self.model_name = "qwen"  # 提取模型名称

    def generate_answer(self, input_ids, attention_mask, seq_len, **kwargs):
        """
        通过vLLM服务器生成答案，使用OpenAI兼容API
        """
        print("  [VLLM] Generating Answer via OpenAI-compatible API...")
        answer = ""

        try:
            from openai import OpenAI
            import base64

            # 初始化OpenAI客户端连接到vLLM服务器
            client = OpenAI(
                api_key="EMPTY",  # vLLM本地服务不需要真实key
                base_url="http://localhost:5666/v1"
            )

            # 准备消息
            messages = []

            # 处理图像数据
            if 'image_paths' in kwargs and kwargs['image_paths']:
                # 多模态消息格式
                content = []

                # 添加图像
                for image_path in kwargs['image_paths']:
                    base64_image = self._encode_image_to_base64(image_path)
                    if base64_image:
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        })

                # 添加文本
                user_text = kwargs.get('text_prompt', 'Please describe the image.')
                content.append({
                    "type": "text",
                    "text": user_text
                })

                messages.append({
                    "role": "user",
                    "content": content
                })
            else:
                # 纯文本模式（备用）
                user_text = kwargs.get('text_prompt', 'Please provide an answer.')
                messages.append({
                    "role": "user",
                    "content": user_text
                })

            # 调用vLLM服务器
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.config.max_new_tokens,
                temperature=0.0
            )

            answer = response.choices[0].message.content.strip()
            print(f"    -> VLLM Answer generated. Length: {len(answer)}")

        except Exception as e:
            answer = f"VLLM_ERROR: {str(e)}"
            print(f"  [VLLM Error] {str(e)}")

        return answer

    def run(self, context: str, question: str, image_paths: List[str]):
        """
        实现run方法，直接通过vLLM服务器生成答案
        """
        # 准备用户文本
        user_text = f"{context}\n\n{question}\n Please output your answer **directly** based on the provided images and text."

        print(f"  [VLLM Image Mode] Images: {len(image_paths)}")

        # 直接生成答案（通过vLLM服务器）
        # 传递必要的参数给generate_answer
        kwargs_for_generation = {
            'image_paths': image_paths,
            'text_prompt': user_text
        }
        answer = self.generate_answer(None, None, None, **kwargs_for_generation)

        return {
            "answer": answer,
            "input_tokens": [],  # VLLM模式下不返回tokens
            "input_len": 0
        }

    def run_no_attention(self, context: str, question: str, image_paths: List[str]):
        """
        简化版本，调用run方法
        """
        return self.run(context, question, image_paths)

    def _load_image(self, image_path):
        """辅助函数：加载图片"""
        if image_path.startswith("http"):
            return Image.open(requests.get(image_path, stream=True).raw)
        else:
            return Image.open(image_path)

    def _encode_image_to_base64(self, image_path):
        """将图片文件编码为 Base64 字符串"""
        try:
            with open(image_path, "rb") as image_file:
                import base64
                return base64.b64encode(image_file.read()).decode('utf-8')
        except IOError as e:
            print(f"错误：无法读取图片文件 {image_path}: {e}")
            return None


class QwenEngine_img_no_eager_masked(QwenEngine_img):
    """
    QwenEngine_img 的双模型版本：
    1. 继承 QwenEngine_img 的图片处理和输入构建逻辑。
    2. 使用 dual-model (flash_attn2 + eager) 架构来加速推理并捕获注意力。
    """

    def __init__(self, config):
        # 不调用父类的 __init__，因为我们需要加载双模型而不是单模型
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading dual models (Multimodal) from {config.model_path}...")
        self.fast_model, self.attention_model = self._load_dual_models()
        self.processor = self._load_processor()
        
        # 初始化注意力监控器 (绑定到 eager 模式的 attention_model)
        self.monitor = AttentionMonitor(self.attention_model, config)

    def _load_dual_models(self):
        """加载两个模型：fast_model用于推理/Prefill，attention_model用于Attention捕获"""
        print("  Loading fast model (flash_attention_2)...")
        fast_model = Qwen3VLForConditionalGeneration_masked.from_pretrained(
            self.config.model_path,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            torch_dtype="auto"
        ).eval()

        print("  Loading attention model (eager)...")
        attention_model = Qwen3VLForConditionalGeneration_masked.from_pretrained(
            self.config.model_path,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
            torch_dtype="auto"
        ).eval()

        return fast_model, attention_model

    def capture_attention(self, input_ids, attention_mask, seq_len, **kwargs):
        """
        双模型注意力捕获逻辑 (多模态版)：
        1. Fast Model: Prefill (处理图片 + 文本前缀)，生成 KV Cache。
        2. Attention Model: Probe (使用 KV Cache 处理最后一个 token)，捕获 Attention。
        """
        print("  [Step 1] Capturing Attention (Dual Model + Multimodal)...")
        attn_data = []
        error = None
        past_key_values = None
        prefill_out = None
        answer = ""
        try:
            '''# --- 1. 使用 fast_model 进行 Prefill ---
            # 这一步非常关键，它负责处理庞大的视觉 Token 和长文本
            print("    -> Fast model prefill (processing images & text)...")
            with torch.no_grad():
                prefill_out = self.fast_model(
                    input_ids=input_ids[:, :-1],
                    attention_mask=attention_mask[:, :-1],
                    use_cache=True,
                    return_dict=True,
                    **kwargs  # <--- 关键：传入 pixel_values, image_grid_thw 等视觉参数
                )
            past_key_values = prefill_out.past_key_values

            # --- 2. 使用 attention_model 进行 Probe ---
            print("    -> Attention model probe...")
            with self.monitor:
                with torch.no_grad():
                    self.attention_model.model(
                        input_ids=input_ids[:, -1:],
                        past_key_values=past_key_values, # 传入 fast_model 计算好的 KV Cache
                        #attention_mask=attention_mask,   # 使用完整的 Mask
                        use_cache=True,
                        output_attentions=True,
                        # Probe 阶段不传入视觉参数，因为视觉信息已经在 past_key_values 中
                        # Qwen3-VL 的视觉编码只在 prefill 阶段处理
                    )
            attn_data = self.monitor.get_results()
            print(f"    -> Attention captured. Layers: {len(attn_data)}")'''

            print("  [Step 2] Generating Answer (Fast Model + Multimodal with Head Mask)...")
            pad_token_id = self.processor.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.processor.tokenizer.eos_token_id

            with torch.no_grad():
                gen_out = self.attention_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pad_token_id=pad_token_id,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                    output_attentions=True,
                    use_cache=True,
                    **kwargs # <--- 关键：传入 pixel_values 等
                )
            
            if gen_out.shape[1] > seq_len:
                new_ids = gen_out[:, seq_len:]
                answer = self.processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
            print(f"    -> Answer generated. Length: {len(answer)}")
        except Exception as e:
            if "out of memory" in str(e).lower():
                error = "OOM during Attention Capture"
            else:
                error = str(e)
            print(f"  [Error] Dual model attention capture failed: {error}")
            import traceback
            traceback.print_exc()
        
        # 6. 清理显存
        self.clear_memory(past_key_values, prefill_out)

        return attn_data, error, answer

    def generate_answer(self, input_ids, attention_mask, seq_len, **kwargs):
        """
        双模型答案生成：使用 attention_model 进行高效生成，同时支持多模态参数
        """
        print("  [Step 2] Generating Answer (Fast Model + Multimodal)...")
        answer = ""
        try:
            pad_token_id = self.processor.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.processor.tokenizer.eos_token_id

            # 使用 fast_model (Flash Attn 2) 进行生成
            with torch.no_grad():
                gen_out = self.attention_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pad_token_id=pad_token_id,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                    output_attentions=False,
                    use_cache=True,
                    **kwargs # <--- 关键：传入 pixel_values 等
                )
            
            if gen_out.shape[1] > seq_len:
                new_ids = gen_out[:, seq_len:]
                answer = self.processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
            print(f"    -> Answer generated. Length: {len(answer)}")
            
        except Exception as e:
            print(f"  [Error] Task B failed: {str(e)}")
            answer = f"ERROR_GEN: {str(e)}"

        return answer

    def run(self, context: str, question: str, image_paths: List[str], mask_heads: Optional[set] = None):
        """
        双模型run方法：支持手动 mask 指定层和索引的注意力头
        mask_heads: 集合，成员为 (head_index, layer_index)
        """
        # [新增逻辑] 应用注意力头掩码
        # 无论 mask_heads 是否为 None，都调用一次以确保状态更新（传入 None 则清除旧 Mask）
        print(f"  [Masked Heads] Applying mask: {mask_heads}")
        # self.fast_model.apply_head_mask(mask_heads)
        self.attention_model.apply_head_mask(mask_heads)

        # 1. 加载图片为 PIL 对象
        images = [self._load_image(p) for p in image_paths]

        # 2. 构建多模态 Message
        content = []
        for _ in image_paths:
            content.append({"type": "image"})

        user_text = f"{context}\n\n{question}\n Please output your answer **directly** based on the provided images and text."
        content.append({"type": "text", "text": user_text})
        messages = [{"role": "user", "content": content}]

        # 3. 预处理
        text_fmt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text_fmt],
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        full_input_ids = inputs["input_ids"]
        full_mask = inputs["attention_mask"]
        seq_len = full_input_ids.shape[1]
        vision_kwargs = {k: v for k, v in inputs.items() if k not in ["input_ids", "attention_mask"]}

        print(f"  [Dual Model Masked Mode] Input: {seq_len} | Images: {len(image_paths)} | Masked Heads: {len(mask_heads) if mask_heads else 0}")

        # 4. 调用双模型捕获和生成 (逻辑不变，内部已应用 Mask)
        attn_data, attn_error, answer = self.capture_attention(full_input_ids, full_mask, seq_len, **vision_kwargs)

        tokens = []
        try:
            tokenizer = getattr(self.processor, "tokenizer", None)
            if tokenizer:
                tokens = tokenizer.convert_ids_to_tokens(full_input_ids[0].tolist())
        except Exception: pass

        return {
            "answer": answer,
            "input_tokens": tokens,
            "attn_data": attn_data,
            "attn_error": attn_error,
            "input_len": seq_len
        }
