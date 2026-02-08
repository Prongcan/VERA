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
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from tqdm import tqdm
from transformers import AutoConfig, Qwen3VLForConditionalGeneration, AutoProcessor
from modeling_qwen3_vl import Qwen3VLForConditionalGeneration_masked
from transformers import AutoProcessor
from intervener import AttentionMonitor, AttentionMasker
import torch
import gc
from abc import ABC, abstractmethod
from PIL import Image
import requests

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
    max_new_tokens: int = 2048
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


class VllmEngine_img(BaseEngine):
    def __init__(self, config):
        # 不调用父类__init__，避免加载模型和占用GPU
        self.config = config
        # VLLM引擎不需要加载模型，只需要配置vLLM服务器连接
        self.vllm_url = "http://localhost:5666/v1/chat/completions"  # 使用OpenAI兼容的端点
        self.model_name = "qwen" # 提取模型名称

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

            # 使用 AttentionMonitor 捕获 generate 过程中的注意力
            with self.monitor:
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
            attn_data = self.monitor.get_results()
            
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
