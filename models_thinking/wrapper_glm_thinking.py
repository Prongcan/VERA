import os
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
import subprocess
import gc
import torch
import json
import numpy as np
import argparse
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Sequence
from tqdm import tqdm
from transformers import AutoProcessor
from modeling_auto import AutoModelForCausalLM, AutoModelForMultimodalLM
from intervener import AttentionMonitor, AttentionMasker
import torch
import gc
from abc import ABC, abstractmethod
from PIL import Image
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, Future

QASPER_FEW_SHOT_TARGET_HEADS = [
    (2,15),
    (27,17),
    (27,19),
    (32,4),
    (26,8)
    ]#few sh

MUSIQUE_FEW_SHOT_TARGET_HEADS = [
    (31,22),
    (29,16),
    (25,16),
    (27,2),
    (29,7)
]
DOCMATH_FEW_SHOT_TARGET_HEADS = [
    (7,11),
    (9,26),
    (9,29),
    (37,30),
    (37,22)
]
HOTPOT_FEW_SHOT_TARGET_HEADS = [
    (29, 5),
    (32, 12),
    (33,17),
    (31,22),
    (27,5)
]

TOTAL_ZERO_SHOT_TARGET_HEADS = [
    (2,15),
    (27,17),                                    
    (27,2),
    (32,4),
    (9,24),
    (26,8),
    (32,12),
    (27,19),
    (28,29),
    (27,1),
    (27,21),
    (28,16),
    (27,7),
    (32,1),
    (23,27),
    (28,2),
    (27,22),
    (27,12),
    (29,4),
    (31,22)
]

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
        return AutoModelForMultimodalLM.from_pretrained(
            self.config.model_path,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
            torch_dtype="auto",
        ).eval()

    def _load_processor(self):
        """加载处理器"""
        return AutoProcessor.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
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

    def capture_attention(self, input_ids, attention_mask, **kwargs):
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
                    output_attentions=False,
                    **kwargs,
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
                        output_attentions=True,
                        **{k: v for k, v in kwargs.items() if k not in ["pixel_values"]},
                    )
            attn_data = self.monitor.get_results()
            print(f"    -> Attention captured. Layers: {len(attn_data)}")

        except Exception as e:
            if "out of memory" in str(e).lower():
                error = "OOM during Attention Capture"
            else:
                error = str(e)
            print(f"  [Error] Task A failed: {error}")

        self.clear_memory(past_key_values, prefill_out)
        return attn_data, error

    def generate_answer(self, input_ids, attention_mask, seq_len, **kwargs):
        """
        通用的生成逻辑
        返回: answer (str)
        """
        print("  [Step 2] Generating Answer (Fresh Start)...")
        answer = ""
        try:
            pad_token_id = getattr(self.processor.tokenizer, "pad_token_id", None)
            if pad_token_id is None:
                pad_token_id = getattr(self.processor.tokenizer, "eos_token_id", None)

            with torch.no_grad():
                gen_out = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pad_token_id=pad_token_id,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                    output_attentions=False,
                    use_cache=True,
                    **kwargs,
                )

            if gen_out.shape[1] > seq_len:
                new_ids = gen_out[:, seq_len:]
                answer = self.processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
            print(f"    -> Answer generated. Length: {len(answer)}")

        except Exception as e:
            print(f"  [Error] Task B failed: {str(e)}")
            answer = f"ERROR_GEN: {str(e)}"

        return answer


class GlmEngine_txt(BaseEngine):
    def __init__(self, config):
        super().__init__(config)

    def run(self, context: str, question: str):
        user_text = f"{context}\n\n{question}\n Please output your answer **directly** based on the article."
        messages = [{"role": "user", "content": user_text}]

        text_fmt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=text_fmt,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(self.model.device)

        full_input_ids = inputs["input_ids"]
        full_mask = inputs["attention_mask"]
        seq_len = full_input_ids.shape[1]

        print(f"  Input Length: {seq_len} tokens")

        attn_data, attn_error = self.capture_attention(full_input_ids, full_mask)
        answer = self.generate_answer(full_input_ids, full_mask, seq_len)

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
            "input_len": seq_len,
        }

    def run_no_attention(self, context: str, question: str):
        user_text = f"{context}\n\n{question}\n Please output your answer **directly** based on the article."
        messages = [{"role": "user", "content": user_text}]

        text_fmt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=text_fmt,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(self.model.device)

        full_input_ids = inputs["input_ids"]
        full_mask = inputs["attention_mask"]
        seq_len = full_input_ids.shape[1]

        print(f"  [No Attention Mode] Input Length: {seq_len} tokens")

        answer = self.generate_answer(full_input_ids, full_mask, seq_len)

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
            "input_len": seq_len,
        }

class GlmEngine_img(BaseEngine):
    def __init__(self, config):
        super().__init__(config)

    def _load_image(self, image_path):
        """辅助函数：加载图片"""
        if image_path.startswith("http"):
            return Image.open(requests.get(image_path, stream=True).raw)
        else:
            return Image.open(image_path)

    def run(self, context: str, question: str, image_paths: List[str]):
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
        )
        inputs = inputs.to(self.model.device)

        full_input_ids = inputs["input_ids"]
        full_mask = inputs["attention_mask"]
        seq_len = full_input_ids.shape[1]

        vision_kwargs = {k: v for k, v in inputs.items() if k not in ["input_ids", "attention_mask"]}

        print(f"  [Image Mode] Input Length: {seq_len} tokens | Images: {len(image_paths)}")

        attn_data, attn_error = self.capture_attention(full_input_ids, full_mask, **vision_kwargs)
        answer = self.generate_answer(full_input_ids, full_mask, seq_len, **vision_kwargs)

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
            "input_len": seq_len,
        }

    def get_first_attention(self, context: str, question: str, image_paths: List[str]):
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
        )
        inputs = inputs.to(self.model.device)

        full_input_ids = inputs["input_ids"]
        full_mask = inputs["attention_mask"]
        seq_len = full_input_ids.shape[1]

        vision_kwargs = {k: v for k, v in inputs.items() if k not in ["input_ids", "attention_mask"]}

        print(f"  [Image Mode] Input Length: {seq_len} tokens | Images: {len(image_paths)}")

        attn_data, attn_error = self.capture_attention(full_input_ids, full_mask, **vision_kwargs)

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
            "input_len": seq_len,
        }

    def run_no_attention(self, context: str, question: str, image_paths: List[str]):
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
        )
        inputs = inputs.to(self.model.device)

        full_input_ids = inputs["input_ids"]
        full_mask = inputs["attention_mask"]
        seq_len = full_input_ids.shape[1]

        vision_kwargs = {k: v for k, v in inputs.items() if k not in ["input_ids", "attention_mask"]}

        print(f"  [No Attention Mode] Input Length: {seq_len} tokens | Images: {len(image_paths)}")

        answer = self.generate_answer(full_input_ids, full_mask, seq_len, **vision_kwargs)

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
            "input_len": seq_len,
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
            with torch.no_grad():
                prefill_out = self.model(
                    input_ids=input_ids[:, :-1],
                    attention_mask=attention_mask[:, :-1],
                    use_cache=True,
                    output_attentions=False,
                    **kwargs,
                )
            past_key_values = prefill_out.past_key_values

            with self.monitor:
                with torch.no_grad():
                    self.model(
                        input_ids=input_ids[:, -1:],
                        past_key_values=past_key_values,
                        attention_mask=attention_mask,
                        use_cache=True,
                        output_attentions=True,
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
            traceback.print_exc()

        self.clear_memory(past_key_values, prefill_out)
        return attn_data, error

    def generate_answer(self, input_ids, attention_mask, seq_len, **kwargs):
        """
        重写以支持传入 pixel_values 等视觉参数
        """
        print("  [Step 2] Generating Answer (Multimodal Fresh Start)...")
        answer = ""
        try:
            pad_token_id = getattr(self.processor.tokenizer, "pad_token_id", None)
            if pad_token_id is None:
                pad_token_id = getattr(self.processor.tokenizer, "eos_token_id", None)

            with torch.no_grad():
                gen_out = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pad_token_id=pad_token_id,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                    output_attentions=False,
                    use_cache=True,
                    **kwargs,
                )

            if gen_out.shape[1] > seq_len:
                new_ids = gen_out[:, seq_len:]
                answer = self.processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
            print(f"    -> Answer generated. Length: {len(answer)}")

        except Exception as e:
            print(f"  [Error] Task B failed: {str(e)}")
            answer = f"ERROR_GEN: {str(e)}"

        return answer



class GlmEngine_txt_no_eager(GlmEngine_txt):
    """GlmEngine_txt 的双模型版本：flash_attention_2推理 + eager attention捕获"""

    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading dual models from {config.model_path}...")
        self.fast_model, self.attention_model = self._load_dual_models()
        self.processor = self._load_processor()

        self.monitor = AttentionMonitor(self.attention_model, config)

    def _load_dual_models(self):
        print("  Loading fast model (flash_attention_2)...")
        fast_model = AutoModelForMultimodalLM.from_pretrained(
            self.config.model_path,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            torch_dtype="auto",
        ).eval()

        print("  Loading attention model (eager)...")
        attention_model = AutoModelForMultimodalLM.from_pretrained(
            self.config.model_path,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
            torch_dtype="auto",
        ).eval()

        return fast_model, attention_model

    def capture_attention(self, input_ids, attention_mask):
        print("  [Step 1] Capturing Attention (Dual Model)...")
        attn_data = []
        error = None
        past_key_values = None
        prefill_out = None

        try:
            print("    -> Fast model prefill...")
            with torch.no_grad():
                prefill_out = self.fast_model(
                    input_ids=input_ids[:, :-1],
                    attention_mask=attention_mask[:, :-1],
                    use_cache=True,
                    return_dict=True,
                )
            past_key_values = prefill_out.past_key_values

            print("    -> Attention model probe...")
            with self.monitor:
                with torch.no_grad():
                    self.attention_model(
                        input_ids=input_ids[:, -1:],
                        past_key_values=past_key_values,
                        use_cache=True,
                        output_attentions=True,
                    )
            attn_data = self.monitor.get_results()
            print(f"    -> Attention captured. Layers: {len(attn_data)}")

        except Exception as e:
            if "out of memory" in str(e).lower():
                error = "OOM during Attention Capture"
            else:
                error = str(e)
            print(f"  [Error] Dual model attention capture failed: {error}")

        self.clear_memory(past_key_values, prefill_out)
        return attn_data, error

    def generate_answer(self, input_ids, attention_mask, seq_len):
        print("  [Step 2] Generating Answer (Fast Model)...")
        answer = ""
        try:
            pad_token_id = getattr(self.processor.tokenizer, "pad_token_id", None)
            if pad_token_id is None:
                pad_token_id = getattr(self.processor.tokenizer, "eos_token_id", None)

            with torch.no_grad():
                gen_out = self.fast_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pad_token_id=pad_token_id,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                    output_attentions=False,
                    use_cache=True,
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
        user_text = f"{context}\n\n{question}\n Please output your answer **directly** based on the article."
        messages = [{"role": "user", "content": user_text}]

        text_fmt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=text_fmt,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(self.device)

        full_input_ids = inputs["input_ids"]
        full_mask = inputs["attention_mask"]
        seq_len = full_input_ids.shape[1]

        print(f"  Input Length: {seq_len} tokens")

        attn_data, attn_error = self.capture_attention(full_input_ids, full_mask)
        answer = self.generate_answer(full_input_ids, full_mask, seq_len)

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
            "input_len": seq_len,
        }

class GlmEngine_img_no_eager(GlmEngine_img):
    """
    GlmEngine_img 的双模型版本：
    1. 继承 GlmEngine_img 的图片处理和输入构建逻辑。
    2. 使用 dual-model (flash_attn2 + eager) 架构来加速推理并捕获注意力。
    """

    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading dual models (Multimodal) from {config.model_path}...")
        self.fast_model, self.attention_model = self._load_dual_models()
        self.processor = self._load_processor()

        self.monitor = AttentionMonitor(self.attention_model, config)

    def _load_dual_models(self):
        print("  Loading fast model (flash_attention_2)...")
        fast_model = AutoModelForMultimodalLM.from_pretrained(
            self.config.model_path,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            torch_dtype="auto",
        ).eval()

        print("  Loading attention model (eager)...")
        attention_model = AutoModelForMultimodalLM.from_pretrained(
            self.config.model_path,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
            torch_dtype="auto",
        ).eval()

        return fast_model, attention_model

    def capture_attention(self, input_ids, attention_mask, **kwargs):
        print("  [Step 1] Capturing Attention (Dual Model + Multimodal)...")
        attn_data = []
        error = None
        past_key_values = None
        prefill_out = None

        try:
            print("    -> Fast model prefill (processing images & text)...")
            with torch.no_grad():
                prefill_out = self.fast_model(
                    input_ids=input_ids[:, :-1],
                    attention_mask=attention_mask[:, :-1],
                    use_cache=True,
                    return_dict=True,
                    **kwargs,
                )
            past_key_values = prefill_out.past_key_values

            print("    -> Attention model probe...")
            with self.monitor:
                with torch.no_grad():
                    self.attention_model(
                        input_ids=input_ids[:, -1:],
                        past_key_values=past_key_values,
                        use_cache=True,
                        output_attentions=True,
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

        self.clear_memory(past_key_values, prefill_out)

        return attn_data, error

    def generate_answer(self, input_ids, attention_mask, seq_len, **kwargs):
        print("  [Step 2] Generating Answer (Fast Model + Multimodal)...")
        answer = ""
        try:
            pad_token_id = getattr(self.processor.tokenizer, "pad_token_id", None)
            if pad_token_id is None:
                pad_token_id = getattr(self.processor.tokenizer, "eos_token_id", None)

            with torch.no_grad():
                gen_out = self.fast_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pad_token_id=pad_token_id,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                    output_attentions=False,
                    use_cache=True,
                    **kwargs,
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
        images = [self._load_image(p) for p in image_paths]

        content = []
        for _ in image_paths:
            content.append({"type": "image"})

        user_text = f"{context}\n\n{question}\n Please output your answer **directly** based on the provided images and text. Don't think too much at most 5000 words."
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

        print(f"  [Dual Model Image Mode] Input Length: {seq_len} tokens | Images: {len(image_paths)}")

        attn_data, attn_error = self.capture_attention(full_input_ids, full_mask, **vision_kwargs)
        answer = self.generate_answer(full_input_ids, full_mask, seq_len, **vision_kwargs)

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
            "input_len": seq_len,
        }

    def get_first_attention(self, context: str, question: str, image_paths: List[str]):
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

        print(f"  [Dual Model Image Mode] Input Length: {seq_len} tokens | Images: {len(image_paths)}")

        attn_data, attn_error = self.capture_attention(full_input_ids, full_mask, **vision_kwargs)

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
            "input_len": seq_len,
        }

    def run_no_attention(self, context: str, question: str, image_paths: List[str]):
        images = [self._load_image(p) for p in image_paths]

        content = []
        for _ in image_paths:
            content.append({"type": "image"})

        user_text = f"{context}\n\n{question}\n Please output your answer **directly** based on the provided images and text, don't add any other text."
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
        )
        inputs = inputs.to(self.device)

        full_input_ids = inputs["input_ids"]
        full_mask = inputs["attention_mask"]
        seq_len = full_input_ids.shape[1]

        vision_kwargs = {k: v for k, v in inputs.items() if k not in ["input_ids", "attention_mask"]}

        print(f"  [Dual Model No Attention Mode] Input Length: {seq_len} tokens | Images: {len(image_paths)}")

        answer = self.generate_answer(full_input_ids, full_mask, seq_len, **vision_kwargs)

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
            "input_len": seq_len,
        }


# ================= RAG配置区域 =================
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

class GlmEngine_img_no_eager_entropy(GlmEngine_img_no_eager):
    """
    GLM多模态双模型版本：逐token计算logits熵，检测高熵token并捕获其attention分布。
    使用fast_model进行快速生成，实时检测熵，当发现高熵token时，
    用attention_model在特定位置计算attention，然后继续生成。
    """

    def __init__(self, config):
        super().__init__(config)
        # 初始化线程池用于异步VLLM任务
        self.vllm_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="vllm")
        self.vllm_futures = {}  # task_id -> future 的映射
        self.vllm_task_counter = 0  # 任务计数器，用于生成唯一task_id

    @staticmethod
    def calculate_patch_distribution(img_path, total_visual_tokens):
        """计算最合适的patch网格分布，与第一个文件保持一致"""
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

    @staticmethod
    def get_precise_bbox_mask_and_debug_img(image_path, grid_w, grid_h, folder_name, save_debug=False):
        """精确的bbox mask处理，与第一个文件保持一致"""
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
            debug_path = os.path.join(os.path.dirname(image_path), f"{folder_name}_debug_boxes.png")
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

    @staticmethod
    def calculate_entropy_confidence(prob_dist):
        """
        计算基于归一化熵的置信度，与第一个文件保持一致
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

    @staticmethod
    def unified_rag(self, attention_data, evidence_image_path, tokens=None, dataset_name="qasper") -> str:
        """
        统一的RAG方法：基于注意力数据从证据图片中提取相关文本信息

        Args:
            attention_data: 注意力数据（CPU上的list结构）
            evidence_image_path: 证据图片路径（merged_evidence.png）
            tokens: 输入的tokens序列，用于确定视觉token范围
            dataset_name: 数据集名称，用于选择合适的few-shot heads

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

        # 获取视觉token索引范围（GLM使用 <|begin_of_image|> 和 <|end_of_image|>）
        visual_indices = None
        if tokens is not None:
            vision_start_idx = None
            vision_end_idx = None
            for i, token in enumerate(tokens):
                if token == '<|begin_of_image|>':
                    vision_start_idx = i
                elif token == '<|end_of_image|>':
                    vision_end_idx = i
            if vision_start_idx is not None and vision_end_idx is not None:
                visual_indices = (vision_start_idx + 1, vision_end_idx)
                print(f"RAG visual_indices: {visual_indices}")

        # 根据数据集名称选择合适的few-shot heads
        if dataset_name.lower() == "qasper":
            TARGET_HEADS = QASPER_FEW_SHOT_TARGET_HEADS
        elif dataset_name.lower() == "musique":
            TARGET_HEADS = MUSIQUE_FEW_SHOT_TARGET_HEADS
        elif dataset_name.lower() == "docmath":
            TARGET_HEADS = DOCMATH_FEW_SHOT_TARGET_HEADS
        elif dataset_name.lower() == "hotpot":
            TARGET_HEADS = HOTPOT_FEW_SHOT_TARGET_HEADS
        elif dataset_name.lower() == "longbenchpro":
            TARGET_HEADS = TOTAL_ZERO_SHOT_TARGET_HEADS
        else:
            print(f"Warning: Unknown dataset '{dataset_name}', using QASPER heads as default")
            TARGET_HEADS = QASPER_FEW_SHOT_TARGET_HEADS

        print(evidence_image_path)
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

            # 计算注意力向量（不使用ground truth evidence信息，避免数据泄露）
            use_weighted = dataset_name.lower() == "longbenchpro"
            avg_attn_vector = np.zeros(visual_token_count)
            valid_heads_count = 0
            head_candidates = []

            for layer_idx, head_idx in TARGET_HEADS:
                if layer_idx < len(attention_data) and head_idx < len(attention_data[layer_idx]):
                    full_target_attn = np.array(attention_data[layer_idx][head_idx][0])
                    visual_attn_part = full_target_attn[visual_start_idx:visual_end_idx]

                    if use_weighted:
                        print("This is Longbench jiaquan")
                        conf = GlmEngine_img_no_eager_entropy.calculate_entropy_confidence(visual_attn_part)
                        head_candidates.append((visual_attn_part, conf))
                    else:
                        avg_attn_vector += visual_attn_part
                        valid_heads_count += 1

            if use_weighted:
                if len(head_candidates) == 0:
                    avg_attn_vector = np.zeros(visual_token_count)
                else:
                    weighted_sum_vector = np.zeros(visual_token_count)
                    total_weight = 0.0
                    for vector, weight in head_candidates:
                        weighted_sum_vector += vector * weight
                        total_weight += weight
                    if total_weight > 0:
                        avg_attn_vector = weighted_sum_vector / total_weight
                    else:
                        avg_attn_vector = np.mean([vector for vector, _ in head_candidates], axis=0)
            else:
                if valid_heads_count > 0:
                    avg_attn_vector /= valid_heads_count

            # 获取top patches的索引
            top_k = TOP_K_PATCHES
            if top_k > len(avg_attn_vector):
                top_k = len(avg_attn_vector)

            valid_patch_indices = np.argsort(avg_attn_vector)[-top_k:]

            # 计算每个patch的像素坐标（与第一个文件保持一致）
            patch_h = h_img // grid_h
            patch_w = w_img // grid_w

            # 收集涉及的行
            involved_lines = set()

            for patch_idx in valid_patch_indices:
                # 计算patch在网格中的位置
                patch_row = patch_idx // grid_w
                patch_col = patch_idx % grid_w

                # 计算patch的像素坐标（使用与第一个文件相同的精确计算）
                y1 = patch_row * patch_h
                y2 = min((patch_row + 1) * patch_h, h_img)
                x1 = patch_col * patch_w
                x2 = min((patch_col + 1) * patch_w, w_img)

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
                # 直接使用evidence图像作为背景（不再依赖ground truth标记）
                debug_img = cv2.imread(evidence_image_path)

                # 计算top patch坐标用于绘制（使用与第一个文件相同的精确计算）
                top_patch_coords = []
                for patch_idx in valid_patch_indices:
                    patch_row = patch_idx // grid_w
                    patch_col = patch_idx % grid_w
                    y1 = patch_row * patch_h
                    y2 = min((patch_row + 1) * patch_h, h_img)
                    x1 = patch_col * patch_w
                    x2 = min((patch_col + 1) * patch_w, w_img)
                    top_patch_coords.append((x1, y1, x2, y2))

                # 在debug图像上绘制top patches，使用与第一个文件一致的颜色
                for i, (x1, y1, x2, y2) in enumerate(top_patch_coords):
                    cv2.rectangle(debug_img, (x1, y1), (x2, y2), TOP_K_COLOR, 2)
                    # 添加patch编号
                    cv2.putText(debug_img, f"{i+1}", (x1 + 5, y1 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, TOP_K_COLOR, 1)

                # 保存debug图像
                debug_output_path = os.path.join(os.path.dirname(evidence_image_path), "debug_top_patches.png")
                cv2.imwrite(debug_output_path, debug_img)
                print(f"RAG debug image saved to: {debug_output_path}")
            except Exception as e:
                print(f"Warning: Failed to generate debug image: {e}")

            print(f"RAG extracted {len(involved_lines)} lines of text from {len(valid_patch_indices)} patches")
            return extracted_text

        except Exception as e:
            print(f"Error in qasper_rag: {e}")
            import traceback
            traceback.print_exc()
            return ""

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
        在特定步骤使用attention_model计算attention。
        使用保存的KV cache和已生成的token，重建到目标步骤的完整输入序列。
        """
        # 重建到目标步骤的完整输入序列
        # full_input_ids是原始输入，saved_generated_ids是高熵token之前生成的token
        target_input_ids = torch.cat([
            full_input_ids[:, :-1],  # 原始输入（不包含最后一个token，因为它在prefill时已经处理）
            torch.tensor([saved_generated_ids[:target_step]], device=self.device)  # 到目标步骤的生成token
        ], dim=1)

        # 使用attention_model进行attention计算
        with self.monitor:  # 使用monitor捕获attention
            with torch.no_grad():
                outputs = self.attention_model(
                    input_ids = target_input_ids[:, -1:],#input_ids=target_input_ids,
                    attention_mask=None,
                    past_key_values=saved_kv_cache,  # 使用保存的KV cache
                    output_attentions=True,
                    use_cache=False,  # 不需要更新KV cache，只计算attention
                    max_length=2,
                    #**vision_kwargs
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

        # GLM模型的输入构建方式：使用chat template和content格式
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

        # GLM模型的图像token范围处理
        # GLM使用 <|begin_of_image|> 开始，然后是多个 <|image|> token
        begin_image_token = "<|begin_of_image|>"
        image_token = "<|image|>"

        begin_image_id = self.processor.tokenizer.convert_tokens_to_ids(begin_image_token)
        image_id = self.processor.tokenizer.convert_tokens_to_ids(image_token)

        input_ids_np = full_input_ids[0].cpu().numpy()
        begin_indices = np.where(input_ids_np == begin_image_id)[0]
        image_indices = np.where(input_ids_np == image_id)[0]

        if len(begin_indices) > 0 and len(image_indices) > 0:
            image_start = int(begin_indices[0])
            image_end = int(image_indices[-1] + 1)
        elif len(begin_indices) > 0:
            image_start = int(begin_indices[0])
            image_end = int(begin_indices[-1] + 1)
        else:
            image_start = 0
            image_end = seq_len
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

        #answer = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # 获取输入的token（不包含生成的token）
        input_tokens = []
        try:
            tokenizer = getattr(self.processor, "tokenizer", None)
            if tokenizer:
                input_tokens = tokenizer.convert_ids_to_tokens(full_input_ids[0].tolist())
        except Exception:
            pass

        '''# 获取生成的token
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

        # GLM图像token范围处理
        begin_image_token = "<|begin_of_image|>"
        end_image_token = "<|end_of_image|>"

        begin_image_id = self.processor.tokenizer.convert_tokens_to_ids(begin_image_token)
        end_image_id = self.processor.tokenizer.convert_tokens_to_ids(end_image_token)

        input_ids_np = full_input_ids[0].cpu().numpy()
        begin_indices = np.where(input_ids_np == begin_image_id)[0]
        end_indices = np.where(input_ids_np == end_image_id)[0]

        if len(begin_indices) > 0 and len(end_indices) > 0:
            image_start = int(begin_indices[0])
            image_end = int(end_indices[-1] + 1)
        elif len(begin_indices) > 0:
            image_start = int(begin_indices[0])
            image_end = int(begin_indices[-1] + 1)
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
                        # 获取完整的tokens序列（GLM使用 <|begin_of_image|> 和 <|end_of_image|>）
                        tokens = self.processor.tokenizer.convert_ids_to_tokens(full_input_ids[0].cpu().numpy())
                        tokens = [str(token) for token in tokens]  # 确保都是字符串

                        # 调用统一的RAG方法，根据数据集名称自动选择合适的heads
                        rag_info = self.unified_rag(self, attention_data, evidence_image_path, tokens, dataset_name)

                    # 释放注意力数据
                    attention_data = None
                    del attention_data
                    gc.collect()

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

        # Phase 3: 异步处理VLLM生成，立即返回占位符
        final_answer = "[PLACEHOLDER] Answer generation in progress..."
        vllm_task_info = None
        self.vllm_task_counter += 1
        task_id = f"task_{self.vllm_task_counter}"  # 生成唯一任务ID

        if high_entropy_detected and rag_info:
            print("Preparing new context with RAG info for async VLLM generation...")

            # 拼接新的上下文：原始context + question + RAG信息
            new_context = f"{context}\n\n{question}\n\nSome text Information (Maybe useful, extracted from image): {rag_info} Judge whether you need it or not first, **do not** hesitate repeatedly. \n\n The answer shouldn't include reason (if not reqiured)."
            vllm_task_info = {
                "context": new_context,
                "image_paths": image_paths,
                "rag_info": rag_info,
                "task_id": task_id
            }
            # 异步提交VLLM任务
            self.submit_vllm_task(new_context, "", image_paths, task_id)
        else:
            # 如果没有检测到高熵或者没有RAG信息，使用简化的上下文
            new_context = f"{context}\n\n{question}\n\nJudge whether you need it or not first, **do not** hesitate repeatedly. \n\n The answer shouldn't include reason (if not reqiured)."
            vllm_task_info = {
                "context": new_context,
                "image_paths": image_paths,
                "rag_info": rag_info if rag_info else "",
                "task_id": task_id
            }
            # 异步提交VLLM任务
            self.submit_vllm_task(new_context, "", image_paths, task_id)
        

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
            "vllm_task_info": vllm_task_info,  # 异步VLLM任务信息
        }

    def _async_vllm_generation(self, context: str, question: str, image_paths: List[str], task_id: str):
        """
        异步执行VLLM生成任务

        Args:
            context: 上下文
            question: 问题（这里为空字符串）
            image_paths: 图片路径列表
            task_id: 任务ID
        """
        try:
            print(f"Starting async VLLM generation for task {task_id}...")
            # 创建VLLM引擎实例
            vllm_engine = VllmEngine_img(self.config)
            # 执行VLLM生成
            vllm_result = vllm_engine.run(context, question, image_paths)
            final_answer = vllm_result["answer"]
            print(f"Async VLLM generation completed for task {task_id}: {final_answer[:100]}...")
            return final_answer
        except Exception as e:
            print(f"Error in async VLLM generation for task {task_id}: {e}")
            return f"[ERROR] VLLM generation failed: {str(e)}"

    def submit_vllm_task(self, context: str, question: str, image_paths: List[str], task_id: str) -> Future:
        """
        提交异步VLLM生成任务

        Args:
            context: 上下文
            question: 问题
            image_paths: 图片路径列表
            task_id: 任务ID

        Returns:
            Future: 异步任务的Future对象
        """
        future = self.vllm_executor.submit(self._async_vllm_generation, context, question, image_paths, task_id)
        self.vllm_futures[task_id] = future  # 建立task_id与future的映射
        return future

    def get_vllm_result(self, task_id: str, timeout: float = 1.0) -> Optional[str]:
        """
        获取异步VLLM任务的结果

        Args:
            task_id: 任务ID
            timeout: 等待超时时间（秒）

        Returns:
            Optional[str]: 如果任务完成返回结果，否则返回None
        """
        if task_id not in self.vllm_futures:
            print(f"Task {task_id} not found in vllm_futures")
            return None

        future = self.vllm_futures[task_id]
        if not future.done():
            return None  # 任务还没完成

        try:
            result = future.result(timeout=timeout)
            # 任务完成后从字典中移除，避免内存泄漏
            del self.vllm_futures[task_id]
            return result
        except Exception as e:
            print(f"Error getting VLLM result for task {task_id}: {e}")
            # 出错也移除，避免一直占用内存
            del self.vllm_futures[task_id]
            return None

    def cleanup_completed_futures(self):
        """
        清理已完成的任务，避免内存泄漏
        """
        # 移除已完成的future（正常情况下get_vllm_result已经移除了，这里作为保险）
        completed_tasks = []
        for task_id, future in self.vllm_futures.items():
            if future.done():
                completed_tasks.append(task_id)

        for task_id in completed_tasks:
            del self.vllm_futures[task_id]

        if completed_tasks:
            print(f"Cleaned up {len(completed_tasks)} completed VLLM tasks")

    def shutdown_vllm_executor(self):
        """
        关闭VLLM线程池
        """
        if hasattr(self, 'vllm_executor'):
            self.vllm_executor.shutdown(wait=True)


class VllmEngine_img():
    def __init__(self, config):
        self.config = config
        self.vllm_url = "http://localhost:5667/v1/chat/completions"
        self.model_name = "glm"

    def generate_answer(self, input_ids, attention_mask, seq_len, **kwargs):
        print("  [VLLM] Generating Answer via OpenAI-compatible API...")
        answer = ""
        # 初始化 token 计数
        prompt_tokens = 0 

        try:
            from openai import OpenAI
            import base64

            client = OpenAI(
                api_key="EMPTY",
                base_url="http://localhost:5667/v1",
            )

            messages = []

            # ... (这部分构建 messages 的代码保持不变) ...
            if 'image_paths' in kwargs and kwargs['image_paths']:
                content = []
                for image_path in kwargs['image_paths']:
                    base64_image = self._encode_image_to_base64(image_path)
                    if base64_image:
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        })
                user_text = kwargs.get('text_prompt', 'Please describe the image.')
                content.append({"type": "text", "text": user_text})
                messages.append({"role": "user", "content": content})
            else:
                user_text = kwargs.get('text_prompt', 'Please provide an answer.')
                messages.append({"role": "user", "content": user_text})
            # ... (message 构建结束) ...

            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=25600,
                temperature=0.5,
            )

            answer = response.choices[0].message.content.strip()
            
            # --- [关键修改点 1] 获取 usage 信息 ---
            if response.usage:
                prompt_tokens = response.usage.prompt_tokens
                print(f"    -> Usage: Prompt Tokens: {prompt_tokens}")
            # ------------------------------------

            print(f"    -> VLLM Answer generated. Length: {len(answer)}")

        except Exception as e:
            answer = f"VLLM_ERROR: {str(e)}"
            print(f"  [VLLM Error] {str(e)}")

        # --- [关键修改点 2] 返回 answer 和 token 数量 ---
        return answer, prompt_tokens 

    def run(self, context: str, question: str, image_paths: List[str]):
        user_text = f"{context}\n\n{question}\n Please output your answer **directly** based on the provided images and text. "

        print(f"  [VLLM Image Mode] Images: {len(image_paths)}")

        kwargs_for_generation = {
            'image_paths': image_paths,
            'text_prompt': user_text,
        }
        
        # --- [关键修改点 3] 接收返回值 ---
        answer, input_len = self.generate_answer(None, None, None, **kwargs_for_generation)

        return {
            "answer": answer,
            "input_tokens": [], # API 不返回具体 ID 列表，通常留空即可
            "input_len": input_len, # 这里现在是真实的 token 数量了
        }

    def run_no_attention(self, context: str, question: str, image_paths: List[str]):
        return self.run(context, question, image_paths)

    def _load_image(self, image_path):
        if image_path.startswith("http"):
            return Image.open(requests.get(image_path, stream=True).raw)
        else:
            return Image.open(image_path)

    def _encode_image_to_base64(self, image_path):
        try:
            with open(image_path, "rb") as image_file:
                import base64
                return base64.b64encode(image_file.read()).decode('utf-8')
        except IOError as e:
            print(f"错误：无法读取图片文件 {image_path}: {e}")
            return None


class GlmEngine_img_no_eager_masked(GlmEngine_img):
    """
    允许对注意力头进行mask的多模态双模型版本。
    """

    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading dual models (Multimodal) from {config.model_path}...")
        self.fast_model, self.attention_model = self._load_dual_models()
        self.processor = self._load_processor()

        self.monitor = AttentionMonitor(self.attention_model, config)

    def _load_dual_models(self):
        print("  Loading fast model (flash_attention_2)...")
        fast_model = AutoModelForMultimodalLM.from_pretrained(
            self.config.model_path,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            torch_dtype="auto",
        ).eval()

        print("  Loading attention model (eager)...")
        attention_model = AutoModelForMultimodalLM.from_pretrained(
            self.config.model_path,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
            torch_dtype="auto",
        ).eval()

        return fast_model, attention_model

    def capture_attention(self, input_ids, attention_mask, seq_len, **kwargs):
        print("  [Step 1] Capturing Attention (Dual Model + Multimodal)...")
        attn_data = []
        error = None
        past_key_values = None
        prefill_out = None
        answer = ""
        try:
            print("  [Step 2] Generating Answer (Fast Model + Multimodal with Head Mask)...")
            pad_token_id = getattr(self.processor.tokenizer, "pad_token_id", None)
            if pad_token_id is None:
                pad_token_id = getattr(self.processor.tokenizer, "eos_token_id", None)

            with torch.no_grad():
                gen_out = self.attention_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pad_token_id=pad_token_id,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                    output_attentions=True,
                    use_cache=True,
                    **kwargs,
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

        self.clear_memory(past_key_values, prefill_out)

        return attn_data, error, answer

    def generate_answer(self, input_ids, attention_mask, seq_len, **kwargs):
        print("  [Step 2] Generating Answer (Fast Model + Multimodal)...")
        answer = ""
        try:
            pad_token_id = getattr(self.processor.tokenizer, "pad_token_id", None)
            if pad_token_id is None:
                pad_token_id = getattr(self.processor.tokenizer, "eos_token_id", None)

            with torch.no_grad():
                gen_out = self.attention_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pad_token_id=pad_token_id,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                    output_attentions=False,
                    use_cache=True,
                    **kwargs,
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
        print(f"  [Masked Heads] Applying mask: {mask_heads}")
        self.attention_model.apply_head_mask(mask_heads)

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

        print(f"  [Dual Model Masked Mode] Input: {seq_len} | Images: {len(image_paths)} | Masked Heads: {len(mask_heads) if mask_heads else 0}")

        attn_data, attn_error, answer = self.capture_attention(full_input_ids, full_mask, seq_len, **vision_kwargs)

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
            "input_len": seq_len,
        }

