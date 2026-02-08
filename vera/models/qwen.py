"""
Qwen model implementations for VERA
提供 Qwen3-VL 模型的推理引擎
"""

import sys
from pathlib import Path
import torch
import requests
from PIL import Image
from typing import List, Dict, Optional, Set
from transformers import AutoProcessor

# Add parent directory to path for imports
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.modeling_qwen3_vl import Qwen3VLForConditionalGeneration_masked
from vera.models.base import BaseEngine, AttentionMonitor


class QwenEngine(BaseEngine):
    """
    Qwen3-VL multimodal inference engine (standard version)
    """

    def __init__(self, model_path: str, max_new_tokens: int = 2048):
        super().__init__(model_path, max_new_tokens)

    def _load_model(self):
        """Load Qwen3-VL model with eager attention"""
        from transformers import Qwen3VLForConditionalGeneration

        return Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
            torch_dtype="auto"
        ).eval()

    def _load_processor(self):
        """Load Qwen3-VL processor"""
        return AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

    def _load_image(self, image_path: str):
        """Load image from path or URL"""
        if image_path.startswith("http"):
            return Image.open(requests.get(image_path, stream=True).raw)
        else:
            return Image.open(image_path)

    def capture_attention(self, input_ids, attention_mask, **kwargs):
        """
        Capture attention using prefill + probe approach

        Returns:
            tuple: (attn_data, error)
        """
        print("  [Step 1] Capturing Attention...")
        attn_data = []
        error = None
        past_key_values = None
        prefill_out = None

        try:
            # Prefill (process N-1 tokens)
            with torch.no_grad():
                prefill_out = self.model(
                    input_ids=input_ids[:, :-1],
                    attention_mask=attention_mask[:, :-1],
                    use_cache=True,
                    output_attentions=False,
                    **kwargs
                )
            past_key_values = prefill_out.past_key_values

            # Probe (process last token with hooks)
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
            print(f"  [Error] Attention capture failed: {error}")

        self.clear_memory(prefill_out, past_key_values)
        return attn_data, error

    def generate_answer(self, input_ids, attention_mask, seq_len, **kwargs):
        """Generate answer from input"""
        print("  [Step 2] Generating Answer...")
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
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    output_attentions=False,
                    use_cache=True,
                    **kwargs
                )

            if gen_out.shape[1] > seq_len:
                new_ids = gen_out[:, seq_len:]
                answer = self.processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
            print(f"    -> Answer generated. Length: {len(answer)}")

        except Exception as e:
            print(f"  [Error] Generation failed: {str(e)}")
            answer = f"ERROR_GEN: {str(e)}"

        return answer

    def run(self, prompt_context: str, question_text: str, image_paths: List[str],
            is_mask_heads: bool = False, heads_positions: Optional[Set[tuple]] = None):
        """
        Run multimodal inference

        Args:
            prompt_context: Prompt context
            question_text: Question text
            image_paths: List of image paths
            is_mask_heads: Whether to mask attention heads (not used in standard version)
            heads_positions: Head positions to mask (not used in standard version)

        Returns:
            dict with keys: answer, input_tokens, attn_data, attn_error, input_len
        """
        # 1. Load images
        images = [self._load_image(p) for p in image_paths]

        # 2. Build multimodal messages
        content = []
        for _ in image_paths:
            content.append({"type": "image"})

        user_text = f"{prompt_context}\n\n{question_text}\n Please output your answer **directly** based on the provided images and text."
        content.append({"type": "text", "text": user_text})

        messages = [{"role": "user", "content": content}]

        # 3. Process inputs
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

        # Extract vision kwargs (pixel_values, etc.)
        vision_kwargs = {k: v for k, v in inputs.items() if k not in ["input_ids", "attention_mask"]}

        print(f"  [Image Mode] Input Length: {seq_len} tokens | Images: {len(image_paths)}")

        # 4. Capture attention
        attn_data, attn_error = self.capture_attention(full_input_ids, full_mask, **vision_kwargs)

        # 5. Generate answer
        answer = self.generate_answer(full_input_ids, full_mask, seq_len, **vision_kwargs)

        # 6. Get tokens
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


class QwenEngineMasked(BaseEngine):
    """
    Qwen3-VL multimodal inference engine with attention head masking support
    Uses dual-model architecture (flash_attention_2 + eager) for efficiency
    """

    def __init__(self, model_path: str, max_new_tokens: int = 2048):
        # Don't call parent __init__ to avoid loading single model
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading dual models (Multimodal) from {model_path}...")
        self.fast_model, self.attention_model = self._load_dual_models()
        self.processor = self._load_processor()

        # Initialize monitor with attention model
        class SimpleConfig:
            save_only_last_token_query = True

        self.monitor = AttentionMonitor(self.attention_model, SimpleConfig())

    def _load_dual_models(self):
        """Load two models: fast_model for inference, attention_model for attention capture"""
        from transformers import Qwen3VLForConditionalGeneration

        # 1. Fast model (flash_attention_2)
        print("  Loading fast model (flash_attention_2)...")
        fast_model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            torch_dtype="auto"
        ).eval()

        # 2. Attention model (eager, with masking support)
        print("  Loading attention model (eager with masking)...")
        attention_model = Qwen3VLForConditionalGeneration_masked.from_pretrained(
            self.model_path,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
            torch_dtype="auto"
        ).eval()

        return fast_model, attention_model

    def _load_processor(self):
        """Load Qwen3-VL processor"""
        return AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

    def _load_image(self, image_path: str):
        """Load image from path or URL"""
        if image_path.startswith("http"):
            return Image.open(requests.get(image_path, stream=True).raw)
        else:
            return Image.open(image_path)

    def run(self, prompt_context: str, question_text: str, image_paths: List[str],
            is_mask_heads: bool = False, heads_positions: Optional[Set[tuple]] = None):
        """
        Run multimodal inference with optional attention head masking

        Args:
            prompt_context: Prompt context
            question_text: Question text
            image_paths: List of image paths
            is_mask_heads: Whether to mask specific attention heads
            heads_positions: Set of (layer, head) tuples to mask

        Returns:
            dict with keys: answer, input_tokens, attn_data, attn_error, input_len
        """
        # Apply attention head mask if specified
        if is_mask_heads and heads_positions is not None:
            print(f"  [Masked Heads] Applying mask: {heads_positions}")
            self.attention_model.apply_head_mask(heads_positions)

        # 1. Load images
        images = [self._load_image(p) for p in image_paths]

        # 2. Build multimodal messages
        content = []
        for _ in image_paths:
            content.append({"type": "image"})

        user_text = f"{prompt_context}\n\n{question_text}\n Please output your answer **directly** based on the provided images and text."
        content.append({"type": "text", "text": user_text})

        messages = [{"role": "user", "content": content}]

        # 3. Process inputs
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

        # Extract vision kwargs
        vision_kwargs = {k: v for k, v in inputs.items() if k not in ["input_ids", "attention_mask"]}

        print(f"  [Dual Model Masked Mode] Input: {seq_len} | Images: {len(image_paths)} | Masked Heads: {len(heads_positions) if heads_positions else 0}")

        # 4. Capture attention and generate answer in one pass
        print("  [Step 1] Capturing Attention & Generating Answer with Mask...")
        attn_data = []
        error = None
        answer = ""

        try:
            pad_token_id = self.processor.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.processor.tokenizer.eos_token_id

            # Generate with attention monitoring
            with self.monitor:
                with torch.no_grad():
                    gen_out = self.attention_model.generate(
                        input_ids=full_input_ids,
                        attention_mask=full_mask,
                        pad_token_id=pad_token_id,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        output_attentions=True,
                        use_cache=True,
                        **vision_kwargs
                    )
            attn_data = self.monitor.get_results()

            if gen_out.shape[1] > seq_len:
                new_ids = gen_out[:, seq_len:]
                answer = self.processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
            print(f"    -> Answer generated. Length: {len(answer)}")

        except Exception as e:
            if "out of memory" in str(e).lower():
                error = "OOM during Inference"
            else:
                error = str(e)
            print(f"  [Error] Inference failed: {error}")
            import traceback
            traceback.print_exc()

        # 5. Get tokens
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
            "attn_error": error,
            "input_len": seq_len
        }
