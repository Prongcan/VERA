"""
Base classes for VERA models
提供基础引擎类和注意力管理类
"""

import os
import gc
import torch
import numpy as np
from typing import List, Dict, Optional, Any, Set
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class InferenceConfig:
    """Inference configuration (deprecated, use kwargs directly)"""
    model_path: str
    dataset_path: str
    save_base_dir: str
    max_input_tokens: int = 256000
    max_new_tokens: int = 2048
    max_context_chars: int = 100000000
    save_only_last_token_query: bool = True


class AttentionMasker:
    """Attention masker for disabling specific attention heads"""

    def __init__(self, model: torch.nn.Module, masked_heads: Set[tuple]):
        """
        初始化注意力mask器，用于将指定头的注意力权重设为0

        Args:
            model: 模型实例
            masked_heads: 要mask的头集合，格式为{(layer_idx, head_idx), ...}
        """
        self.model = model
        self.masked_heads = masked_heads
        self.hooks = []

    def _mask_hook_fn_factory(self, layer_idx: int):
        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) > 1:
                attn_weights = output[1]
                if attn_weights is None:
                    return output

                # 检查当前层是否需要mask
                heads_to_mask = [head_idx for l_idx, head_idx in self.masked_heads if l_idx == layer_idx]
                if heads_to_mask:
                    # attn_weights shape: [batch_size, num_heads, seq_len, seq_len]
                    for head_idx in heads_to_mask:
                        if head_idx < attn_weights.shape[1]:
                            attn_weights[:, head_idx, :, :] = 0.0

                # 返回修改后的输出
                return (output[0], attn_weights, *output[2:])
            return output
        return hook

    def __enter__(self):
        self.clear()
        target_modules = []
        for name, module in self.model.named_modules():
            if "self_attn" in name and "layers" in name:
                try:
                    parts = name.split('.')
                    layer_idx = int(parts[parts.index('layers') + 1])
                    # 只为需要mask的层注册hook
                    if any(l_idx == layer_idx for l_idx, _ in self.masked_heads):
                        target_modules.append((layer_idx, module))
                except:
                    continue

        for layer_idx, module in target_modules:
            handle = module.register_forward_hook(self._mask_hook_fn_factory(layer_idx))
            self.hooks.append(handle)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def clear(self):
        pass


class AttentionMonitor:
    """Attention monitor for capturing attention weights during inference"""

    def __init__(self, model: torch.nn.Module, config: Any):
        self.model = model
        self.config = config
        self.hooks = []
        self.captured_data: Dict[int, np.ndarray] = {}

    def _hook_fn_factory(self, layer_idx: int):
        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) > 1:
                attn_weights = output[1]
                if attn_weights is None:
                    return

                if self.config.save_only_last_token_query:
                    # 保留所有头，只切片 token 维度
                    attn_weights = attn_weights[:, :, -1:, :]

                self.captured_data[layer_idx] = attn_weights.detach().cpu().float().numpy()
        return hook

    def __enter__(self):
        self.clear()
        target_modules = []
        for name, module in self.model.named_modules():
            if "self_attn" in name and "layers" in name:
                try:
                    parts = name.split('.')
                    layer_idx = int(parts[parts.index('layers') + 1])
                    target_modules.append((layer_idx, module))
                except:
                    continue

        for layer_idx, module in target_modules:
            handle = module.register_forward_hook(self._hook_fn_factory(layer_idx))
            self.hooks.append(handle)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def clear(self):
        self.captured_data = {}

    def get_results(self):
        if not self.captured_data:
            return []
        results = []
        for l in sorted(self.captured_data.keys()):
            data = self.captured_data[l]
            if data is None:
                print(f"Warning: Attention data for layer {l} is None")
                continue
            try:
                # Handle different tensor shapes
                if data.ndim == 4:  # [batch, heads, seq_len, seq_len]
                    results.append(data[0].tolist())  # Take first batch
                elif data.ndim == 3:  # [heads, seq_len, seq_len]
                    results.append(data.tolist())
                else:
                    print(f"Warning: Unexpected attention data shape for layer {l}: {data.shape}")
                    results.append(data.tolist())
            except Exception as e:
                print(f"Error processing attention data for layer {l}: {e}")
                continue
        return results


class BaseEngine(ABC):
    """Base engine class for model inference"""

    def __init__(self, model_path: str, max_new_tokens: int = 2048):
        """
        Initialize base engine

        Args:
            model_path: Path to the model checkpoint
            max_new_tokens: Maximum number of tokens to generate
        """
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading model from {model_path}...")
        self.model = self._load_model()
        self.processor = self._load_processor()

        # Create a simple config object for monitor
        class SimpleConfig:
            save_only_last_token_query = True

        self.monitor = AttentionMonitor(self.model, SimpleConfig())

    def _load_model(self):
        """Load model - to be implemented by subclasses"""
        raise NotImplementedError

    def _load_processor(self):
        """Load processor - to be implemented by subclasses"""
        raise NotImplementedError

    def clear_memory(self, *objs):
        """Clear GPU memory"""
        for obj in objs:
            if obj is not None:
                del obj
        gc.collect()
        torch.cuda.empty_cache()

    @abstractmethod
    def run(self, prompt_context: str, question_text: str, image_paths: List[str],
            is_mask_heads: bool = False, heads_positions: Optional[Set[tuple]] = None):
        """
        Run inference

        Args:
            prompt_context: Prompt context
            question_text: Question text
            image_paths: List of image paths
            is_mask_heads: Whether to mask specific attention heads
            heads_positions: Set of (layer, head) tuples to mask

        Returns:
            dict: {
                "answer": str,
                "input_tokens": List[str],
                "attn_data": List,
                "attn_error": str,
                "input_len": int
            }
        """
        pass
