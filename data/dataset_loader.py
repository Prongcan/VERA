import json
import os
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd

class DatasetUtils:
    @staticmethod
    def load_qasper(path):
        """
        加载数据。截图显示是一个 Dict: {"PaperID": {...}, ...}
        我们需要把它转化为 Paper 列表，并把 ID 注入进去。
        """
        with open(path, "r", encoding='utf-8') as f:
            raw_data = json.load(f)
        
        papers = []
        if isinstance(raw_data, dict):
            # 兼容：如果根节点就是 PaperID 映射 (截图情况)
            # 或者包含 "data" 键
            if "data" in raw_data and isinstance(raw_data["data"], list):
                papers = raw_data["data"]
            else:
                # 遍历字典 {"1904.09131": {...}, ...}
                for pid, content in raw_data.items():
                    # 确保 content 是字典
                    if isinstance(content, dict):
                        # 注入 ID 方便后续文件名生成
                        if "id" not in content:
                            content["id"] = pid
                        papers.append(content)
        elif isinstance(raw_data, list):
            papers = raw_data
            
        return papers

    @staticmethod
    def build_context(paper, max_chars):
        title = paper.get("title", "")
        parts = []
        # full_text 是一个 List (section list)
        for sec in paper.get("full_text", []):
            # section 可能包含 paragraphs
            if isinstance(sec, dict):
                paras = sec.get("paragraphs", [])
                if isinstance(paras, list):
                    parts.extend(paras)
        
        full_text = "\n".join(parts)
        ctx = f"Title: {title}\n\nFull Text:\n{full_text}"
        return paper.get("id", "na"), ctx[:max_chars]

    @staticmethod
    def extract_evidence(qa_obj):
        """
        根据截图结构提取 Evidence。
        结构: qa_obj -> "answers" [list] -> item -> "answer" [dict] -> "evidence" / "highlighted_evidence"
        """
        evs = []
        
        # 1. 尝试直接层级 (兼容旧格式)
        evs.extend(qa_obj.get("evidence", []))
        evs.extend(qa_obj.get("highlighted_evidence", []))
        
        # 2. 遍历 "answers" 列表 (截图核心结构)
        answers_list = qa_obj.get("answers", [])
        if isinstance(answers_list, list):
            for entry in answers_list:
                # entry 对应截图中的对象
                if isinstance(entry, dict):
                    # 获取嵌套的 "answer" 字典
                    ans_data = entry.get("answer", {})
                    if isinstance(ans_data, dict):
                        evs.extend(ans_data.get("evidence", []))
                        evs.extend(ans_data.get("highlighted_evidence", []))
        
        # 3. 兼容单数 "answer"
        ans_single = qa_obj.get("answer")
        if isinstance(ans_single, dict):
             evs.extend(ans_single.get("evidence", []))
             evs.extend(ans_single.get("highlighted_evidence", []))

        # 清洗：去重、去空
        unique_evs = set()
        for e in evs:
            if isinstance(e, str) and e.strip():
                unique_evs.add(e.strip())
        
        return list(unique_evs)

    @staticmethod
    def safe_filename(s: str) -> str:
        keep = [c if c.isalnum() or c in "-_." else "_" for c in s]
        return "".join(keep)[:100]


class LongBenchLoader:
    """
    LongBench数据集加载器
    专门处理LongBench QA数据集，数据格式为JSONL，每行包含input和context字段
    """

    def __init__(self, data_path: str):
        """
        初始化加载器

        Args:
            data_path: LongBench 数据文件路径 (JSONL 格式)
        """
        self.data_path = data_path
        self.samples = []
        self._load_data()

    def _load_data(self):
        """加载 LongBench 数据"""
        print(f"Loading LongBench data from: {self.data_path}")

        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        item = json.loads(line.strip())
                        # 为每个样本添加 question_id
                        if 'question_id' not in item:
                            item['question_id'] = item.get('_id', f"longbench_{line_num-1}")
                        self.samples.append(item)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line {line_num}: {e}")
                        continue
            print(f"Successfully loaded {len(self.samples)} LongBench samples.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        except Exception as e:
            raise Exception(f"Error loading data: {e}")

    def get_samples(self, limit: Optional[int] = None) -> List[Dict]:
        """
        获取样本列表

        Args:
            limit: 限制返回的样本数量

        Returns:
            样本列表
        """
        return self.samples[:limit] if limit else self.samples

    @staticmethod
    def get_question(sample: Dict) -> str:
        """
        从样本中提取问题文本

        Args:
            sample: 样本字典

        Returns:
            问题文本
        """
        return sample.get('input', '')

    @staticmethod
    def build_context(sample: Dict, max_chars: int) -> Tuple[str, str]:
        """
        构建上下文文本

        Args:
            sample: 样本字典
            max_chars: 最大字符数限制

        Returns:
            (question_id, context_text) 元组
        """
        question_id = sample.get('question_id', 'unknown')
        context = sample.get('context', '')

        return question_id, context[:max_chars]

    @staticmethod
    def extract_evidence(sample: Dict) -> List[str]:
        """
        提取正确答案列表

        Args:
            sample: 样本字典

        Returns:
            正确答案列表
        """
        return sample.get('answers', [])

    @staticmethod
    def safe_filename(s: str) -> str:
        """
        生成安全的文件名

        Args:
            s: 原始字符串

        Returns:
            安全的文件名
        """
        import re
        # 替换不安全的字符为下划线
        safe = re.sub(r'[^\w\-_\.]', '_', s)
        # 限制长度
        return safe[:100]


class LongBenchProLoader:
    """
    LongBench-Pro 数据集加载器
    专门处理 LongBench-Pro 数据集，数据格式为 JSON，每行包含完整的样本信息
    """

    def __init__(self, data_path: str):
        """
        初始化加载器

        Args:
            data_path: LongBench-Pro 数据文件路径 (JSON 格式)
        """
        self.data_path = data_path
        self.samples = []
        self._load_data()

    def _load_data(self):
        """加载 LongBench-Pro 数据"""
        print(f"Loading LongBench-Pro data from: {self.data_path}")

        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError("LongBench-Pro data should be a JSON array")

            self.samples = data
            print(f"Successfully loaded {len(self.samples)} LongBench-Pro samples.")

        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        except json.JSONDecodeError as e:
            raise Exception(f"Error parsing JSON: {e}")
        except Exception as e:
            raise Exception(f"Error loading data: {e}")

    def get_samples(self, limit: Optional[int] = None) -> List[Dict]:
        """
        获取样本列表

        Args:
            limit: 限制返回的样本数量

        Returns:
            样本列表
        """
        return self.samples[:limit] if limit else self.samples

    @staticmethod
    def get_question(sample: Dict) -> str:
        """
        从样本中提取问题文本 (使用 question_nonthinking)

        Args:
            sample: 样本字典

        Returns:
            问题文本
        """
        return sample.get('question_nonthinking', '')

    @staticmethod
    def build_context(sample: Dict, max_chars: int) -> Tuple[str, str]:
        """
        构建上下文文本

        Args:
            sample: 样本字典
            max_chars: 最大字符数限制

        Returns:
            (question_id, context_text) 元组
        """
        question_id = sample.get('id', 'unknown')
        context = sample.get('context', '')

        return question_id, context[:max_chars]

    @staticmethod
    def extract_answer(sample: Dict) -> List[str]:
        """
        提取标准答案列表

        Args:
            sample: 样本字典

        Returns:
            标准答案列表
        """
        answer = sample.get('answer', [])
        if isinstance(answer, list):
            return answer
        elif isinstance(answer, str):
            return [answer]
        else:
            return []

    @staticmethod
    def safe_filename(s: str) -> str:
        """
        生成安全的文件名

        Args:
            s: 原始字符串

        Returns:
            安全的文件名
        """
        import re
        # 替换不安全的字符为下划线
        safe = re.sub(r'[^\w\-_\.]', '_', s)
        # 限制长度
        return safe[:100]


class TextOrPixelsLoader:
    """
    Text-or-Pixels数据集加载器
    专门处理needle-in-haystack风格的文本记忆测试数据集
    """

    def __init__(self, data_dir: str):
        """
        初始化加载器

        Args:
            data_dir: 数据集根目录路径
        """
        self.data_dir = data_dir
        self._length_folders = None
        self._samples_cache = {}

    def get_available_lengths(self) -> List[int]:
        """
        获取可用的文本长度列表

        Returns:
            排序的长度列表
        """
        if self._length_folders is None:
            all_folders = [f for f in os.listdir(self.data_dir)
                          if f.startswith('ruler_niah_single_1_len')]
            lengths = []
            for folder in all_folders:
                try:
                    length = int(folder.split('len')[-1])
                    lengths.append(length)
                except (ValueError, IndexError):
                    continue
            self._length_folders = sorted(lengths)

        return self._length_folders

    def load_data_by_lengths(self, lengths: Optional[List[int]] = None,
                           limit_per_length: Optional[int] = None) -> Tuple[List[Dict], Dict[str, int]]:
        """
        按长度加载数据集

        Args:
            lengths: 要加载的长度列表，如果为None则加载所有长度
            limit_per_length: 每个长度的最大样本数

        Returns:
            samples: 样本列表
            folder_info: 文件夹信息字典 {folder_name: sample_count}
        """
        samples = []
        folder_info = {}

        # 获取要处理的长度
        available_lengths = self.get_available_lengths()
        if lengths is None:
            target_lengths = available_lengths
        else:
            target_lengths = [l for l in lengths if l in available_lengths]
            if not target_lengths:
                print(f"Warning: None of the requested lengths {lengths} are available. "
                      f"Available: {available_lengths}")
                return samples, folder_info

        print(f"Loading data for lengths: {target_lengths}")

        for length in target_lengths:
            folder_name = f"ruler_niah_single_1_len{length}"
            folder_path = os.path.join(self.data_dir, folder_name)

            if not os.path.isdir(folder_path):
                print(f"Warning: Folder {folder_name} not found")
                continue

            # 加载该文件夹的数据
            folder_samples = self._load_single_folder(folder_path, folder_name, limit_per_length)
            samples.extend(folder_samples)
            folder_info[folder_name] = len(folder_samples)

            print(f"Loaded {len(folder_samples)} samples from {folder_name}")

        print(f"Total loaded {len(samples)} samples from {len(folder_info)} folders")
        return samples, folder_info

    def _load_single_folder(self, folder_path: str, folder_name: str,
                           limit: Optional[int] = None) -> List[Dict]:
        """
        加载单个文件夹的数据

        Args:
            folder_path: 文件夹路径
            folder_name: 文件夹名称
            limit: 最大样本数限制

        Returns:
            样本列表
        """
        # 查找jsonl文件
        jsonl_files = [f for f in os.listdir(folder_path) if f.endswith('.jsonl')]
        if not jsonl_files:
            print(f"Warning: No jsonl file found in {folder_path}")
            return []

        jsonl_file = os.path.join(folder_path, jsonl_files[0])

        # 检查缓存
        cache_key = jsonl_file
        if cache_key in self._samples_cache:
            samples = self._samples_cache[cache_key]
        else:
            samples = []
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            sample = json.loads(line.strip())
                            # 添加元数据
                            sample['folder'] = folder_name
                            sample['length_category'] = int(folder_name.split('len')[-1])
                            samples.append(sample)
                        except json.JSONDecodeError as e:
                            print(f"Warning: Failed to parse line {line_num} in {jsonl_file}: {e}")
                            continue

                # 缓存结果
                self._samples_cache[cache_key] = samples

            except Exception as e:
                print(f"Error loading {jsonl_file}: {e}")
                return []

        # 应用限制
        if limit and len(samples) > limit:
            samples = samples[:limit]

        return samples

    @staticmethod
    def extract_question_and_context(sample: Dict) -> Tuple[str, str, str]:
        """
        从text_or_pixels样本中提取问题、上下文和目标答案

        Args:
            sample: 单个样本的json对象

        Returns:
            question: 问题文本
            context: 上下文文本
            target_answer: 目标答案
        """
        doc = sample['doc']
        input_text = doc['input']
        target_answer = doc['outputs'][0] if doc['outputs'] else ""

        # 分离问题和上下文
        # 通常问题在最后，以"What is..."开头
        lines = input_text.strip().split('\n')
        question_line = None

        # 找到问题行（通常是最后一行）
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip().startswith("What is the special magic number"):
                question_line = i
                break

        if question_line is not None:
            context = '\n'.join(lines[:question_line])
            question = lines[question_line]
        else:
            # 如果找不到问题行，将整个文本作为上下文，最后一行作为问题
            context = '\n'.join(lines[:-1]) if len(lines) > 1 else ""
            question = lines[-1] if lines else ""

        return question, context, target_answer

    @staticmethod
    def get_sample_info(sample: Dict) -> Dict[str, Any]:
        """
        获取样本的基本信息

        Args:
            sample: 样本字典

        Returns:
            包含关键信息的字典
        """
        doc = sample['doc']
        question, context, target_answer = TextOrPixelsLoader.extract_question_and_context(sample)

        return {
            'doc_id': sample['doc_id'],
            'folder': sample.get('folder', ''),
            'length_category': sample.get('length_category', 0),
            'target_answer': target_answer,
            'question': question,
            'context_length': len(context),
            'has_target': bool(target_answer)
        }


class QMSumLoader:
    """
    QMSum数据集加载器
    专门处理QMSum会议摘要数据集
    """

    def __init__(self, data_path: str):
        """
        初始化加载器

        Args:
            data_path: QMSum 数据文件路径 (JSONL 格式)
        """
        self.data_path = data_path
        self.samples = []
        self._load_data()

    def _load_data(self):
        """加载 QMSum 数据"""
        print(f"Loading QMSum data from: {self.data_path}")

        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        item = json.loads(line.strip())
                        # 为每个样本添加 question_id
                        if 'question_id' not in item:
                            item['question_id'] = f"qmsum_{line_num-1}"
                        self.samples.append(item)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line {line_num}: {e}")
                        continue
            print(f"Successfully loaded {len(self.samples)} QMSum samples.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        except Exception as e:
            raise Exception(f"Error loading data: {e}")

    def get_samples(self, limit: Optional[int] = None) -> List[Dict]:
        """
        获取样本列表

        Args:
            limit: 限制返回的样本数量

        Returns:
            样本列表
        """
        return self.samples[:limit] if limit else self.samples

    @staticmethod
    def get_question(sample: Dict) -> str:
        """
        从样本中提取问题文本

        Args:
            sample: 样本字典

        Returns:
            问题文本
        """
        return sample.get('question', '')

    @staticmethod
    def build_context(sample: Dict, max_chars: int) -> Tuple[str, str]:
        """
        构建上下文文本

        Args:
            sample: 样本字典
            max_chars: 最大字符数限制

        Returns:
            (paper_id, context_text) 元组
        """
        # 对于 QMSum，每个样本就是一组问题-上下文对
        # 使用 question_id 作为 paper_id
        paper_id = sample.get('question_id', 'unknown')

        # 直接使用 context 字段
        context = sample.get('context', '')

        return paper_id, context[:max_chars]

    @staticmethod
    def extract_evidence(sample: Dict) -> str:
        """
        提取证据文本

        Args:
            sample: 样本字典

        Returns:
            证据文本
        """
        return sample.get('golden_evidence', '')


class HotpotLoader:
    """
    Hotpot数据集加载器
    专门处理Hotpot QA数据集，上下文存储在单独的文件中
    """

    def __init__(self, data_path: str, context_dir: str):
        """
        初始化加载器

        Args:
            data_path: Hotpot 数据文件路径 (JSONL 格式)
            context_dir: 上下文文件目录路径
        """
        self.data_path = data_path
        self.context_dir = context_dir
        self.samples = []
        self._load_data()

    def _load_data(self):
        """加载 Hotpot 数据"""
        print(f"Loading Hotpot data from: {self.data_path}")
        print(f"Context directory: {self.context_dir}")

        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        item = json.loads(line.strip())
                        # 为每个样本添加 question_id（使用 _id 字段）
                        if 'question_id' not in item:
                            item['question_id'] = item.get('_id', f"hotpot_{line_num-1}")

                        # 跳过证据为空的样本
                        if not item.get('golden_evidence', []):
                            continue

                        self.samples.append(item)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line {line_num}: {e}")
                        continue
            print(f"Successfully loaded {len(self.samples)} Hotpot samples (skipped empty evidence).")
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        except Exception as e:
            raise Exception(f"Error loading data: {e}")

    def get_samples(self, limit: Optional[int] = None) -> List[Dict]:
        """
        获取样本列表

        Args:
            limit: 限制返回的样本数量

        Returns:
            样本列表
        """
        return self.samples[:limit] if limit else self.samples

    @staticmethod
    def get_question(sample: Dict) -> str:
        """
        从样本中提取问题文本

        Args:
            sample: 样本字典

        Returns:
            问题文本
        """
        return sample.get('question', '')

    def build_context(self, sample: Dict, max_chars: int) -> Tuple[str, str]:
        """
        构建上下文文本

        Args:
            sample: 样本字典
            max_chars: 最大字符数限制

        Returns:
            (question_id, context_text) 元组
        """
        question_id = sample.get('question_id', 'unknown')
        context_id = sample.get('context_id', '')

        # 从对应的txt文件中读取上下文
        context_file = os.path.join(self.context_dir, f"{context_id}.txt")
        try:
            with open(context_file, 'r', encoding='utf-8') as f:
                context = f.read()
        except FileNotFoundError:
            print(f"Warning: Context file not found: {context_file}")
            context = ""
        except Exception as e:
            print(f"Warning: Error reading context file {context_file}: {e}")
            context = ""

        return question_id, context[:max_chars]

    @staticmethod
    def extract_evidence(sample: Dict) -> List[str]:
        """
        提取证据文本

        Args:
            sample: 样本字典

        Returns:
            证据文本列表
        """
        return sample.get('golden_evidence', [])

    @staticmethod
    def safe_filename(s: str) -> str:
        """
        生成安全的文件名

        Args:
            s: 原始字符串

        Returns:
            安全的文件名
        """
        import re
        # 替换不安全的字符为下划线
        safe = re.sub(r'[^\w\-_\.]', '_', s)
        # 限制长度
        return safe[:100]


class MusiqueLoader:
    """
    Musique数据集加载器
    专门处理Musique QA数据集，上下文直接存储在数据文件中
    """

    def __init__(self, data_path: str):
        """
        初始化加载器

        Args:
            data_path: Musique 数据文件路径 (JSONL 格式)
        """
        self.data_path = data_path
        self.samples = []
        self._load_data()

    def _load_data(self):
        """加载 Musique 数据"""
        print(f"Loading Musique data from: {self.data_path}")

        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        item = json.loads(line.strip())
                        # 为每个样本添加 question_id（使用 id 字段）
                        if 'question_id' not in item:
                            item['question_id'] = item.get('id', f"musique_{line_num-1}")

                        # 跳过证据为空的样本
                        if not item.get('golden_evidence', []):
                            continue

                        self.samples.append(item)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line {line_num}: {e}")
                        continue
            print(f"Successfully loaded {len(self.samples)} Musique samples (skipped empty evidence).")
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        except Exception as e:
            raise Exception(f"Error loading data: {e}")

    def get_samples(self, limit: Optional[int] = None) -> List[Dict]:
        """
        获取样本列表

        Args:
            limit: 限制返回的样本数量

        Returns:
            样本列表
        """
        return self.samples[:limit] if limit else self.samples

    @staticmethod
    def get_question(sample: Dict) -> str:
        """
        从样本中提取问题文本

        Args:
            sample: 样本字典

        Returns:
            问题文本
        """
        return sample.get('question', '')

    @staticmethod
    def build_context(sample: Dict, max_chars: int) -> Tuple[str, str]:
        """
        构建上下文文本

        Args:
            sample: 样本字典
            max_chars: 最大字符数限制

        Returns:
            (question_id, context_text) 元组
        """
        question_id = sample.get('question_id', 'unknown')

        # 直接使用 context 字段
        context = sample.get('context', '')

        return question_id, context[:max_chars]

    @staticmethod
    def extract_evidence(sample: Dict) -> List[str]:
        """
        提取证据文本

        Args:
            sample: 样本字典

        Returns:
            证据文本列表
        """
        return sample.get('golden_evidence', [])

    @staticmethod
    def safe_filename(s: str) -> str:
        """
        生成安全的文件名

        Args:
            s: 原始字符串

        Returns:
            安全的文件名
        """
        import re
        # 替换不安全的字符为下划线
        safe = re.sub(r'[^\w\-_\.]', '_', s)
        # 限制长度
        return safe[:100]


class MMLongBenchLoader:
    """
    MMLongBench-Doc数据集加载器
    专门处理MMLongBench-Doc QA数据集，上下文是PDF文档
    """

    def __init__(self, data_path: str, documents_dir: str):
        """
        初始化加载器

        Args:
            data_path: MMLongBench-Doc 数据文件路径 (samples.json)
            documents_dir: PDF文档目录路径
        """
        self.data_path = data_path
        self.documents_dir = documents_dir
        self.samples = []
        self._load_data()

    def _load_data(self):
        """加载 MMLongBench-Doc 数据"""
        print(f"Loading MMLongBench-Doc data from: {self.data_path}")
        print(f"Documents directory: {self.documents_dir}")

        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 如果是列表格式，直接处理
            if isinstance(data, list):
                self.samples = data
            else:
                print(f"Warning: Unexpected data format in {self.data_path}")
                self.samples = []

            print(f"Successfully loaded {len(self.samples)} MMLongBench-Doc samples.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        except Exception as e:
            raise Exception(f"Error loading data: {e}")

    def get_samples(self, limit: Optional[int] = None) -> List[Dict]:
        """
        获取样本列表

        Args:
            limit: 限制返回的样本数量

        Returns:
            样本列表
        """
        return self.samples[:limit] if limit else self.samples

    @staticmethod
    def get_question(sample: Dict) -> str:
        """
        从样本中提取问题文本

        Args:
            sample: 样本字典

        Returns:
            问题文本
        """
        return sample.get("question", "")

    def build_context(self, sample: Dict, max_chars: int) -> Tuple[str, str]:
        """
        构建上下文 - 对于MMLongBench-Doc，返回PDF路径而不是文本

        Args:
            sample: 样本字典
            max_chars: 最大字符数限制（这里不使用）

        Returns:
            (doc_id, pdf_path) 元组
        """
        doc_id = sample.get("doc_id", "unknown")
        pdf_path = os.path.join(self.documents_dir, doc_id)

        # 验证PDF文件是否存在
        if not os.path.exists(pdf_path):
            print(f"Warning: PDF file not found: {pdf_path}")

        return doc_id, pdf_path

    @staticmethod
    def extract_evidence(sample: Dict) -> str:
        """
        提取证据信息

        Args:
            sample: 样本字典

        Returns:
            证据字符串（页面和来源信息）
        """
        evidence_pages = sample.get("evidence_pages", "")
        evidence_sources = sample.get("evidence_sources", "")
        return f"Pages: {evidence_pages}, Sources: {evidence_sources}"

    @staticmethod
    def safe_filename(s: str) -> str:
        """
        生成安全的文件名

        Args:
            s: 原始字符串

        Returns:
            安全的文件名
        """
        import re
        # 替换不安全的字符为下划线
        safe = re.sub(r"[^\w\-_\.]", "_", s)
        # 限制长度
        return safe[:100]


class DocMathLoader:
    """
    DocMath数据集加载器
    专门处理DocMath数学问题数据集，上下文直接存储在数据文件中
    """

    def __init__(self, data_path: str):
        """
        初始化加载器

        Args:
            data_path: DocMath 数据文件路径 (JSONL 格式)
        """
        self.data_path = data_path
        self.samples = []
        self._load_data()

    def _load_data(self):
        """加载 DocMath 数据"""
        print(f"Loading DocMath data from: {self.data_path}")

        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        item = json.loads(line.strip())
                        # 确保 question_id 存在
                        if 'question_id' not in item:
                            item['question_id'] = f"docmath_{line_num-1}"
                        self.samples.append(item)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line {line_num}: {e}")
                        continue
            print(f"Successfully loaded {len(self.samples)} DocMath samples.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        except Exception as e:
            raise Exception(f"Error loading data: {e}")

    def get_samples(self, limit: Optional[int] = None) -> List[Dict]:
        """
        获取样本列表

        Args:
            limit: 限制返回的样本数量

        Returns:
            样本列表
        """
        return self.samples[:limit] if limit else self.samples

    @staticmethod
    def get_question(sample: Dict) -> str:
        """
        从样本中提取问题文本

        Args:
            sample: 样本字典

        Returns:
            问题文本
        """
        return sample.get('question', '')

    @staticmethod
    def build_context(sample: Dict, max_chars: int) -> Tuple[str, str]:
        """
        构建上下文文本

        Args:
            sample: 样本字典
            max_chars: 最大字符数限制

        Returns:
            (question_id, context_text) 元组
        """
        question_id = sample.get('question_id', 'unknown')
        
        # 直接使用 context 字段
        context = sample.get('context', '')

        return question_id, context[:max_chars]

    @staticmethod
    def extract_evidence(sample: Dict) -> str:
        """
        提取证据文本

        Args:
            sample: 样本字典

        Returns:
            证据文本（字符串格式）
        """
        evidence = sample.get('golden_evidence', '')
        # 如果是列表，转换为字符串
        if isinstance(evidence, list):
            return '\n'.join(evidence) if evidence else ''
        return evidence if evidence else ''

    @staticmethod
    def safe_filename(s: str) -> str:
        """
        生成安全的文件名

        Args:
            s: 原始字符串

        Returns:
            安全的文件名
        """
        import re
        # 替换不安全的字符为下划线
        safe = re.sub(r'[^\w\-_\.]', '_', s)
        # 限制长度
        return safe[:100]


class VTCLoader:
    """
    VTCBench数据集加载器
    专门处理VTCBench QA数据集，数据格式为JSONL，包含问题、上下文、答案等信息
    """

    def __init__(self, data_path: str):
        """
        初始化加载器

        Args:
            data_path: VTCBench 数据文件路径 (JSONL 格式)
        """
        self.data_path = data_path
        self.samples = []
        self._load_data()

    def _load_data(self):
        """加载 VTCBench 数据"""
        print(f"Loading VTCBench data from: {self.data_path}")

        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        item = json.loads(line.strip())
                        # 确保 question_id 存在
                        if 'question_id' not in item:
                            item['question_id'] = f"vtc_{line_num-1}"
                        # 确保 real_id 存在（用于唯一标识）
                        if 'real_id' not in item:
                            item['real_id'] = f"real_{line_num:04d}"
                        self.samples.append(item)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line {line_num}: {e}")
                        continue
            print(f"Successfully loaded {len(self.samples)} VTCBench samples.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        except Exception as e:
            raise Exception(f"Error loading data: {e}")

    def get_samples(self, limit: Optional[int] = None) -> List[Dict]:
        """
        获取样本列表

        Args:
            limit: 限制返回的样本数量

        Returns:
            样本列表
        """
        return self.samples[:limit] if limit else self.samples

    @staticmethod
    def get_question(sample: Dict) -> str:
        """
        从样本中提取问题文本

        Args:
            sample: 样本字典

        Returns:
            问题文本
        """
        return sample.get('question', '')

    @staticmethod
    def build_context(sample: Dict, max_chars: int) -> Tuple[str, str]:
        """
        构建上下文文本

        Args:
            sample: 样本字典
            max_chars: 最大字符数限制

        Returns:
            (real_id, context_text) 元组，使用real_id作为唯一标识符
        """
        real_id = VTCLoader.get_real_id(sample)

        # 直接使用 context 字段
        context = sample.get('context', '')

        return real_id, context[:max_chars]

    @staticmethod
    def extract_evidence(sample: Dict) -> str:
        """
        提取证据文本

        Args:
            sample: 样本字典

        Returns:
            证据文本（字符串格式）
        """
        evidence = sample.get('golden_evidence', '')
        return evidence if evidence else ''

    @staticmethod
    def get_real_id(sample: Dict) -> str:
        """
        获取样本的真实唯一标识符

        Args:
            sample: 样本字典

        Returns:
            real_id: 唯一标识符，如果没有real_id则返回question_id
        """
        return sample.get('real_id', sample.get('question_id', 'unknown'))

    @staticmethod
    def safe_filename(s: str) -> str:
        """
        生成安全的文件名

        Args:
            s: 原始字符串

        Returns:
            安全的文件名
        """
        import re
        # 替换不安全的字符为下划线
        safe = re.sub(r'[^\w\-_\.]', '_', s)
        # 限制长度
        return safe[:100]

class NIAHLoader:
    """
    NIAH (Needle-In-A-Haystack)数据集加载器
    专门处理NIAH内存测试数据集，数据格式为JSONL，按文本长度分类存储
    """

    def __init__(self, data_dir: str):
        """
        初始化加载器

        Args:
            data_dir: 数据集根目录路径
        """
        self.data_dir = data_dir
        self._length_folders = None
        self._samples_cache = {}

    def get_available_lengths(self) -> List[int]:
        """
        获取可用的文本长度列表

        Returns:
            排序的长度列表
        """
        if self._length_folders is None:
            all_folders = [f for f in os.listdir(self.data_dir)
                          if f.startswith('ruler_niah_single_1_len')]
            lengths = []
            for folder in all_folders:
                try:
                    length = int(folder.split('len')[-1])
                    lengths.append(length)
                except (ValueError, IndexError):
                    continue
            self._length_folders = sorted(lengths)

        return self._length_folders

    def load_data_by_lengths(self, lengths: Optional[List[int]] = None,
                           limit_per_length: Optional[int] = None) -> Tuple[List[Dict], Dict[str, int]]:
        """
        按长度加载数据集

        Args:
            lengths: 要加载的长度列表，如果为None则加载所有长度
            limit_per_length: 每个长度的最大样本数

        Returns:
            samples: 样本列表
            folder_info: 文件夹信息字典 {folder_name: sample_count}
        """
        samples = []
        folder_info = {}

        # 获取要处理的长度
        available_lengths = self.get_available_lengths()
        if lengths is None:
            target_lengths = available_lengths
        else:
            target_lengths = [l for l in lengths if l in available_lengths]
            if not target_lengths:
                print(f"Warning: None of the requested lengths {lengths} are available. "
                      f"Available: {available_lengths}")
                return samples, folder_info

        print(f"Loading data for lengths: {target_lengths}")

        for length in target_lengths:
            folder_name = f"ruler_niah_single_1_len{length}"
            folder_path = os.path.join(self.data_dir, folder_name)

            if not os.path.isdir(folder_path):
                print(f"Warning: Folder {folder_name} not found")
                continue

            # 加载该文件夹的数据
            folder_samples = self._load_single_folder(folder_path, folder_name, limit_per_length)
            samples.extend(folder_samples)
            folder_info[folder_name] = len(folder_samples)

            print(f"Loaded {len(folder_samples)} samples from {folder_name}")

        print(f"Total loaded {len(samples)} samples from {len(folder_info)} folders")
        return samples, folder_info

    def _load_single_folder(self, folder_path: str, folder_name: str,
                           limit: Optional[int] = None) -> List[Dict]:
        """
        加载单个文件夹的数据

        Args:
            folder_path: 文件夹路径
            folder_name: 文件夹名称
            limit: 最大样本数限制

        Returns:
            样本列表
        """
        # 查找jsonl文件
        jsonl_files = [f for f in os.listdir(folder_path) if f.endswith('.jsonl')]
        if not jsonl_files:
            print(f"Warning: No jsonl file found in {folder_path}")
            return []

        jsonl_file = os.path.join(folder_path, jsonl_files[0])

        # 检查缓存
        cache_key = jsonl_file
        if cache_key in self._samples_cache:
            samples = self._samples_cache[cache_key]
        else:
            samples = []
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            sample = json.loads(line.strip())
                            # 添加元数据
                            sample['folder'] = folder_name
                            sample['length_category'] = int(folder_name.split('len')[-1])
                            samples.append(sample)
                        except json.JSONDecodeError as e:
                            print(f"Warning: Failed to parse line {line_num} in {jsonl_file}: {e}")
                            continue

                # 缓存结果
                self._samples_cache[cache_key] = samples

            except Exception as e:
                print(f"Error loading {jsonl_file}: {e}")
                return []

        # 应用限制
        if limit and len(samples) > limit:
            samples = samples[:limit]

        return samples

    def get_samples(self, lengths: Optional[List[int]] = None,
                   limit_per_length: Optional[int] = None) -> List[Dict]:
        """
        获取样本列表

        Args:
            lengths: 要加载的长度列表
            limit_per_length: 每个长度的最大样本数

        Returns:
            样本列表
        """
        samples, _ = self.load_data_by_lengths(lengths, limit_per_length)
        return samples

    @staticmethod
    def get_question(sample: Dict) -> str:
        """
        从样本中提取问题文本

        Args:
            sample: 样本字典

        Returns:
            问题文本
        """
        doc = sample['doc']
        input_text = doc['input']

        # 分离问题和上下文
        # 通常问题在最后，以"What is..."开头
        lines = input_text.strip().split('\n')
        question_line = None

        # 找到问题行（通常是最后一行）
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip().startswith("What is the special magic number"):
                question_line = i
                break

        if question_line is not None:
            return lines[question_line]
        else:
            # 如果找不到问题行，返回最后一行
            return lines[-1] if lines else ""

    @staticmethod
    def build_context(sample: Dict, max_chars: int) -> Tuple[str, str]:
        """
        构建上下文文本

        Args:
            sample: 样本字典
            max_chars: 最大字符数限制

        Returns:
            (question_id, context_text) 元组
        """
        question_id = f"niah_{sample['doc_id']}"

        doc = sample['doc']
        input_text = doc['input']

        # 分离问题和上下文
        lines = input_text.strip().split('\n')
        question_line = None

        # 找到问题行
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip().startswith("What is the special magic number"):
                question_line = i
                break

        if question_line is not None:
            context = '\n'.join(lines[:question_line])
        else:
            # 如果找不到问题行，将整个文本作为上下文，最后一行作为问题
            context = '\n'.join(lines[:-1]) if len(lines) > 1 else ""

        # 处理 max_chars 参数
        if isinstance(max_chars, float) and max_chars == float('inf'):
            return question_id, context
        else:
            return question_id, context[:int(max_chars)]

    @staticmethod
    def extract_evidence(sample: Dict) -> List[str]:
        """
        提取正确答案列表

        Args:
            sample: 样本字典

        Returns:
            正确答案列表
        """
        doc = sample['doc']
        outputs = doc.get('outputs', [])
        if isinstance(outputs, list) and outputs:
            return [str(output) for output in outputs]
        return []

    @staticmethod
    def get_sample_info(sample: Dict) -> Dict[str, Any]:
        """
        获取样本的基本信息

        Args:
            sample: 样本字典

        Returns:
            包含关键信息的字典
        """
        doc = sample['doc']
        question = NIAHLoader.get_question(sample)
        _, context = NIAHLoader.build_context(sample, float('inf'))
        answers = NIAHLoader.extract_evidence(sample)

        return {
            'doc_id': sample['doc_id'],
            'folder': sample.get('folder', ''),
            'length_category': sample.get('length_category', 0),
            'target_answer': answers[0] if answers else "",
            'question': question,
            'context_length': len(context),
            'has_target': bool(answers)
        }

    @staticmethod
    def safe_filename(s: str) -> str:
        """
        生成安全的文件名

        Args:
            s: 原始字符串

        Returns:
            安全的文件名
        """
        import re
        # 替换不安全的字符为下划线
        safe = re.sub(r'[^\w\-_\.]', '_', s)
        # 限制长度
        return safe[:100]
