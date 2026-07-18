#!/usr/bin/env python3
"""
重排与评估主程序

功能：
1. 加载检索结果和 Gold 标注
2. 根据配置文件应用不同的重排策略
3. 计算重排前后的评估指标
4. 生成对比报告和可视化
"""

import json
import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 移除logger_utils依赖

# 设置日志
def setup_logging(config: dict):
    """设置日志配置"""
    log_level = getattr(logging, config.get('logging', {}).get('level', 'INFO'))
    log_file = config.get('logging', {}).get('log_file', 'results/rerank_evaluation.log')
    
    # 创建日志目录
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


class BaseReranker:
    """重排器基类"""

    def __init__(self, name: str, normalize_scores: bool = True):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.normalize_scores = normalize_scores
        self.logger.info(f"分数归一化功能: {'已启用' if normalize_scores else '已禁用'}")

    def normalize_score(self, score: float) -> float:
        """
        将分数归一化到0~1范围
        子类可以根据自身分数特点重写此方法

        Args:
            score: 原始分数

        Returns:
            归一化后的分数（0~1）
        """
        if not self.normalize_scores:
            return score

        # 默认实现：如果分数已经在0~1范围内直接返回，否则用sigmoid转换
        if 0 <= score <= 1:
            return score
        else:
            # sigmoid函数将任意实数映射到0~1
            import numpy as np
            return 1 / (1 + np.exp(-score))

    def rerank(self, query: str, docs: List[Dict], top_k: int) -> List[Dict]:
        """
        重排文档

        Args:
            query: 查询文本
            docs: 候选文档列表（包含 doc_id, text, original_rank, original_score）
            top_k: 返回前 K 个结果
        Returns:
            重排后的文档列表
        """
        raise NotImplementedError

    def cleanup(self):
        """清理资源"""
        pass


class NoReranker(BaseReranker):
    """无重排（基线）"""

    def __init__(self, normalize_scores: bool = True):
        super().__init__("NoReranker", normalize_scores)

    def rerank(self, query: str, docs: List[Dict], top_k: int) -> List[Dict]:
        """直接返回原始排序的前K个文档，保留原始分数作为rerank_score"""
        # 为每个文档添加rerank_score，使用原始分数
        scored_docs = []
        for doc in docs[:top_k]:
            original_score = doc.get('original_score', 1.0 / doc.get('original_rank', 1))
            # NoReranker的分数已经是检索器输出的归一化结果，直接使用
            normalized_score = float(original_score) if self.normalize_scores else float(original_score)
            scored_docs.append({
                **doc,
                'rerank_score': float(original_score),
                'rerank_score_normalized': normalized_score
            })

        return scored_docs


class RougeReranker(BaseReranker):
    """ROUGE-L 重排器"""

    def __init__(self, normalize_scores: bool = True):
        super().__init__("RougeReranker", normalize_scores)
        try:
            from rouge_score import rouge_scorer
            self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            self.logger.info("ROUGE-L scorer initialized successfully")
        except ImportError as e:
            self.logger.error(f"Failed to import rouge_score: {e}")
            self.logger.error("Please install: pip install rouge-score")
            raise

    def rerank(self, query: str, docs: List[Dict], top_k: int) -> List[Dict]:
        """使用ROUGE-L分数重排文档"""
        scored_docs = []
        for doc in docs:
            try:
                score = self.scorer.score(query, doc['text'])['rougeL'].fmeasure
                # ROUGE分数已经在0~1范围，直接归一化
                normalized_score = self.normalize_score(score)
                scored_docs.append({
                    **doc,
                    'rerank_score': float(score),
                    'rerank_score_normalized': float(normalized_score)
                })
            except Exception as e:
                self.logger.warning(f"ROUGE scoring failed for doc {doc['doc_id']}: {e}")
                scored_docs.append({
                    **doc,
                    'rerank_score': 0.0,
                    'rerank_score_normalized': 0.0
                })

        return sorted(scored_docs, key=lambda x: x['rerank_score'], reverse=True)[:top_k]


class NLIReranker(BaseReranker):
    """NLI 模型重排器"""

    def __init__(self, model_name: str, batch_size: int = 16, device: str = 'cuda', entailment_idx: int = None, normalize_scores: bool = True):
        super().__init__("NLIReranker", normalize_scores)
        self.batch_size = batch_size
        self.device = device
        self.entailment_idx = entailment_idx

        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            self.logger.info(f"Loading NLI model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)

            # 检查设备可用性
            if device == 'cuda' and not torch.cuda.is_available():
                self.logger.warning("CUDA not available, falling back to CPU")
                self.device = 'cpu'

            self.model.to(self.device)
            self.model.eval()

            # 确定 entailment 标签的索引
            if self.entailment_idx is None:
                id2label = self.model.config.id2label
                entailment_labels = [i for i, label in id2label.items() if 'entail' in label.lower()]
                if entailment_labels:
                    self.entailment_idx = entailment_labels[0]
                else:
                    # 默认假设是标准MNLI顺序：contradiction(0), neutral(1), entailment(2)
                    self.entailment_idx = 2
                    self.logger.warning(f"Could not find entailment label, using index {self.entailment_idx}")
            else:
                self.logger.info(f"Using configured entailment index: {self.entailment_idx}")

            # 验证标签索引有效性
            num_labels = self.model.config.num_labels
            if self.entailment_idx < 0 or self.entailment_idx >= num_labels:
                raise ValueError(f"Entailment index {self.entailment_idx} is out of range for model with {num_labels} labels")

            self.logger.info(f"NLI model loaded successfully. Entailment index: {self.entailment_idx}")
            
        except ImportError as e:
            self.logger.error(f"Failed to import transformers: {e}")
            self.logger.error("Please install: pip install transformers torch")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load NLI model: {e}")
            raise
    
    def rerank(self, query: str, docs: List[Dict], top_k: int) -> List[Dict]:
        """使用NLI模型重排文档"""
        import torch
        
        scored_docs = []
        
        try:
            for i in range(0, len(docs), self.batch_size):
                batch_docs = docs[i:i+self.batch_size]
                
                # 构造输入：query [SEP] doc
                inputs = self.tokenizer(
                    [query] * len(batch_docs),
                    [doc['text'][:512] for doc in batch_docs],  # 截断到512
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=1)
                    scores = probs[:, self.entailment_idx].cpu().numpy()
                
                for doc, score in zip(batch_docs, scores):
                    # NLI分数已经是概率（0~1），直接归一化
                    normalized_score = self.normalize_score(float(score))
                    scored_docs.append({
                        **doc,
                        'rerank_score': float(score),
                        'rerank_score_normalized': float(normalized_score)
                    })
                    
        except Exception as e:
            self.logger.error(f"NLI reranking failed: {e}")
            # 降级到原始排序
            for doc in docs:
                fallback_score = 1.0 / doc['original_rank']
                normalized_score = self.normalize_score(fallback_score)
                scored_docs.append({
                    **doc,
                    'rerank_score': float(fallback_score),
                    'rerank_score_normalized': float(normalized_score)
                })
        
        return sorted(scored_docs, key=lambda x: x['rerank_score'], reverse=True)[:top_k]
    
    def cleanup(self):
        """清理GPU内存"""
        try:
            import torch
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'tokenizer'):
                del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.logger.info("NLI model resources cleaned up")
        except:
            pass


class CrossEncoderReranker(BaseReranker):
    """Cross-Encoder 重排器"""

    def __init__(self, model_name: str, batch_size: int = 16, device: str = 'cuda', normalize_scores: bool = True):
        super().__init__("CrossEncoderReranker", normalize_scores)
        self.batch_size = batch_size
        
        try:
            from sentence_transformers import CrossEncoder
            import torch
            
            # 检查设备可用性
            if device == 'cuda' and not torch.cuda.is_available():
                self.logger.warning("CUDA not available, falling back to CPU")
                device = 'cpu'
            
            self.logger.info(f"Loading Cross-Encoder model: {model_name}")
            self.model = CrossEncoder(model_name, max_length=512, device=device, local_files_only=True)
            self.logger.info("Cross-Encoder model loaded successfully")
            
        except ImportError as e:
            self.logger.error(f"Failed to import sentence_transformers: {e}")
            self.logger.error("Please install: pip install sentence-transformers")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load Cross-Encoder model: {e}")
            raise
    
    def rerank(self, query: str, docs: List[Dict], top_k: int) -> List[Dict]:
        """使用Cross-Encoder重排文档"""
        try:
            # 构造查询-文档对
            pairs = [[query, doc['text'][:512]] for doc in docs]

            # 批量计算分数
            scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)

            # 添加分数并排序
            scored_docs = []
            for doc, score in zip(docs, scores):
                # CrossEncoder输出是logits，需要归一化
                normalized_score = self.normalize_score(float(score))
                scored_docs.append({
                    **doc,
                    'rerank_score': float(score),
                    'rerank_score_normalized': float(normalized_score)
                })
                
        except Exception as e:
            self.logger.error(f"Cross-Encoder reranking failed: {e}")
            # 降级到原始排序
            scored_docs = []
            for doc in docs:
                fallback_score = 1.0 / doc['original_rank']
                normalized_score = self.normalize_score(fallback_score)
                scored_docs.append({
                    **doc,
                    'rerank_score': float(fallback_score),
                    'rerank_score_normalized': float(normalized_score)
                })
        
        return sorted(scored_docs, key=lambda x: x['rerank_score'], reverse=True)[:top_k]
    
    def cleanup(self):
        """清理资源"""
        try:
            import torch
            if hasattr(self, 'model'):
                del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.logger.info("Cross-Encoder model resources cleaned up")
        except:
            pass


class MSMarcoCrossEncoderReranker(BaseReranker):
    """MS-MARCO Cross-Encoder 重排器"""

    def __init__(self, model_name: str, batch_size: int = 4, device: str = 'cuda',
                 max_query_length: int = 256, max_doc_length: int = 256,
                 max_total_length: int = 512, use_fp16: bool = True,
                 score_type: str = 'auto', normalize_scores: bool = True):
        super().__init__("MSMarcoCrossEncoderReranker", normalize_scores)
        self.batch_size = batch_size
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length
        self.max_total_length = max_total_length
        self.use_fp16 = use_fp16
        self.score_type = score_type

        # 支持的分数处理类型
        self.valid_score_types = ['auto', 'raw', 'sigmoid', 'softmax']
        if self.score_type not in self.valid_score_types:
            raise ValueError(f"Invalid score_type: {score_type}. Valid types: {self.valid_score_types}")
        
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            
            # 检查设备可用性
            if device == 'cuda' and not torch.cuda.is_available():
                self.logger.warning("CUDA not available, falling back to CPU")
                device = 'cpu'
                self.use_fp16 = False  # CPU不支持fp16
            
            self.device = device
            
            self.logger.info(f"Loading MS-MARCO Cross-Encoder model: {model_name}")
            self.logger.info(f"Device: {device}, Batch size: {batch_size}, FP16: {use_fp16}")
            
            # 加载tokenizer和model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
            
            # 移动到指定设备
            self.model.to(self.device)
            self.model.eval()
            
            # 启用fp16推理以节省显存
            if self.use_fp16 and device == 'cuda':
                self.model.half()
                self.logger.info("Enabled FP16 inference for memory optimization")
            
            # 自动检测分数处理方式
            if self.score_type == 'auto':
                # 根据常见模型名称自动匹配处理方式
                model_name_lower = model_name.lower()
                if 'cross-encoder/ms-marco-' in model_name_lower and ('-v2' in model_name_lower or '-v3' in model_name_lower):
                    # 官方cross-encoder模型输出raw分数，不需要sigmoid
                    self.score_type = 'raw'
                    self.logger.info(f"Auto-detected score_type: {self.score_type} (official cross-encoder model)")
                elif 'bge-reranker' in model_name_lower:
                    # BGE reranker输出raw分数
                    self.score_type = 'raw'
                    self.logger.info(f"Auto-detected score_type: {self.score_type} (BGE reranker model)")
                else:
                    # 默认使用sigmoid处理单输出，softmax处理多输出
                    self.score_type = 'sigmoid' if self.model.config.num_labels == 1 else 'softmax'
                    self.logger.info(f"Auto-detected score_type: {self.score_type} ({self.model.config.num_labels} labels)")

            self.logger.info("MS-MARCO Cross-Encoder model loaded successfully")
            
        except ImportError as e:
            self.logger.error(f"Failed to import transformers: {e}")
            self.logger.error("Please install: pip install transformers torch")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load MS-MARCO Cross-Encoder model: {e}")
            raise
    
    def _preprocess_text(self, query: str, doc: str) -> Tuple[str, str]:
        """
        预处理查询和文档文本，确保长度符合要求
        
        Args:
            query: 原始查询文本
            doc: 原始文档文本
            
        Returns:
            处理后的(query, doc)元组
        """
        # 先对query和doc分别进行初步截断
        query_tokens = self.tokenizer.tokenize(query)[:self.max_query_length]
        doc_tokens = self.tokenizer.tokenize(doc)[:self.max_doc_length]
        
        # 重新组合文本
        truncated_query = self.tokenizer.convert_tokens_to_string(query_tokens)
        truncated_doc = self.tokenizer.convert_tokens_to_string(doc_tokens)
        
        # 检查总长度是否超限
        test_encoding = self.tokenizer(
            truncated_query, 
            truncated_doc, 
            add_special_tokens=True,
            return_tensors='pt'
        )
        
        total_length = test_encoding['input_ids'].shape[1]
        
        # 如果总长度仍然超限，进一步截断文档
        if total_length > self.max_total_length:
            # 计算可用于文档的最大token数
            query_length = len(self.tokenizer.tokenize(truncated_query))
            special_tokens_length = 3  # [CLS], [SEP], [SEP]
            available_doc_length = self.max_total_length - query_length - special_tokens_length
            
            if available_doc_length > 0:
                doc_tokens = doc_tokens[:available_doc_length]
                truncated_doc = self.tokenizer.convert_tokens_to_string(doc_tokens)
            else:
                # 如果query太长，也需要截断query
                available_query_length = self.max_total_length - special_tokens_length - 50  # 为doc保留至少50个token
                query_tokens = query_tokens[:available_query_length]
                truncated_query = self.tokenizer.convert_tokens_to_string(query_tokens)
                
                doc_tokens = doc_tokens[:50]
                truncated_doc = self.tokenizer.convert_tokens_to_string(doc_tokens)
        
        return truncated_query, truncated_doc
    
    def rerank(self, query: str, docs: List[Dict], top_k: int) -> List[Dict]:
        """使用MS-MARCO Cross-Encoder重排文档"""
        import torch
        
        scored_docs = []
        
        try:
            # 分批处理以避免显存溢出
            for i in range(0, len(docs), self.batch_size):
                batch_docs = docs[i:i+self.batch_size]
                
                # 预处理文本对
                processed_pairs = []
                for doc in batch_docs:
                    try:
                        proc_query, proc_doc = self._preprocess_text(query, doc['text'])
                        processed_pairs.append((proc_query, proc_doc))
                    except Exception as e:
                        self.logger.warning(f"Text preprocessing failed for doc {doc['doc_id']}: {e}")
                        # 使用简单截断作为降级策略
                        processed_pairs.append((query[:256], doc['text'][:256]))
                
                # 批量编码
                try:
                    batch_queries = [pair[0] for pair in processed_pairs]
                    batch_texts = [pair[1] for pair in processed_pairs]
                    
                    inputs = self.tokenizer(
                        batch_queries,
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=self.max_total_length,
                        return_tensors='pt'
                    ).to(self.device)
                    
                    # 推理
                    with torch.no_grad():
                        if self.use_fp16 and self.device == 'cuda':
                            with torch.autocast(device_type='cuda'):
                                outputs = self.model(**inputs)
                        else:
                            outputs = self.model(**inputs)
                        
                        # 获取相关性分数
                        logits = outputs.logits

                        # 根据配置的分数类型处理
                        if self.score_type == 'raw':
                            # 直接使用原始分数
                            if logits.shape[1] == 1:
                                scores = logits.squeeze(-1)
                            else:
                                # 多分类时默认取正类分数
                                scores = logits[:, 1]
                        elif self.score_type == 'sigmoid':
                            # 应用sigmoid
                            scores = torch.sigmoid(logits.squeeze(-1))
                        elif self.score_type == 'softmax':
                            # 应用softmax并取正类
                            scores = torch.softmax(logits, dim=-1)[:, 1]
                        else:
                            # 降级处理
                            self.logger.warning(f"Unknown score_type: {self.score_type}, using raw scores")
                            if logits.shape[1] == 1:
                                scores = logits.squeeze(-1)
                            else:
                                scores = logits[:, 1]
                        
                        scores = scores.cpu().numpy()
                    
                    # 添加分数到文档（使用原始的batch_docs而不是batch_texts）
                    for doc, score in zip(batch_docs, scores):
                        normalized_score = self.normalize_score(float(score))
                        scored_docs.append({
                            **doc,
                            'rerank_score': float(score),
                            'rerank_score_normalized': float(normalized_score)
                        })
                    
                    # 清理批次缓存
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    self.logger.warning(f"Batch processing failed: {e}")
                    # 降级到原始排序
                    for doc in batch_docs:
                        fallback_score = 1.0 / doc['original_rank']
                        normalized_score = self.normalize_score(fallback_score)
                        scored_docs.append({
                            **doc,
                            'rerank_score': float(fallback_score),
                            'rerank_score_normalized': float(normalized_score)
                        })
                    
        except Exception as e:
            self.logger.error(f"MS-MARCO Cross-Encoder reranking failed: {e}")
            # 完全降级到原始排序
            for doc in docs:
                fallback_score = 1.0 / doc['original_rank']
                normalized_score = self.normalize_score(fallback_score)
                scored_docs.append({
                    **doc,
                    'rerank_score': float(fallback_score),
                    'rerank_score_normalized': float(normalized_score)
                })
        
        # 按分数排序并返回top_k
        sorted_docs = sorted(scored_docs, key=lambda x: x['rerank_score'], reverse=True)
        return sorted_docs[:top_k]
    
    def cleanup(self):
        """清理GPU资源"""
        try:
            import torch
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'tokenizer'):
                del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.logger.info("MS-MARCO Cross-Encoder model resources cleaned up")
        except Exception as e:
            self.logger.warning(f"Cleanup warning: {e}")


class MonoT5Reranker(BaseReranker):
    """
    MonoT5重排器 - T5模型用于passage ranking
    
    MonoT5的特点：
    1. 输入格式：Query: <query> Document: <doc> Relevant:
    2. 输出：生成'true'或'false'，通过logits计算relevance score
    3. 比Cross-Encoder更强，但推理速度更慢
    
    参考论文：Nogueira et al. 2020 - Document Ranking with a Pretrained 
              Sequence-to-Sequence Model
    """
    
    def __init__(self,
                 model_name: str,
                 batch_size: int = 4,
                 device: str = "cuda",
                 max_query_length: int = 256,
                 max_doc_length: int = 256,
                 max_total_length: int = 512,
                 use_fp16: bool = False,
                 normalize_scores: bool = True):
        """
        初始化MonoT5重排器

        Args:
            model_name: 模型路径或名称
            batch_size: 批处理大小
            device: 设备类型
            max_query_length: 查询最大长度
            max_doc_length: 文档最大长度
            max_total_length: 总最大长度
            use_fp16: 是否使用半精度
            normalize_scores: 是否将分数归一化到0~1范围
        """
        super().__init__("MonoT5Reranker", normalize_scores)
        
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length
        self.max_total_length = max_total_length
        self.use_fp16 = use_fp16
        
        self.logger.info(f"初始化MonoT5重排器: {model_name}")
        self.logger.info(f"设备: {device}, 批大小: {batch_size}")
        
        try:
            from transformers import T5Tokenizer, T5ForConditionalGeneration
            import torch
            
            # 加载tokenizer和模型
            self.logger.info(f"加载MonoT5模型: {model_name}")
            
            # 尝试加载本地模型，如果失败则使用在线模型
            try:
                self.tokenizer = T5Tokenizer.from_pretrained(model_name, local_files_only=True)
                self.model = T5ForConditionalGeneration.from_pretrained(model_name, local_files_only=True)
                self.logger.info("成功加载本地MonoT5模型")
            except Exception as local_e:
                self.logger.warning(f"本地模型加载失败: {local_e}")
                self.logger.info("尝试从HuggingFace Hub加载模型...")
                # 使用在线模型名称
                online_model_name = "castorini/monot5-base-msmarco-10k"
                self.tokenizer = T5Tokenizer.from_pretrained(online_model_name)
                self.model = T5ForConditionalGeneration.from_pretrained(online_model_name)
                self.logger.info("成功加载在线MonoT5模型")
            
            # 获取'true'和'false'的token IDs
            self.token_false_id = self.tokenizer.encode('false')[0]
            self.token_true_id = self.tokenizer.encode('true')[0]
            self.logger.info(f"Token IDs - true: {self.token_true_id}, false: {self.token_false_id}")
            
            # 设置模型为评估模式并移动到设备
            self.model.eval()
            self.model.to(device)
            
            # 如果使用FP16
            if use_fp16 and device == 'cuda':
                self.model.half()
                self.logger.info("启用FP16推理")
            
            self.logger.info("MonoT5模型加载成功")
            
        except Exception as e:
            self.logger.error(f"MonoT5模型加载失败: {e}")
            raise
    
    def _compute_relevance_score(self, logits, token_true_id, token_false_id):
        """
        从模型输出的logits计算relevance score
        
        公式：P(relevant) = exp(logit_true) / (exp(logit_true) + exp(logit_false))
        
        Args:
            logits: 模型输出的logits
            token_true_id: 'true'的token ID
            token_false_id: 'false'的token ID
            
        Returns:
            0-1之间的分数，越高越相关
        """
        import torch
        
        # 获取'true'和'false'对应的logits
        logit_true = logits[:, token_true_id]
        logit_false = logits[:, token_false_id]
        
        # 计算概率：P(true) = exp(logit_true) / (exp(logit_true) + exp(logit_false))
        # 使用log-sum-exp技巧避免数值不稳定
        max_logit = torch.max(logit_true, logit_false)
        exp_true = torch.exp(logit_true - max_logit)
        exp_false = torch.exp(logit_false - max_logit)
        
        prob_true = exp_true / (exp_true + exp_false)
        
        return prob_true.cpu().numpy()
    
    def _preprocess_text(self, query: str, doc: str) -> str:
        """
        将query和doc格式化为MonoT5的输入格式
        
        格式：Query: {query} Document: {doc} Relevant:
        
        Args:
            query: 查询文本
            doc: 文档文本
            
        Returns:
            格式化后的输入文本
        """
        # 截断query和doc到指定长度
        query_tokens = self.tokenizer.encode(query, add_special_tokens=False)
        doc_tokens = self.tokenizer.encode(doc, add_special_tokens=False)
        
        # 计算前缀长度："Query: " + " Document: " + " Relevant:"
        prefix_tokens = self.tokenizer.encode("Query:  Document:  Relevant:", add_special_tokens=False)
        prefix_length = len(prefix_tokens)
        
        # 计算可用长度
        available_length = self.max_total_length - prefix_length - 2  # 减去特殊token
        
        # 按比例分配query和doc的长度
        query_max = min(self.max_query_length, len(query_tokens))
        doc_max = min(self.max_doc_length, len(doc_tokens))
        
        # 如果总长度超出限制，按比例缩减
        if query_max + doc_max > available_length:
            ratio = available_length / (query_max + doc_max)
            query_max = int(query_max * ratio)
            doc_max = available_length - query_max
        
        # 截断并解码
        query_truncated = self.tokenizer.decode(query_tokens[:query_max], skip_special_tokens=True)
        doc_truncated = self.tokenizer.decode(doc_tokens[:doc_max], skip_special_tokens=True)
        
        # 格式化为MonoT5输入格式
        formatted_text = f"Query: {query_truncated} Document: {doc_truncated} Relevant:"
        
        return formatted_text
    
    def rerank(self, query: str, docs: List[Dict], top_k: int) -> List[Dict]:
        """
        使用MonoT5重排文档
        
        Args:
            query: 查询文本
            docs: 候选文档列表
            top_k: 返回前K个结果
            
        Returns:
            重排后的文档列表
        """
        import torch
        from tqdm import tqdm
        
        if not docs:
            return []
        
        self.logger.info(f"MonoT5重排开始: {len(docs)}个文档")
        
        scored_docs = []
        
        try:
            # 批处理重排
            for i in tqdm(range(0, len(docs), self.batch_size), 
                         desc=f"MonoT5重排 (batch_size={self.batch_size})"):
                batch_docs = docs[i:i + self.batch_size]
                
                batch_processed = False  # 标记批次是否已处理
                
                try:
                    # 预处理文本
                    batch_texts = []
                    for doc in batch_docs:
                        formatted_text = self._preprocess_text(query, doc.get('text', ''))
                        batch_texts.append(formatted_text)
                    
                    # Tokenize
                    inputs = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=self.max_total_length,
                        return_tensors='pt'
                    ).to(self.device)

                    # 推理：使用forward直接获取第一个token的logits，比generate效率更高
                    # MonoT5是encoder-decoder模型，需要构造decoder_input_ids
                    decoder_input_ids = torch.tensor([[self.model.config.decoder_start_token_id]] * len(batch_texts)).to(self.device)

                    with torch.no_grad():
                        if self.use_fp16 and self.device == 'cuda':
                            with torch.autocast(device_type='cuda'):
                                outputs = self.model(
                                    input_ids=inputs['input_ids'],
                                    attention_mask=inputs['attention_mask'],
                                    decoder_input_ids=decoder_input_ids
                                )
                        else:
                            outputs = self.model(
                                input_ids=inputs['input_ids'],
                                attention_mask=inputs['attention_mask'],
                                decoder_input_ids=decoder_input_ids
                            )

                    # 获取第一个decoder位置的logits
                    logits = outputs.logits[:, 0, :]  # [batch_size, vocab_size]

                    # 计算相关性分数
                    relevance_scores = self._compute_relevance_score(
                        logits, self.token_true_id, self.token_false_id
                    )
                    
                    # 添加分数到文档
                    for doc, score in zip(batch_docs, relevance_scores):
                        # MonoT5分数已经是概率（0~1），直接归一化
                        normalized_score = self.normalize_score(float(score))
                        scored_docs.append({
                            **doc,
                            'rerank_score': float(score),
                            'rerank_score_normalized': float(normalized_score)
                        })
                    
                    batch_processed = True  # 标记批次已成功处理
                
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and not batch_processed:
                        self.logger.warning(f"批处理OOM错误，降级处理: {e}")
                        # 逐个处理文档
                        for doc in batch_docs:
                            try:
                                formatted_text = self._preprocess_text(query, doc.get('text', ''))
                                inputs = self.tokenizer(
                                    formatted_text,
                                    return_tensors='pt',
                                    truncation=True,
                                    max_length=self.max_total_length
                                ).to(self.device)
                                
                                with torch.no_grad():
                                    # 使用forward方式处理单样本
                                    decoder_input_ids = torch.tensor([[self.model.config.decoder_start_token_id]]).to(self.device)
                                    outputs = self.model(
                                        input_ids=inputs['input_ids'],
                                        attention_mask=inputs['attention_mask'],
                                        decoder_input_ids=decoder_input_ids
                                    )

                                    logits = outputs.logits[:, 0, :]
                                    score = self._compute_relevance_score(
                                        logits, self.token_true_id, self.token_false_id
                                    )[0]
                                
                                normalized_score = self.normalize_score(float(score))
                                scored_docs.append({
                                    **doc,
                                    'rerank_score': float(score),
                                    'rerank_score_normalized': float(normalized_score)
                                })
                                
                            except Exception as single_e:
                                self.logger.error(f"单文档处理失败: {single_e}")
                                # 使用原始分数作为fallback
                                fallback_score = doc.get('original_score', 0.0)
                                normalized_score = self.normalize_score(fallback_score)
                                scored_docs.append({
                                    **doc,
                                    'rerank_score': float(fallback_score),
                                    'rerank_score_normalized': float(normalized_score)
                                })
                        batch_processed = True  # 标记降级处理已完成
                    elif not batch_processed:
                        self.logger.error(f"批处理推理失败: {e}")
                        # 使用原始分数作为fallback
                        for doc in batch_docs:
                            fallback_score = doc.get('original_score', 0.0)
                            normalized_score = self.normalize_score(fallback_score)
                            scored_docs.append({
                                **doc,
                                'rerank_score': float(fallback_score),
                                'rerank_score_normalized': float(normalized_score)
                            })
                        batch_processed = True
                
                except Exception as e:
                    if not batch_processed:
                        self.logger.error(f"批处理失败: {e}")
                        # 使用原始分数作为fallback
                        for doc in batch_docs:
                            fallback_score = doc.get('original_score', 0.0)
                            normalized_score = self.normalize_score(fallback_score)
                            scored_docs.append({
                                **doc,
                                'rerank_score': float(fallback_score),
                                'rerank_score_normalized': float(normalized_score)
                            })
                        batch_processed = True
        
        except Exception as e:
            self.logger.error(f"MonoT5重排失败: {e}")
            # 返回原始文档
            return docs[:top_k]
        
        # 按重排分数降序排序
        scored_docs.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        self.logger.info(f"MonoT5重排完成，返回前{top_k}个结果")
        
        return scored_docs[:top_k]
    
    def cleanup(self):
        """
        清理GPU资源
        """
        try:
            import torch
            
            self.logger.info("清理MonoT5模型资源")
            
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'tokenizer'):
                del self.tokenizer
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            self.logger.info("MonoT5资源清理完成")
            
        except Exception as e:
            self.logger.warning(f"MonoT5资源清理警告: {e}")


class RerankerFactory:
    """重排器工厂类"""
    
    @staticmethod
    def create(reranker_name: str, config: dict, logger):
        """根据名称创建重排器实例"""
        try:
            if reranker_name == "no_rerank":
                return NoReranker(
                    normalize_scores=config.get('normalize_scores', True)
                )
            elif reranker_name == "rouge_l":
                return RougeReranker(
                    normalize_scores=config.get('normalize_scores', True)
                )
            elif reranker_name == "nli":
                return NLIReranker(
                    model_name=config.get('model_name', 'microsoft/deberta-v3-base'),
                    batch_size=config.get('batch_size', 16),
                    device=config.get('device', 'cuda'),
                    entailment_idx=config.get('entailment_idx'),
                    normalize_scores=config.get('normalize_scores', True)
                )
            elif reranker_name == "cross_encoder":
                return CrossEncoderReranker(
                    model_name=config.get('model_name', 'BAAI/bge-reranker-base'),
                    batch_size=config.get('batch_size', 16),
                    device=config.get('device', 'cuda'),
                    normalize_scores=config.get('normalize_scores', True)
                )
            elif reranker_name == "ms_marco_l12":
                return MSMarcoCrossEncoderReranker(
                    model_name=config.get('model_name', 'cross-encoder/ms-marco-MiniLM-L-12-v2'),
                    batch_size=config.get('batch_size', 4),
                    device=config.get('device', 'cuda'),
                    max_query_length=config.get('max_query_length', 256),
                    max_doc_length=config.get('max_doc_length', 256),
                    max_total_length=config.get('max_total_length', 512),
                    use_fp16=config.get('use_fp16', True),
                    score_type=config.get('score_type', 'auto'),
                    normalize_scores=config.get('normalize_scores', True)
                )
            elif reranker_name == "monot5_base":
                return MonoT5Reranker(
                    model_name=config.get('model_name', 'castorini/monot5-base-msmarco-10k'),
                    batch_size=config.get('batch_size', 4),
                    device=config.get('device', 'cuda'),
                    max_query_length=config.get('max_query_length', 256),
                    max_doc_length=config.get('max_doc_length', 256),
                    max_total_length=config.get('max_total_length', 512),
                    use_fp16=config.get('use_fp16', False),
                    normalize_scores=config.get('normalize_scores', True)
                )
            else:
                raise ValueError(f"Unknown reranker: {reranker_name}")
        except Exception as e:
            logger.error(f"Failed to create reranker {reranker_name}: {e}")
            raise


def load_corpus(corpus_file: str, logger) -> Dict[str, str]:
    """加载语料库"""
    logger.info(f"Loading corpus from {corpus_file}...")
    corpus = {}
    
    if not Path(corpus_file).exists():
        logger.error(f"Corpus file not found: {corpus_file}")
        return corpus
    
    try:
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(tqdm(f, desc="Loading corpus"), 1):
                if line.strip():
                    try:
                        doc = json.loads(line)
                        # 兼容不同的字段名称
                        doc_id = doc.get('doc_id') or doc.get('id')
                        doc_text = doc.get('text') or doc.get('contents')
                        
                        if doc_id and doc_text:
                            corpus[doc_id] = doc_text
                        else:
                            logger.warning(f"Missing required fields at line {line_num}: available keys = {list(doc.keys())}")
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON decode error at line {line_num}: {e}")
                        continue
                    except KeyError as e:
                        logger.warning(f"Missing key at line {line_num}: {e}")
                        continue
        
        logger.info(f"Loaded {len(corpus):,} documents")
    except Exception as e:
        logger.error(f"Failed to load corpus: {e}")
    
    return corpus


def load_gold_annotations(gold_file: str, logger) -> Dict[str, List[str]]:
    """加载 Gold 标注"""
    logger.info(f"Loading gold annotations from {gold_file}...")
    gold_docs = {}
    
    if not Path(gold_file).exists():
        logger.error(f"Gold file not found: {gold_file}")
        return gold_docs
    
    try:
        df = pd.read_csv(gold_file, sep='\t')
        logger.info(f"Gold file columns: {df.columns.tolist()}")
        
        for _, row in df.iterrows():
            if row['Relevance'] == 1:
                qid = str(row['flashrag_qid'])
                doc_id = str(row['wiki_100w_id'])
                if qid not in gold_docs:
                    gold_docs[qid] = []
                gold_docs[qid].append(doc_id)
        
        logger.info(f"Loaded gold docs for {len(gold_docs)} queries")
        
        # 显示统计信息
        total_relevant = sum(len(docs) for docs in gold_docs.values())
        logger.info(f"Total relevant documents: {total_relevant}")
        
    except Exception as e:
        logger.error(f"Failed to load gold annotations: {e}")
    
    return gold_docs


def evaluate(retrieved_docs: List[str], gold_docs: List[str], k: int) -> Dict[str, float]:
    """
    计算评估指标
    
    Args:
        retrieved_docs: 检索到的文档 ID 列表（已排序）
        gold_docs: 相关文档 ID 列表
        k: 评估 top-k 结果
    Returns:
        指标字典
    """
    if not gold_docs:
        return {'recall': 0.0, 'precision': 0.0, 'mrr': 0.0, 'ndcg': 0.0}
    
    top_k_docs = retrieved_docs[:k]
    gold_set = set(gold_docs)
    
    # Recall@K
    recalled = len([doc for doc in top_k_docs if doc in gold_set])
    recall = recalled / len(gold_set)
    
    # Precision@K
    precision = recalled / k if k > 0 else 0.0
    
    # MRR
    mrr = 0.0
    for i, doc_id in enumerate(top_k_docs, 1):
        if doc_id in gold_set:
            mrr = 1.0 / i
            break
    
    # NDCG@K (简化版本)
    dcg = 0.0
    for i, doc_id in enumerate(top_k_docs, 1):
        if doc_id in gold_set:
            dcg += 1.0 / np.log2(i + 1)
    
    # 理想DCG
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gold_set), k)))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    
    return {
        'recall': recall,
        'precision': precision,
        'mrr': mrr,
        'ndcg': ndcg
    }


def evaluate_retrieval(retrieved_docs: List[str], gold_docs: List[str], eval_config: Dict) -> Dict[str, float]:
    """
    评估检索结果
    
    Args:
        retrieved_docs: 检索到的文档ID列表
        gold_docs: 相关文档ID列表
        eval_config: 评估配置
    
    Returns:
        评估指标字典
    """
    results = {}
    
    # Recall@K
    for k in eval_config.get('recall_k', [1, 5, 10, 20, 50]):
        metrics = evaluate(retrieved_docs, gold_docs, k)
        results[f'recall@{k}'] = metrics['recall']
    
    # Precision@K
    for k in eval_config.get('precision_k', [1, 5, 10, 20, 50]):
        metrics = evaluate(retrieved_docs, gold_docs, k)
        results[f'precision@{k}'] = metrics['precision']
    
    # NDCG@K
    for k in eval_config.get('ndcg_k', [1, 5, 10, 20, 50]):
        metrics = evaluate(retrieved_docs, gold_docs, k)
        results[f'ndcg@{k}'] = metrics['ndcg']
    
    # MRR
    if eval_config.get('compute_mrr', True):
        metrics = evaluate(retrieved_docs, gold_docs, len(retrieved_docs))
        results['mrr'] = metrics['mrr']
    
    return results


def load_retrieval_results(input_file: Path, logger) -> List[Dict]:
    """加载检索结果"""
    logger.info(f"Loading retrieval results from {input_file}")
    results = []
    
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return results
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        item = json.loads(line)
                        results.append(item)
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON decode error at line {line_num}: {e}")
                        continue
        
        logger.info(f"Loaded {len(results)} retrieval results")
    except Exception as e:
        logger.error(f"Failed to load retrieval results: {e}")
    
    return results


def main():
    """主程序"""
    # 加载配置
    parser = argparse.ArgumentParser(description='重排评估系统')
    parser.add_argument(
        'config',
        nargs='?',
        default='rerank_config.json',
        help='配置文件路径，默认使用 rerank_config.json'
    )
    args = parser.parse_args()

    config_file = Path(args.config)
    if not config_file.exists():
        print(f"Configuration file not found: {config_file}")
        return
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 设置日志
    logger = setup_logging(config)
    logger.info("="*80)
    logger.info("重排评估系统启动")
    logger.info("="*80)
    
    # 创建输出目录
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 加载数据
        logger.info("Step 1: 数据加载")
        logger.info("-" * 40)
        
        corpus = load_corpus(config['corpus_file'], logger)
        if not corpus:
            logger.error("Failed to load corpus, exiting")
            return
        
        gold_annotations = load_gold_annotations(config['gold_file'], logger)
        if not gold_annotations:
            logger.warning("Gold annotations not found or empty; will continue reranking without evaluation metrics")
        
        logger.info(f"数据加载完成 - 语料库: {len(corpus)} 文档, Gold标注: {len(gold_annotations)} 查询")
        
        # 实验结果存储
        all_results = []
        all_detailed_results = {}  # 修改：按检索器+重排器组合分别存储详细结果
        
        # 遍历检索器
        for retriever_name, retriever_config in config['retrievers'].items():
            if not retriever_config['enabled']:
                logger.info(f"Skipping disabled retriever: {retriever_name}")
                continue
            
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing retriever: {retriever_name}")
            logger.info(f"{'='*80}")
            
            # 加载检索结果
            input_file = Path(config['input_dir']) / retriever_config['input_file']
            retrieval_results = load_retrieval_results(input_file, logger)
            
            if not retrieval_results:
                logger.error(f"No retrieval results loaded for {retriever_name}")
                continue
            
            # 遍历重排器
            for reranker_name, reranker_config in config['rerankers'].items():
                if not reranker_config['enabled']:
                    logger.info(f"Skipping disabled reranker: {reranker_name}")
                    continue
                
                # 开始重排器处理
                logger.info(f"Step: 处理 {retriever_name} + {reranker_name}")
                logger.info(f"总查询数: {len(retrieval_results)}")
                
                logger.info(f"\nApplying reranker: {reranker_name}")
                
                # 为当前组合创建详细结果存储
                combination_key = f"{retriever_name}_{reranker_name}"
                all_detailed_results[combination_key] = []
                
                # 创建重排器
                reranker = None
                try:
                    reranker = RerankerFactory.create(reranker_name, reranker_config, logger)
                    
                    # 评估每个查询（合并相同flashrag_id的所有检索结果）
                    query_results = []
                    processed_queries = 0
                    
                    # 合并相同flashrag_id的检索结果
                    merged_results = {}
                    for item in retrieval_results:
                        qid = str(item['flashrag_id'])
                        if qid not in merged_results:
                            merged_results[qid] = {
                                'flashrag_id': qid,
                                'flashrag_question': item.get('flashrag_question', item.get('beir_question', '')),
                                'beir_qid': item.get('beir_qid'),
                                'merged_docs': []
                            }
                        
                        # 合并文档列表：同时支持标准格式（top50_docs/top200_docs）和融合格式（fused_docs）
                        doc_list = item.get('top50_docs', item.get('top200_docs', item.get('fused_docs', [])))
                        merged_results[qid]['merged_docs'].extend(doc_list)
                    
                    # 处理合并后的结果
                    for qid, merged_item in tqdm(merged_results.items(), desc=f"{retriever_name}+{reranker_name}"):
                        
                        query_text = merged_item['flashrag_question']
                        
                        # 当gold不可用或该qid无gold时：仍生成打分与重排结果，但评测指标将置为None
                        has_gold = qid in gold_annotations
                        
                        # 准备候选文档（去重并保持最佳排名）
                        docs_dict = {}
                        for doc_idx, doc_item in enumerate(merged_item['merged_docs']):
                            doc_id = str(doc_item['doc_id'])
                            if doc_id in corpus:
                                # 处理rank字段：如果不存在则使用文档在列表中的索引+1作为默认排名
                                doc_rank = doc_item.get('rank', doc_idx + 1)
                                # 如果文档已存在，保留排名更好的版本
                                if doc_id not in docs_dict or doc_rank < docs_dict[doc_id]['original_rank']:
                                    docs_dict[doc_id] = {
                                        'doc_id': doc_id,
                                        'text': corpus[doc_id],
                                        'original_rank': doc_rank,
                                        'original_score': doc_item.get('score', 0.0)
                                    }
                        
                        docs = list(docs_dict.values())
                        # 按原始排名排序
                        docs.sort(key=lambda x: x['original_rank'])
                        
                        if not docs:
                            continue
                        
                        # 重排
                        try:
                            reranked_docs = reranker.rerank(
                                query_text, 
                                docs, 
                                config['evaluation']['top_k_for_eval']
                            )
                            
                            # 🔧 新增：保存原始排序（用于对比）- 去除重复文档ID
                            seen_original_ids = set()
                            original_doc_ids = []
                            for d in docs[:config['evaluation']['top_k_for_eval']]:
                                doc_id = d['doc_id']
                                if doc_id not in seen_original_ids:
                                    original_doc_ids.append(doc_id)
                                    seen_original_ids.add(doc_id)
                            
                            seen_top50_ids = set()
                            original_top50_ids = []
                            for d in docs:
                                doc_id = d['doc_id']
                                if doc_id not in seen_top50_ids:
                                    original_top50_ids.append(doc_id)
                                    seen_top50_ids.add(doc_id)
                            
                            # 评估重排后结果 - 去除重复文档ID
                            seen_ids = set()
                            retrieved_ids = []
                            for d in reranked_docs:
                                doc_id = d['doc_id']
                                if doc_id not in seen_ids:
                                    retrieved_ids.append(doc_id)
                                    seen_ids.add(doc_id)
                            metrics = None
                            gold_set = set(gold_annotations[qid]) if has_gold else set()
                            if has_gold:
                                metrics = evaluate(
                                    retrieved_ids, 
                                    gold_annotations[qid], 
                                    config['evaluation']['top_k_for_eval']
                                )
                            
                            # 🔧 新增：计算额外指标（BestRank, AvgRank）
                            # BestRank（第一个 Gold Doc 的位置）
                            best_rank = None
                            avg_rank = None
                            original_metrics = None
                            if has_gold:
                                for i, doc_id in enumerate(retrieved_ids, 1):
                                    if doc_id in gold_set:
                                        best_rank = i
                                        break
                                
                                gold_ranks = [i for i, doc_id in enumerate(retrieved_ids, 1) if doc_id in gold_set]
                                avg_rank = sum(gold_ranks) / len(gold_ranks) if gold_ranks else None
                                
                                # 🔧 新增：原始排序的指标（用于对比）
                                original_metrics = evaluate(
                                    original_doc_ids,
                                    gold_annotations[qid],
                                    config['evaluation']['top_k_for_eval']
                                )
                            
                            if has_gold and metrics is not None:
                                query_results.append({
                                    'qid': qid,
                                    **metrics,
                                    'best_rank': best_rank,
                                    'avg_rank': avg_rank
                                })
                            
                            # 🔧 修改：保存详细结果到对应的组合中
                            all_detailed_results[combination_key].append({
                                'qid': qid,
                                'query': query_text,
                                'retriever': retriever_name,
                                'reranker': reranker_name,
                                'gold_docs': gold_annotations[qid] if has_gold else [],
                                'original_top50': original_top50_ids,
                                'reranked_top50': [d['doc_id'] for d in reranked_docs],
                                'original_top10_for_eval': original_doc_ids,
                                'reranked_top10_for_eval': retrieved_ids,
                                'metrics_before': {
                                    'recall': original_metrics['recall'] if original_metrics else None,
                                    'precision': original_metrics['precision'] if original_metrics else None,
                                    'mrr': original_metrics['mrr'] if original_metrics else None,
                                    'ndcg': original_metrics['ndcg'] if original_metrics else None
                                },
                                'metrics_after': {
                                    'recall': metrics['recall'] if metrics else None,
                                    'precision': metrics['precision'] if metrics else None,
                                    'mrr': metrics['mrr'] if metrics else None,
                                    'ndcg': metrics['ndcg'] if metrics else None,
                                    'best_rank': best_rank,
                                    'avg_rank': avg_rank
                                },
                                'improvement': {
                                    'recall_delta': (metrics['recall'] - original_metrics['recall']) if (metrics and original_metrics) else None,
                                    'mrr_delta': (metrics['mrr'] - original_metrics['mrr']) if (metrics and original_metrics) else None,
                                    'ndcg_delta': (metrics['ndcg'] - original_metrics['ndcg']) if (metrics and original_metrics) else None
                                },
                                'reranked_docs_with_scores': [
                                    {
                                        'doc_id': d['doc_id'],
                                        'rerank_score': d.get('rerank_score', 0.0),
                                        'original_rank': d['original_rank'],
                                        'new_rank': i + 1,
                                        'is_gold': d['doc_id'] in gold_set
                                    }
                                    for i, d in enumerate(reranked_docs)
                                ],
                                'note': f"original_top50包含完整的{len(original_top50_ids)}个文档，reranked_top50包含重排后的{len(reranked_docs)}个文档，评估基于前{config['evaluation']['top_k_for_eval']}个文档"
                            })
                            
                            processed_queries += 1
                            
                        except Exception as e:
                            logger.warning(f"Failed to process query {qid}: {e}")
                            continue
                    
                    # 计算平均指标
                    if query_results:
                        # 计算BestRank和AvgRank的平均值
                        best_ranks = [r['best_rank'] for r in query_results if r['best_rank'] is not None]
                        avg_ranks = [r['avg_rank'] for r in query_results if r['avg_rank'] is not None]
                        
                        avg_metrics = {
                            'retriever': retriever_name,
                            'reranker': reranker_name,
                            'num_queries': len(query_results),
                            'avg_recall': sum(r['recall'] for r in query_results) / len(query_results),
                            'avg_precision': sum(r['precision'] for r in query_results) / len(query_results),
                            'avg_mrr': sum(r['mrr'] for r in query_results) / len(query_results),
                            'avg_ndcg': sum(r['ndcg'] for r in query_results) / len(query_results),
                            'avg_best_rank': sum(best_ranks) / len(best_ranks) if best_ranks else None,
                            'avg_avg_rank': sum(avg_ranks) / len(avg_ranks) if avg_ranks else None
                        }
                        all_results.append(avg_metrics)
                        
                        logger.info(f"Processed {processed_queries} queries")
                        logger.info(f"Results: Recall={avg_metrics['avg_recall']:.4f}, "
                                  f"Precision={avg_metrics['avg_precision']:.4f}, "
                                  f"MRR={avg_metrics['avg_mrr']:.4f}, "
                                  f"NDCG={avg_metrics['avg_ndcg']:.4f}")
                        if avg_metrics['avg_best_rank']:
                            logger.info(f"BestRank={avg_metrics['avg_best_rank']:.2f}, "
                                      f"AvgRank={avg_metrics['avg_avg_rank']:.2f}")
                    else:
                        logger.warning(f"No valid results for {retriever_name}+{reranker_name}")
                
                except Exception as e:
                    logger.error(f"Failed to process {retriever_name}+{reranker_name}: {e}")
                
                finally:
                    # 清理重排器资源
                    if reranker:
                        reranker.cleanup()
        
        # 保存结果
        if all_results:
            logger.info("\nStep: 保存结果")
            logger.info("-" * 40)
            
            # 1. 保存汇总指标（CSV）
            results_df = pd.DataFrame(all_results)
            summary_file = output_dir / 'reranking_summary.csv'
            results_df.to_csv(summary_file, index=False, encoding='utf-8')
            logger.info(f"Summary results saved to {summary_file}")
            
            # 2. 为每个检索器+重排器组合生成独立的详细结果文件
            logger.info("Saving individual detailed results for each combination...")
            for combination_key, detailed_results in all_detailed_results.items():
                if detailed_results:  # 只保存有结果的组合
                    # 详细结果JSON文件
                    detailed_file = output_dir / f'{combination_key}_detailed_results.json'
                    with open(detailed_file, 'w', encoding='utf-8') as f:
                        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
                    logger.info(f"Detailed results for {combination_key}: {detailed_file}")
                    
                    # 对比分析CSV文件
                    comparison_data = []
                    for result in detailed_results:
                        comparison_data.append({
                            'qid': result['qid'],
                            'query': result['query'][:50] + '...' if len(result['query']) > 50 else result['query'],
                            'recall_before': result['metrics_before']['recall'],
                            'recall_after': result['metrics_after']['recall'],
                            'recall_delta': result['improvement']['recall_delta'],
                            'mrr_before': result['metrics_before']['mrr'],
                            'mrr_after': result['metrics_after']['mrr'],
                            'mrr_delta': result['improvement']['mrr_delta'],
                            'ndcg_before': result['metrics_before']['ndcg'],
                            'ndcg_after': result['metrics_after']['ndcg'],
                            'ndcg_delta': result['improvement']['ndcg_delta'],
                            'best_rank': result['metrics_after'].get('best_rank'),
                            'avg_rank': result['metrics_after'].get('avg_rank'),
                            'gold_docs_count': len(result['gold_docs'])
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    comparison_file = output_dir / f'{combination_key}_comparison.csv'
                    comparison_df.to_csv(comparison_file, index=False, encoding='utf-8')
                    logger.info(f"Comparison results for {combination_key}: {comparison_file}")
                    
                    # 生成单独的统计报告
                    report_file = output_dir / f'{combination_key}_report.txt'
                    with open(report_file, 'w', encoding='utf-8') as f:
                        retriever, reranker = combination_key.split('_', 1)
                        f.write("="*80 + "\n")
                        f.write(f"RERANKING EVALUATION REPORT - {retriever.upper()} + {reranker.upper()}\n")
                        f.write("="*80 + "\n\n")
                        
                        # 找到对应的汇总统计
                        summary_row = results_df[(results_df['retriever'] == retriever) & 
                                               (results_df['reranker'] == reranker)]
                        if not summary_row.empty:
                            f.write("1. Summary Statistics\n")
                            f.write("-"*40 + "\n")
                            f.write(f"Retriever: {retriever}\n")
                            f.write(f"Reranker: {reranker}\n")
                            f.write(f"Queries Processed: {summary_row.iloc[0]['num_queries']}\n")
                            f.write(f"Average Recall: {summary_row.iloc[0]['avg_recall']:.4f}\n")
                            f.write(f"Average Precision: {summary_row.iloc[0]['avg_precision']:.4f}\n")
                            f.write(f"Average MRR: {summary_row.iloc[0]['avg_mrr']:.4f}\n")
                            f.write(f"Average NDCG: {summary_row.iloc[0]['avg_ndcg']:.4f}\n")
                            if summary_row.iloc[0]['avg_best_rank']:
                                f.write(f"Average Best Rank: {summary_row.iloc[0]['avg_best_rank']:.2f}\n")
                                f.write(f"Average Avg Rank: {summary_row.iloc[0]['avg_avg_rank']:.2f}\n")
                            f.write("\n")
                        
                        f.write("2. Improvement Analysis\n")
                        f.write("-"*40 + "\n")
                        f.write(f"Average Recall Improvement: {comparison_df['recall_delta'].mean():.4f}\n")
                        f.write(f"Average MRR Improvement: {comparison_df['mrr_delta'].mean():.4f}\n")
                        f.write(f"Average NDCG Improvement: {comparison_df['ndcg_delta'].mean():.4f}\n")
                        f.write(f"Queries Improved (Recall): {(comparison_df['recall_delta'] > 0).sum()}/{len(comparison_df)}\n")
                        f.write(f"Queries Degraded (Recall): {(comparison_df['recall_delta'] < 0).sum()}/{len(comparison_df)}\n")
                        f.write(f"Queries Unchanged (Recall): {(comparison_df['recall_delta'] == 0).sum()}/{len(comparison_df)}\n")
                        
                        f.write("\n3. Top 10 Most Improved Queries (by Recall)\n")
                        f.write("-"*40 + "\n")
                        top_improved = comparison_df.nlargest(10, 'recall_delta')[['qid', 'query', 'recall_delta', 'mrr_delta']]
                        f.write(top_improved.to_string(index=False))
                        
                        f.write("\n\n4. Top 10 Most Degraded Queries (by Recall)\n")
                        f.write("-"*40 + "\n")
                        top_degraded = comparison_df.nsmallest(10, 'recall_delta')[['qid', 'query', 'recall_delta', 'mrr_delta']]
                        f.write(top_degraded.to_string(index=False))
                    
                    logger.info(f"Report for {combination_key}: {report_file}")
            
            # 3. 生成总体对比文件（所有组合）
            all_comparison_data = []
            for combination_key, detailed_results in all_detailed_results.items():
                for result in detailed_results:
                    all_comparison_data.append({
                        'combination': combination_key,
                        'qid': result['qid'],
                        'retriever': result['retriever'],
                        'reranker': result['reranker'],
                        'recall_before': result['metrics_before']['recall'],
                        'recall_after': result['metrics_after']['recall'],
                        'recall_delta': result['improvement']['recall_delta'],
                        'mrr_before': result['metrics_before']['mrr'],
                        'mrr_after': result['metrics_after']['mrr'],
                        'mrr_delta': result['improvement']['mrr_delta'],
                        'best_rank': result['metrics_after'].get('best_rank'),
                        'avg_rank': result['metrics_after'].get('avg_rank'),
                        'gold_docs_count': len(result['gold_docs'])
                    })
            
            if all_comparison_data:
                all_comparison_df = pd.DataFrame(all_comparison_data)
                all_comparison_file = output_dir / 'all_combinations_comparison.csv'
                all_comparison_df.to_csv(all_comparison_file, index=False, encoding='utf-8')
                logger.info(f"All combinations comparison: {all_comparison_file}")
            
            # 4. 生成总体统计报告
            overall_report_file = output_dir / 'overall_report.txt'
            with open(overall_report_file, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("OVERALL RERANKING EVALUATION REPORT\n")
                f.write("="*80 + "\n\n")
                
                f.write("1. Summary Statistics (All Combinations)\n")
                f.write("-"*40 + "\n")
                f.write(results_df.to_string(index=False))
                f.write("\n\n")
                
                f.write("2. Best Performing Combinations\n")
                f.write("-"*40 + "\n")
                f.write("By Recall:\n")
                best_recall = results_df.nlargest(3, 'avg_recall')[['retriever', 'reranker', 'avg_recall', 'avg_mrr']]
                f.write(best_recall.to_string(index=False))
                f.write("\n\nBy MRR:\n")
                best_mrr = results_df.nlargest(3, 'avg_mrr')[['retriever', 'reranker', 'avg_recall', 'avg_mrr']]
                f.write(best_mrr.to_string(index=False))
                f.write("\n\n")
                
                f.write("3. Generated Files Summary\n")
                f.write("-"*40 + "\n")
                f.write(f"Total combinations processed: {len(all_detailed_results)}\n")
                f.write("Individual files generated for each combination:\n")
                for combination_key in all_detailed_results.keys():
                    if all_detailed_results[combination_key]:
                        f.write(f"  - {combination_key}_detailed_results.json\n")
                        f.write(f"  - {combination_key}_comparison.csv\n")
                        f.write(f"  - {combination_key}_report.txt\n")
            
            logger.info(f"Overall report: {overall_report_file}")
            
            # 打印汇总表格
            print("\n" + "="*80)
            print("FINAL RESULTS - ALL COMBINATIONS")
            print("="*80)
            print(results_df.to_string(index=False))
            print(f"\n📁 Generated {len(all_detailed_results) * 3 + 3} files for {len(all_detailed_results)} combinations")
            
        else:
            logger.error("No results generated")
    
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        raise
    
    finally:
        logger.info("重排评估系统结束")


if __name__ == '__main__':
    main()
