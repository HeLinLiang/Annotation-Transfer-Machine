"""
向量编码器模块
支持 E5 和 BGE 模型的文档编码，自动处理前缀、归一化等
"""

import torch
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class BaseEncoder:
    """向量编码器基类"""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_seq_length: int = 512,
        normalize: bool = True,
        prefix: str = "",
        torch_dtype: str = "float32",
    ):
        self.model_path = model_path
        self.device = device
        self.max_seq_length = max_seq_length
        self.normalize = normalize
        self.prefix = prefix
        self.torch_dtype = torch_dtype
        
        logger.info(f"Loading model from {model_path}...")
        logger.info(f"Using torch_dtype: {torch_dtype}")
        
        # 转换torch_dtype
        if torch_dtype == "float16":
            import torch
            dtype = torch.float16
        else:
            import torch
            dtype = torch.float32
            
        # SentenceTransformer可能不支持torch_dtype参数，使用替代方法
        self.model = SentenceTransformer(model_path, device=device)
        self.model.max_seq_length = max_seq_length
        
        # 手动设置模型精度
        if torch_dtype == "float16":
            self.model.half()
            logger.info("Model converted to float16 precision")
        
        # 获取向量维度
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 512,
        show_progress: bool = False,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        """
        编码文本为向量
        
        Args:
            texts: 单个文本或文本列表
            batch_size: 批处理大小
            show_progress: 是否显示进度条
            convert_to_numpy: 是否转换为 numpy 数组
            
        Returns:
            向量数组，shape: (n_texts, embedding_dim)
        """
        # 添加前缀
        if isinstance(texts, str):
            texts = [texts]
        
        if self.prefix:
            texts = [self.prefix + text for text in texts]
        
        # 编码
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=self.normalize,
            device=self.device,
        )
        
        return embeddings
    
    def encode_stream(
        self,
        texts_iterator,
        batch_size: int = 512,
        total: int = None,
    ):
        """
        流式编码文本（逐批次处理，节省内存）
        
        Args:
            texts_iterator: 文本迭代器
            batch_size: 批处理大小
            total: 总文本数量（用于进度条）
            
        Yields:
            每个 batch 的向量数组
        """
        batch = []
        pbar = tqdm(total=total, desc="Encoding", disable=not total)
        
        for text in texts_iterator:
            batch.append(text)
            
            if len(batch) >= batch_size:
                embeddings = self.encode(batch, batch_size=batch_size, show_progress=False)
                yield embeddings
                pbar.update(len(batch))
                batch = []
        
        # 处理剩余的文本
        if batch:
            embeddings = self.encode(batch, batch_size=batch_size, show_progress=False)
            yield embeddings
            pbar.update(len(batch))
        
        pbar.close()


class E5Encoder(BaseEncoder):
    """E5 模型编码器"""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_seq_length: int = 512,
        torch_dtype: str = "float32",
    ):
        super().__init__(
            model_path=model_path,
            device=device,
            max_seq_length=max_seq_length,
            normalize=True,      # E5 需要L2归一化
            prefix="passage: ",  # E5 文档编码前缀
            torch_dtype=torch_dtype,
        )
        logger.info(f"E5 encoder initialized with 'passage: ' prefix, L2 normalization, and {torch_dtype} precision")
    
    def encode_query(
        self,
        queries: Union[str, List[str]],
        batch_size: int = 128,
    ) -> np.ndarray:
        """
        编码查询（使用 'query: ' 前缀）
        
        Args:
            queries: 单个查询或查询列表
            batch_size: 批处理大小
            
        Returns:
            查询向量数组
        """
        if isinstance(queries, str):
            queries = [queries]
        
        # 查询使用 'query: ' 前缀
        queries_with_prefix = ["query: " + q for q in queries]
        
        embeddings = self.model.encode(
            queries_with_prefix,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=self.device,
        )
        
        return embeddings


class BGEEncoder(BaseEncoder):
    """BGE 模型编码器"""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_seq_length: int = 512,
        torch_dtype: str = "float32",
    ):
        super().__init__(
            model_path=model_path,
            device=device,
            max_seq_length=max_seq_length,
            normalize=True,  # BGE normalize_embeddings=True
            prefix="",       # BGE 不需要前缀
            torch_dtype=torch_dtype,
        )
        logger.info(f"BGE encoder initialized with normalize_embeddings=True and {torch_dtype} precision")
    
    def encode_query(
        self,
        queries: Union[str, List[str]],
        batch_size: int = 128,
    ) -> np.ndarray:
        """
        编码查询（BGE 查询和文档编码方式相同）
        
        Args:
            queries: 单个查询或查询列表
            batch_size: 批处理大小
            
        Returns:
            查询向量数组
        """
        return self.encode(queries, batch_size=batch_size, show_progress=False)


def get_encoder(encoder_type: str, model_path: str, device: str = "cuda", max_seq_length: int = 512, torch_dtype: str = "float32"):
    """
    工厂方法：根据类型创建编码器
    
    Args:
        encoder_type: 编码器类型 ('e5', 'e5_large' 或 'bge')
        model_path: 模型路径
        device: 设备
        max_seq_length: 最大序列长度
        torch_dtype: torch数据类型 ('float32' 或 'float16')
        
    Returns:
        编码器实例
    """
    encoder_type = encoder_type.lower()
    
    if encoder_type in ["e5", "e5_large"]:
        return E5Encoder(model_path, device, max_seq_length, torch_dtype)
    elif encoder_type == "bge":
        return BGEEncoder(model_path, device, max_seq_length, torch_dtype)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}. Supported: 'e5', 'e5_large', 'bge'")
