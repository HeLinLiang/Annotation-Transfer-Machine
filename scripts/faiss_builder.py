"""
FAISS 索引构建器
支持千万级向量、断点恢复、GPU 加速、流式处理
"""

import os
import json
import pickle
import numpy as np
import faiss
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import logging
import time
import torch

logger = logging.getLogger(__name__)


class FAISSIndexBuilder:
    """FAISS 索引构建器（支持断点恢复）"""
    
    def __init__(
        self,
        embedding_dim: int,
        index_type: str = "IndexIVFPQ",
        nlist: int = 4096,
        nprobe: int = 32,
        pq_m: int = 64,
        pq_nbits: int = 8,
        use_gpu: bool = True,
        metric: str = "inner_product",
    ):
        """
        初始化 FAISS 索引构建器
        
        Args:
            embedding_dim: 向量维度
            index_type: 索引类型 ('IndexIVFPQ' 或 'IndexIVFFlat')
            nlist: IVF 聚类中心数量
            nprobe: 搜索时探测的聚类数量
            pq_m: PQ 子向量数量（仅 IndexIVFPQ）
            pq_nbits: PQ 每个子向量的比特数（仅 IndexIVFPQ）
            use_gpu: 是否使用 GPU
            metric: 相似度度量 ('inner_product' 或 'l2')
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.pq_m = pq_m
        self.pq_nbits = pq_nbits
        self.use_gpu = use_gpu
        self.metric = metric
        
        # 索引和元数据
        self.index = None
        self.doc_ids = []  # doc_id 列表（向量行号 -> doc_id）
        self.processed_count = 0  # 已处理的文档数量
        
        # GPU 资源
        self.gpu_resource = None
        if use_gpu and faiss.get_num_gpus() > 0:
            self.gpu_resource = faiss.StandardGpuResources()
            logger.info(f"GPU available: {faiss.get_num_gpus()} GPU(s)")
        
        logger.info(f"FAISSIndexBuilder initialized: dim={embedding_dim}, type={index_type}")
    
    def _create_index(self) -> faiss.Index:
        """创建 FAISS 索引"""
        # 选择度量类型
        if self.metric == "inner_product":
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
        else:
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
        
        # 创建索引
        if self.index_type == "IndexFlatIP":
            # Flat 索引，精确搜索，不需要训练
            index = faiss.IndexFlatIP(self.embedding_dim)
            logger.info(f"Created IndexFlatIP: exact search, no training needed")
        elif self.index_type == "IndexFlatL2":
            # Flat 索引，L2 距离
            index = faiss.IndexFlatL2(self.embedding_dim)
            logger.info(f"Created IndexFlatL2: exact search, no training needed")
        elif self.index_type == "IndexIVFPQ":
            index = faiss.IndexIVFPQ(
                quantizer,
                self.embedding_dim,
                self.nlist,
                self.pq_m,
                self.pq_nbits,
            )
            logger.info(f"Created IndexIVFPQ: nlist={self.nlist}, m={self.pq_m}, nbits={self.pq_nbits}")
        elif self.index_type == "IndexIVFFlat":
            index = faiss.IndexIVFFlat(
                quantizer,
                self.embedding_dim,
                self.nlist,
            )
            logger.info(f"Created IndexIVFFlat: nlist={self.nlist}")
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        # 设置 nprobe（仅对 IVF 索引有效）
        if hasattr(index, 'nprobe'):
            index.nprobe = self.nprobe
        
        return index
    
    def train(self, train_vectors: np.ndarray):
        """
        训练索引（IVF 需要训练聚类中心）
        
        Args:
            train_vectors: 训练向量，shape: (n_samples, embedding_dim)
        """
        if self.index is None:
            self.index = self._create_index()
        
        if self.index.is_trained:
            logger.info("Index already trained, skipping training")
            return
        
        logger.info(f"Training index with {len(train_vectors)} vectors...")
        start_time = time.time()
        
        # 确保向量是 float32
        train_vectors = train_vectors.astype(np.float32)
        
        # 训练
        if self.use_gpu and self.gpu_resource:
            # 对于大维度向量，使用CPU训练以避免GPU共享内存限制
            if self.embedding_dim > 512 and self.index_type == "IndexIVFPQ":
                logger.info(f"Using CPU training for high-dimensional vectors (dim={self.embedding_dim})")
                self.index.train(train_vectors)
            else:
                # 使用 GPU 训练
                # 对于 IndexIVFPQ，启用 float16 查找表以减少 shared memory 占用
                if self.index_type == "IndexIVFPQ":
                    self.index.use_precomputed_table = -1  # 禁用预计算表
                
                index_gpu = faiss.index_cpu_to_gpu(self.gpu_resource, 0, self.index)
                
                # 对于 GPU 上的 IndexIVFPQ，启用 float16 查找表
                if hasattr(index_gpu, 'useFloat16LookupTables'):
                    index_gpu.useFloat16LookupTables = True
                    logger.info("Enabled float16 lookup tables for GPU IndexIVFPQ")
                
                index_gpu.train(train_vectors)
                self.index = faiss.index_gpu_to_cpu(index_gpu)
        else:
            self.index.train(train_vectors)
        
        elapsed = time.time() - start_time
        logger.info(f"Index training completed in {elapsed:.2f}s")
    
    def add_vectors(self, vectors: np.ndarray, doc_ids: List[str]):
        """
        添加向量到索引（智能混合策略：小索引用 GPU，大索引用 CPU）
        
        Args:
            vectors: 向量数组，shape: (n_vectors, embedding_dim)
            doc_ids: 文档 ID 列表
        """
        if self.index is None:
            raise RuntimeError("Index not initialized. Call train() first.")
        
        if not self.index.is_trained:
            raise RuntimeError("Index not trained. Call train() first.")
        
        # 确保向量是 float32
        vectors = vectors.astype(np.float32)
        
        # 智能选择 GPU 或 CPU 添加
        # 当索引较小时（< 5M 向量），使用 GPU 加速
        # 当索引较大时（>= 5M 向量），使用 CPU 避免 OOM
        use_gpu_for_add = self.use_gpu and self.gpu_resource and self.index.ntotal < 5000000
        
        if use_gpu_for_add:
            try:
                # 使用 GPU 加速添加（仅在索引较小时）
                logger.debug(f"Adding vectors on GPU (index size: {self.index.ntotal:,})...")
                cloner_options = faiss.GpuClonerOptions()
                cloner_options.useFloat16 = True  # 使用 FP16 精度减少显存
                
                index_gpu = faiss.index_cpu_to_gpu(self.gpu_resource, 0, self.index, cloner_options)
                index_gpu.add(vectors)
                self.index = faiss.index_gpu_to_cpu(index_gpu)
                
                # 显式清理 GPU 资源
                del index_gpu
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                # GPU 添加失败（可能显存不足），回退到 CPU
                logger.warning(f"GPU add failed ({e}), falling back to CPU")
                self.index.add(vectors)
        else:
            # 使用 CPU 添加（索引较大或 GPU 不可用）
            if self.index.ntotal >= 5000000:
                logger.debug(f"Adding vectors on CPU (index size: {self.index.ntotal:,}, switched to CPU mode for stability)")
            self.index.add(vectors)
        
        # 更新元数据
        self.doc_ids.extend(doc_ids)
        self.processed_count += len(doc_ids)
    
    def search(
        self,
        query_vectors: np.ndarray,
        top_k: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        搜索最相似的向量
        
        Args:
            query_vectors: 查询向量，shape: (n_queries, embedding_dim)
            top_k: 返回 top-k 结果
            
        Returns:
            scores: 相似度分数，shape: (n_queries, top_k)
            indices: 向量索引，shape: (n_queries, top_k)
        """
        if self.index is None:
            raise RuntimeError("Index not initialized")
        
        # 确保向量是 float32
        query_vectors = query_vectors.astype(np.float32)
        
        # 搜索
        if self.use_gpu and self.gpu_resource:
            index_gpu = faiss.index_cpu_to_gpu(self.gpu_resource, 0, self.index)
            
            # 对于 GPU 上的 IndexIVFPQ，启用 float16 查找表
            if hasattr(index_gpu, 'useFloat16LookupTables'):
                index_gpu.useFloat16LookupTables = True
            
            scores, indices = index_gpu.search(query_vectors.astype(np.float32), top_k)
        else:
            scores, indices = self.index.search(query_vectors.astype(np.float32), top_k)
        
        return scores, indices
    
    def save(self, index_path: str, metadata_path: str):
        """
        保存索引和元数据
        
        Args:
            index_path: 索引文件路径
            metadata_path: 元数据文件路径
        """
        if self.index is None:
            raise RuntimeError("Index not initialized")
        
        # 保存索引
        logger.info(f"Saving index to {index_path}...")
        faiss.write_index(self.index, index_path)
        
        # 保存元数据
        metadata = {
            "doc_ids": self.doc_ids,
            "processed_count": self.processed_count,
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "pq_m": self.pq_m,
            "pq_nbits": self.pq_nbits,
        }
        
        logger.info(f"Saving metadata to {metadata_path}...")
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Index saved: {self.processed_count} vectors")
    
    def load(self, index_path: str, metadata_path: str):
        """
        加载索引和元数据
        
        Args:
            index_path: 索引文件路径
            metadata_path: 元数据文件路径
        """
        # 加载索引
        logger.info(f"Loading index from {index_path}...")
        self.index = faiss.read_index(index_path)
        
        # 加载元数据
        logger.info(f"Loading metadata from {metadata_path}...")
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        
        self.doc_ids = metadata["doc_ids"]
        self.processed_count = metadata["processed_count"]
        self.embedding_dim = metadata["embedding_dim"]
        self.index_type = metadata["index_type"]
        self.nlist = metadata["nlist"]
        self.nprobe = metadata["nprobe"]
        self.pq_m = metadata.get("pq_m", self.pq_m)
        self.pq_nbits = metadata.get("pq_nbits", self.pq_nbits)
        
        logger.info(f"Index loaded: {self.processed_count} vectors")
    
    def save_checkpoint(self, checkpoint_path: str):
        """
        保存 checkpoint（用于断点恢复）
        
        Args:
            checkpoint_path: checkpoint 文件路径
        """
        checkpoint = {
            "processed_count": self.processed_count,
            "doc_ids": self.doc_ids,
        }
        
        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint, f)
        
        logger.info(f"Checkpoint saved: {self.processed_count} documents processed")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        加载 checkpoint
        
        Args:
            checkpoint_path: checkpoint 文件路径
            
        Returns:
            已处理的文档数量
        """
        if not os.path.exists(checkpoint_path):
            logger.info("No checkpoint found, starting from scratch")
            return 0
        
        logger.info(f"Loading checkpoint from {checkpoint_path}...")
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
        
        self.processed_count = checkpoint["processed_count"]
        self.doc_ids = checkpoint["doc_ids"]
        
        logger.info(f"Checkpoint loaded: {self.processed_count} documents already processed")
        return self.processed_count
    
    def get_stats(self) -> Dict:
        """获取索引统计信息"""
        if self.index is None:
            return {"status": "not_initialized"}
        
        return {
            "total_vectors": self.index.ntotal,
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "is_trained": self.index.is_trained,
            "processed_count": self.processed_count,
        }
