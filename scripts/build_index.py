"""
主索引构建脚本
支持 E5 和 BGE 模型的 FAISS 索引构建，具备断点恢复功能
"""

import os
import json
import numpy as np
import logging
import time
import datetime
import argparse
from typing import Iterator, Tuple, List
import torch

from config import *
from encoders import get_encoder
from faiss_builder import FAISSIndexBuilder

# 创建日志目录
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log")
os.makedirs(LOG_DIR, exist_ok=True)

def setup_logging(encoder_type: str):
    """设置日志，同时输出到文件和控制台"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"build_index_{encoder_type}_{timestamp}.log")
    
    # 配置根日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__), log_file

logger = logging.getLogger(__name__)  # 这里的 logger 后续会被 setup_logging 重新配置覆盖

def log_gpu_usage():
    """记录当前 GPU 使用情况"""
    if torch.cuda.is_available():
        try:
            # 获取当前设备的显存使用
            device = torch.cuda.current_device()
            memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
            max_memory = torch.cuda.max_memory_allocated(device) / 1024**3
            
            logger.info(f"GPU Memory: Allocated={memory_allocated:.2f}GB, Reserved={memory_reserved:.2f}GB, Max={max_memory:.2f}GB")
        except Exception as e:
            logger.warning(f"Failed to get GPU stats: {e}")


def read_jsonl_stream(file_path: str, skip_lines: int = 0) -> Iterator[Tuple[str, str]]:
    """
    流式读取 JSONL 文件（逐行读取，不加载全部数据）
    
    Args:
        file_path: JSONL 文件路径
        skip_lines: 跳过前 N 行（用于断点恢复）
        
    Yields:
        (doc_id, text) 元组
    """
    logger.info(f"Reading corpus from {file_path} (skipping first {skip_lines} lines)...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < skip_lines:
                continue
            
            try:
                doc = json.loads(line.strip())
                doc_id = doc.get('id', str(i))
                text = (
                    doc.get('text')
                    or doc.get('contents')
                    or doc.get('content')
                    or doc.get('passage')
                    or ''
                )
                if isinstance(text, str):
                    text = text.strip()
                
                if text:
                    yield doc_id, text
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {i}: {e}")
                continue


def count_lines(file_path: str) -> int:
    """快速统计文件行数"""
    logger.info(f"Counting lines in {file_path}...")
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for _ in f:
            count += 1
    logger.info(f"Total lines: {count:,}")
    return count


def collect_training_samples(
    file_path: str,
    sample_size: int,
    encoder,
    batch_size: int,
) -> np.ndarray:
    """
    收集训练样本（用于训练 IVF 聚类中心）
    
    Args:
        file_path: 语料文件路径
        sample_size: 样本数量
        encoder: 编码器
        batch_size: 批处理大小
        
    Returns:
        训练向量数组
    """
    logger.info(f"Collecting {sample_size} training samples...")
    
    texts = []
    for doc_id, text in read_jsonl_stream(file_path):
        texts.append(text)
        if len(texts) >= sample_size:
            break
    
    logger.info(f"Collected {len(texts)} samples, encoding...")
    if len(texts) == 0:
        raise RuntimeError(
            "Collected 0 training samples from corpus. "
            "Please check JSONL field names (expected one of: text/contents/content/passage) "
            "and that the file is not empty."
        )
    train_vectors = encoder.encode(texts, batch_size=batch_size, show_progress=True)
    
    return train_vectors


def build_index(
    encoder_type: str,
    corpus_file: str,
    index_path: str,
    metadata_path: str,
    checkpoint_path: str,
    resume: bool = True,
):
    """
    构建 FAISS 索引
    
    Args:
        encoder_type: 编码器类型 ('e5', 'e5_large' 或 'bge')
        corpus_file: 语料文件路径
        index_path: 索引输出路径
        metadata_path: 元数据输出路径
        checkpoint_path: checkpoint 路径
        resume: 是否从 checkpoint 恢复
    """
    logger.info(f"=" * 80)
    logger.info(f"Building {encoder_type.upper()} index")
    logger.info(f"=" * 80)
    
    # 初始化编码器
    if encoder_type == "e5":
        model_path = E5_MODEL_PATH
        encoder_config = E5_CONFIG
        batch_size = ENCODE_BATCH_SIZE
        pq_m = PQ_M
    elif encoder_type == "e5_large":
        model_path = E5_LARGE_MODEL_PATH
        encoder_config = E5_LARGE_CONFIG
        batch_size = E5_LARGE_BATCH_SIZE
        pq_m = PQ_M_LARGE
    elif encoder_type == "bge":
        model_path = BGE_MODEL_PATH
        encoder_config = BGE_CONFIG
        batch_size = ENCODE_BATCH_SIZE
        pq_m = PQ_M
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    encoder = get_encoder(
        encoder_type=encoder_type,
        model_path=model_path,
        device=DEVICE,
        max_seq_length=MAX_SEQ_LENGTH,
        torch_dtype=TORCH_DTYPE,
    )
    
    # 初始化 FAISS 索引构建器
    builder = FAISSIndexBuilder(
        embedding_dim=encoder.embedding_dim,
        index_type=INDEX_TYPE,
        nlist=NLIST,
        nprobe=NPROBE,
        pq_m=pq_m,
        pq_nbits=PQ_NBITS,
        use_gpu=USE_GPU_FOR_INDEX,
        metric="inner_product",  # 使用内积（向量已归一化，等价于 cosine）
    )
    
    # 检查是否从 checkpoint 恢复
    skip_lines = 0
    if resume and os.path.exists(checkpoint_path):
        skip_lines = builder.load_checkpoint(checkpoint_path)
        if skip_lines > 0 and os.path.exists(index_path):
            logger.info("Loading existing index...")
            builder.load(index_path, metadata_path)
    
    # 如果索引未训练，先训练
    if builder.index is None or not builder.index.is_trained:
        train_vectors = collect_training_samples(
            corpus_file,
            TRAIN_SAMPLE_SIZE,
            encoder,
            batch_size,
        )
        builder.train(train_vectors)
        del train_vectors  # 释放内存
    
    # 统计总文档数
    total_docs = count_lines(corpus_file)
    remaining_docs = total_docs - skip_lines
    
    if remaining_docs <= 0:
        logger.info("All documents already processed!")
        logger.info(f"Index stats: {builder.get_stats()}")
        return
    
    logger.info(f"Processing {remaining_docs:,} documents (skipping first {skip_lines:,})...")
    
    # 流式编码并添加到索引
    start_time = time.time()
    batch_texts = []
    batch_doc_ids = []
    processed_in_session = 0
    
    try:
        for doc_id, text in read_jsonl_stream(corpus_file, skip_lines):
            batch_texts.append(text)
            batch_doc_ids.append(doc_id)
            
            # 达到 batch size，进行编码
            if len(batch_texts) >= batch_size:
                # 编码
                embeddings = encoder.encode(
                    batch_texts,
                    batch_size=batch_size,
                    show_progress=False,
                )
                
                # 添加到索引
                builder.add_vectors(embeddings, batch_doc_ids)
                
                processed_in_session += len(batch_texts)
                
                # 打印进度
                if processed_in_session % LOG_INTERVAL == 0:
                    elapsed = time.time() - start_time
                    speed = processed_in_session / elapsed
                    eta = (remaining_docs - processed_in_session) / speed if speed > 0 else 0
                    logger.info(
                        f"Processed: {builder.processed_count:,} / {total_docs:,} "
                        f"({builder.processed_count/total_docs*100:.1f}%) | "
                        f"Speed: {speed:.1f} docs/s | "
                        f"ETA: {eta/3600:.1f}h"
                    )
                    log_gpu_usage()
                
                # 保存 checkpoint
                if processed_in_session % CHECKPOINT_INTERVAL == 0:
                    logger.info("Saving checkpoint...")
                    builder.save(index_path, metadata_path)
                    builder.save_checkpoint(checkpoint_path)
                
                # 清空 batch
                batch_texts = []
                batch_doc_ids = []
        
        # 处理剩余的文档
        if batch_texts:
            embeddings = encoder.encode(
                batch_texts,
                batch_size=ENCODE_BATCH_SIZE,
                show_progress=False,
            )
            builder.add_vectors(embeddings, batch_doc_ids)
            processed_in_session += len(batch_texts)
        
        # 最终保存
        logger.info("Saving final index...")
        builder.save(index_path, metadata_path)
        
        # 删除 checkpoint（构建完成）
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            logger.info("Checkpoint removed (build completed)")
        
    except KeyboardInterrupt:
        logger.warning("Build interrupted by user, saving checkpoint...")
        builder.save(index_path, metadata_path)
        builder.save_checkpoint(checkpoint_path)
        logger.info("Checkpoint saved, you can resume later")
        return
    
    except Exception as e:
        logger.error(f"Error during build: {e}", exc_info=True)
        logger.info("Saving checkpoint before exit...")
        builder.save(index_path, metadata_path)
        builder.save_checkpoint(checkpoint_path)
        raise
    
    # 打印统计信息
    elapsed = time.time() - start_time
    logger.info(f"=" * 80)
    logger.info(f"Index build completed!")
    logger.info(f"Total time: {elapsed/3600:.2f} hours")
    logger.info(f"Average speed: {processed_in_session/elapsed:.1f} docs/s")
    logger.info(f"Index stats: {builder.get_stats()}")
    logger.info(f"=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Build FAISS dense retrieval index")
    parser.add_argument(
        "--encoder",
        type=str,
        choices=["e5", "e5_large", "bge", "all"],
        default="all",
        help="Encoder type to use (e5, e5_large, bge, or all)",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=CORPUS_FILE,
        help="Path to corpus JSONL file",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume from checkpoint (start from scratch)",
    )
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(E5_INDEX_DIR, exist_ok=True)
    os.makedirs(E5_LARGE_INDEX_DIR, exist_ok=True)
    os.makedirs(BGE_INDEX_DIR, exist_ok=True)
    
    # 初始化日志
    global logger
    logger, log_file_path = setup_logging(args.encoder)
    logger.info(f"Arguments: {args}")
    logger.info(f"Log file saved to: {log_file_path}")
    
    # 检查语料文件
    if not os.path.exists(args.corpus):
        logger.error(f"Corpus file not found: {args.corpus}")
        return
    
    # 检查 GPU
    if torch.cuda.is_available():
        logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("No GPU available, using CPU (will be slower)")
    
    # 构建索引
    resume = not args.no_resume
    
    if args.encoder in ["e5", "all"]:
        build_index(
            encoder_type="e5",
            corpus_file=args.corpus,
            index_path=E5_INDEX_PATH,
            metadata_path=E5_METADATA_PATH,
            checkpoint_path=E5_CHECKPOINT_PATH,
            resume=resume,
        )
    
    if args.encoder in ["e5_large", "all"]:
        build_index(
            encoder_type="e5_large",
            corpus_file=args.corpus,
            index_path=E5_LARGE_INDEX_PATH,
            metadata_path=E5_LARGE_METADATA_PATH,
            checkpoint_path=E5_LARGE_CHECKPOINT_PATH,
            resume=resume,
        )
    
    if args.encoder in ["bge", "all"]:
        build_index(
            encoder_type="bge",
            corpus_file=args.corpus,
            index_path=BGE_INDEX_PATH,
            metadata_path=BGE_METADATA_PATH,
            checkpoint_path=BGE_CHECKPOINT_PATH,
            resume=resume,
        )
    
    logger.info("All done!")


if __name__ == "__main__":
    main()
