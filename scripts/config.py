"""
FAISS 稠密检索索引构建配置
集中管理所有关键参数，便于快速调参和复现
"""

import os

# ==================== 路径配置 ====================
BASE_DIR = "/home/asus/projects_hll/windsurf_projects/ATM_2080"
MODELS_DIR = os.path.join(BASE_DIR, "models")

# 索引文件存储在 index/ 目录下，按模型分类
INDEX_BASE_DIR = os.path.join(BASE_DIR, "index")
E5_INDEX_DIR = os.path.join(INDEX_BASE_DIR, "e5_index")
E5_LARGE_INDEX_DIR = os.path.join(INDEX_BASE_DIR, "e5_large_index")
BGE_INDEX_DIR = os.path.join(INDEX_BASE_DIR, "bge_index")

# 数据文件
CORPUS_FILE = os.path.join(BASE_DIR, "data", "wiki18_100w.jsonl")

# 模型路径
E5_MODEL_PATH = os.path.join(MODELS_DIR, "e5-small-v2")  # 切换到e5-small-v2 (384维)
E5_LARGE_MODEL_PATH = os.path.join(MODELS_DIR, "e5-large-v2")  # e5-large-v2 (1024维)
BGE_MODEL_PATH = os.path.join(MODELS_DIR, "bge-small-en-v1.5")

# E5 索引输出路径
E5_INDEX_PATH = os.path.join(E5_INDEX_DIR, "index_e5.faiss")
E5_METADATA_PATH = os.path.join(E5_INDEX_DIR, "index_e5_metadata.pkl")
E5_CHECKPOINT_PATH = os.path.join(E5_INDEX_DIR, "index_e5_checkpoint.pkl")

# E5 Large 索引输出路径
E5_LARGE_INDEX_PATH = os.path.join(E5_LARGE_INDEX_DIR, "index_e5_large.faiss")
E5_LARGE_METADATA_PATH = os.path.join(E5_LARGE_INDEX_DIR, "index_e5_large_metadata.pkl")
E5_LARGE_CHECKPOINT_PATH = os.path.join(E5_LARGE_INDEX_DIR, "index_e5_large_checkpoint.pkl")

# BGE 索引输出路径
BGE_INDEX_PATH = os.path.join(BGE_INDEX_DIR, "index_bge.faiss")
BGE_METADATA_PATH = os.path.join(BGE_INDEX_DIR, "index_bge_metadata.pkl")
BGE_CHECKPOINT_PATH = os.path.join(BGE_INDEX_DIR, "index_bge_checkpoint.pkl")

# ==================== 编码配置 ====================
# GPU 配置
DEVICE = "cuda"
MAX_SEQ_LENGTH = 512  # 最大序列长度
TORCH_DTYPE = "float16"  # 使用FP16减少显存占用，特别是large模型

# Batch size（针对 RTX 2080 Ti 11GB 显存优化 - 基于实际GPU使用情况优化）
# E5-small-v2: 384 维, E5-large-v2: 1024 维
ENCODE_BATCH_SIZE = 160   # E5-small-v2 batch size (从96增加到160)
E5_LARGE_BATCH_SIZE = 96  # E5-large-v2 batch size (从48增加到96)
MIN_BATCH_SIZE = 64       # 最小 batch size (从32增加到64)
MAX_BATCH_SIZE = 256      # 最大 batch size (从128增加到256)

# 编码器配置
E5_CONFIG = {
    "model_path": E5_MODEL_PATH,
    "prefix": "passage: ",  # E5 文档编码前缀
    "normalize": True,      # L2 归一化
    "max_seq_length": MAX_SEQ_LENGTH,
}

E5_LARGE_CONFIG = {
    "model_path": E5_LARGE_MODEL_PATH,
    "prefix": "passage: ",  # E5 文档编码前缀
    "normalize": True,      # L2 归一化
    "max_seq_length": MAX_SEQ_LENGTH,
}

BGE_CONFIG = {
    "model_path": BGE_MODEL_PATH,
    "prefix": "",           # BGE 不需要前缀
    "normalize": True,      # normalize_embeddings=True
    "max_seq_length": MAX_SEQ_LENGTH,
}

# ==================== FAISS 索引配置 ====================
# 索引类型选择：IndexIVFPQ（高压缩比）或 IndexIVFFlat（高精度）
INDEX_TYPE = "IndexIVFPQ"  # 可选: "IndexIVFPQ", "IndexIVFFlat"

# IVF 参数
NLIST = 4096              # 聚类中心数量（推荐 sqrt(N) 到 4*sqrt(N)）
NPROBE = 32               # 搜索时探测的聚类数量

# PQ 参数（针对不同维度向量优化）
PQ_M = 48               # 子向量数量（384维可以被48整除）
PQ_NBITS = 8            # 每个子向量的比特数
PQ_M_LARGE = 32         # E5-large 1024维的子向量数量（减少以适应GPU共享内存限制）
USE_FLOAT16_LOOKUP = True # 使用 float16 查找表减少 shared memory 占用

# 训练参数（基于GPU性能优化）
TRAIN_SAMPLE_SIZE = 300000  # 用于训练 IVF 的样本数量（增加到300K提高训练质量）
USE_GPU_FOR_INDEX = True    # 是否使用 GPU 加速索引构建

# ==================== 断点恢复配置 ====================
CHECKPOINT_INTERVAL = 100000  # 每处理多少条文档保存一次 checkpoint（减少IO开销）
RESUME_FROM_CHECKPOINT = True  # 是否从 checkpoint 恢复

# ==================== 日志配置 ====================
LOG_INTERVAL = 50000          # 每处理多少条文档打印一次日志（减少IO开销）
VERBOSE = True                # 是否打印详细日志

# ==================== 搜索配置 ====================
DEFAULT_TOP_K = 100           # 默认返回 top-k 结果
SEARCH_BATCH_SIZE = 128       # 批量搜索的 batch size
