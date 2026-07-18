#!/usr/bin/env python3
"""
使用现有模型索引检索TOP50文档

流程：
1. 从 nq_rengong_filtered.jsonl 读取 beir_qid
2. 从 test.tsv 找到对应的 corpus-id
3. 从 corpus.jsonl 读取 BEIR 文档内容作为 query
4. 使用指定模型在 wiki18_100w.jsonl 索引中检索 TOP50
5. 支持通过 config.json 选择不同的模型
"""

import json
import sys
import pickle
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import argparse
import logging
import datetime
import time
import subprocess
import shutil

# 配置日志
logger = logging.getLogger("Retriever")

def get_system_status():
    """获取系统资源使用情况 (CPU, RAM, GPU)"""
    status = []
    
    # 1. 获取 GPU 信息 (nvidia-smi)
    try:
        if shutil.which("nvidia-smi"):
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    idx, name, util, mem_used, mem_total = [x.strip() for x in line.split(',')]
                    status.append(f"GPU {idx} ({name}): Util {util}%, Mem {mem_used}/{mem_total} MB")
    except Exception as e:
        status.append(f"GPU Info Error: {e}")

    # 2. 获取 CPU/RAM 信息 (尝试使用 psutil，如果没有则用简单方法)
    try:
        import psutil
        cpu_percent = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        status.append(f"CPU: {cpu_percent}% | RAM: {mem.used/1024**3:.1f}/{mem.total/1024**3:.1f} GB ({mem.percent}%)")
    except ImportError:
        # Fallback: 读取 /proc/loadavg
        try:
            with open("/proc/loadavg", "r") as f:
                load = f.read().split()[0]
            status.append(f"CPU Load (1min): {load}")
        except:
            pass
            
    return " | ".join(status)

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_BASE_DIR = _SCRIPT_DIR
_BASE_DIR = Path(os.environ.get("RETRIEVER_TOP50_BASE_DIR", str(_DEFAULT_BASE_DIR))).resolve()
_AUTODL_TMP = os.environ.get("AUTODL_TMP", "/root/autodl-tmp")

# 添加模型索引路径到 sys.path
_MODEL_INDEX_DIR = os.environ.get("MODEL_INDEX_DIR")
if _MODEL_INDEX_DIR:
    sys.path.insert(0, _MODEL_INDEX_DIR)
else:
    # 默认指向 ../index 目录
    sys.path.insert(0, str(_SCRIPT_DIR.parent / "index"))

from searcher import DenseRetriever




def _resolve_config_path(value: str, config_dir: Path) -> str:
    if value is None:
        return value

    value = os.path.expandvars(value)
    p = Path(value)

    if p.is_absolute():
        old_prefix = "/root/autodl-tmp"
        if str(p).startswith(old_prefix):
            return str(Path(_AUTODL_TMP) / str(p)[len(old_prefix):].lstrip("/"))
        return str(p)

    return str((config_dir / p).resolve())


def _resolve_config_paths(config: dict, config_dir: Path) -> dict:
    for section_key in ("data_paths", "output_config"):
        section = config.get(section_key)
        if isinstance(section, dict):
            for k, v in list(section.items()):
                if isinstance(v, str):
                    section[k] = _resolve_config_path(v, config_dir)

    model_configs = config.get("model_configs")
    if isinstance(model_configs, dict):
        for _, mc in model_configs.items():
            if not isinstance(mc, dict):
                continue
            for k in ("model_path", "index_path", "metadata_path"):
                if isinstance(mc.get(k), str):
                    mc[k] = _resolve_config_path(mc[k], config_dir)

    return config


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    config_path_obj = Path(config_path).resolve()
    with open(config_path_obj, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return _resolve_config_paths(config, config_dir=config_path_obj.parent)


def load_question_matches(file_path: str) -> List[Dict]:
    """
    加载 nq_rengong_filtered.jsonl
    
    Returns:
        [{"beir_qid": "test0", "flashrag_id": "...", ...}, ...]
    """
    print(f"加载问题匹配文件: {file_path}")
    data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line.strip())
                if 'beir_qid' in item:
                    data.append(item)
    
    print(f"✓ 加载了 {len(data)} 个问题匹配记录")
    return data


def load_qrels(tsv_path: str) -> Dict[str, List[str]]:
    """
    加载 test.tsv，构建 query-id -> [corpus-id] 映射
    
    Returns:
        {"test0": ["doc0", "doc1"], ...}
    """
    print(f"加载 qrels 文件: {tsv_path}")
    qrels = {}
    
    with open(tsv_path, 'r', encoding='utf-8') as f:
        next(f)  # 跳过表头
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                query_id = parts[0]
                corpus_id = parts[1]
                
                if query_id not in qrels:
                    qrels[query_id] = []
                qrels[query_id].append(corpus_id)
    
    print(f"✓ 加载了 {len(qrels)} 个查询的相关文档")
    return qrels


def load_corpus(corpus_file: str) -> Dict[str, str]:
    """
    加载 BEIR corpus.jsonl
    
    Returns:
        {"doc0": "title + text", ...}
    """
    print(f"加载 BEIR corpus: {corpus_file}")
    corpus = {}
    
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="加载 corpus"):
            if line.strip():
                doc = json.loads(line.strip())
                doc_id = doc['_id']
                title = doc.get('title', '')
                text = doc.get('text', '')
                # 合并 title 和 text
                content = f"{title}\n{text}" if title else text
                corpus[doc_id] = content
    
    print(f"✓ 加载了 {len(corpus):,} 个文档")
    return corpus


def initialize_retriever(
    model_config: dict,
    device: str = "cuda",
    use_gpu_for_search: bool = True,
    encode_batch_size: int = 128,
    faiss_nprobe: Optional[int] = None,
):
    """
    初始化检索器（支持 Dense 和 BM25）
    
    Args:
        model_config: 模型配置字典
        device: 设备
        use_gpu_for_search: 是否使用 GPU 进行搜索
        
    Returns:
        检索器实例（DenseRetriever 或 BM25Retriever）
    """
    retriever_type = model_config.get('retriever_type', 'dense')
    
    logger.info(f"\n初始化 {model_config['model_name']} 检索器 (类型: {retriever_type})...")
    
    if retriever_type == 'bm25':
        try:
            from bm25_retriever import BM25Retriever
        except Exception as e:
            raise ImportError(f"无法导入 BM25Retriever (可能缺少 Java 环境或 pyserini): {e}")

        logger.info(f"  索引路径: {model_config['index_path']}")
        logger.info(f"  BM25 参数: k1={model_config.get('k1', 0.9)}, b={model_config.get('b', 0.4)}")
        logger.info(f"  BM25 线程数: {model_config.get('threads', 16)}")
        
        retriever = BM25Retriever(
            index_path=model_config['index_path'],
            k1=model_config.get('k1', 0.9),
            b=model_config.get('b', 0.4),
            threads=model_config.get('threads', 16),
        )
        
        stats = retriever.get_stats()
        logger.info(f"✓ 检索器初始化完成")
        logger.info(f"  总文档数: {stats['total_documents']:,}")
        logger.info(f"  检索器类型: {stats['retriever_type']}")
    else:
        logger.info(f"  模型路径: {model_config['model_path']}")
        logger.info(f"  索引路径: {model_config['index_path']}")
        logger.info(f"  元数据路径: {model_config['metadata_path']}")
        
        retriever = DenseRetriever(
            encoder_type=model_config['encoder_type'],
            model_path=model_config['model_path'],
            index_path=model_config['index_path'],
            metadata_path=model_config['metadata_path'],
            device=device,
            use_gpu_for_search=use_gpu_for_search,
            encode_batch_size=encode_batch_size,
            faiss_nprobe=faiss_nprobe,
        )
        
        stats = retriever.get_stats()
        logger.info(f"✓ 检索器初始化完成")
        logger.info(f"  总文档数: {stats['total_documents']:,}")
        logger.info(f"  向量维度: {stats['embedding_dim']}")
        logger.info(f"  索引类型: {stats['index_type']}")
    
    return retriever


def retrieve_top50(
    question_matches: List[Dict],
    qrels: Dict[str, List[str]],
    corpus: Dict[str, str],
    retriever,
    top_k: int = 50,
    batch_size: int = 32,
    output_file: str = None
) -> List[Dict]:
    """
    对每个 beir_qid 检索 TOP50 文档
    
    Args:
        question_matches: 问题匹配列表
        qrels: query-id -> corpus-id 映射
        corpus: corpus-id -> 文档内容 映射
        retriever: 检索器实例
        top_k: 返回 top-k 结果
        batch_size: 批处理大小
        output_file: 输出文件路径
        
    Returns:
        结果列表
    """
    print("\n" + "=" * 80)
    print("开始检索 TOP50 文档")
    print("=" * 80)
    
    results = []
    batch_queries = []
    batch_metadata = []
    
    # 统计信息
    total_queries = 0
    missing_qrels = 0
    missing_corpus = 0
    
    for item in tqdm(question_matches, desc="处理查询"):
        beir_qid = item['beir_qid']
        
        # 获取相关的 corpus-ids
        if beir_qid not in qrels:
            missing_qrels += 1
            continue
        
        corpus_ids = qrels[beir_qid]
        
        # 对每个 corpus-id 进行检索
        for corpus_id in corpus_ids:
            if corpus_id not in corpus:
                missing_corpus += 1
                continue
            
            doc_text = corpus[corpus_id]
            
            # 🔧 修复：为 E5 模型添加查询前缀
            # 检查 retriever 是否有 encoder_type 属性
            if hasattr(retriever, 'encoder_type') and 'e5' in retriever.encoder_type.lower():
                query_text = "query: " + doc_text
            else:
                query_text = doc_text
            
            batch_queries.append(query_text)
            batch_metadata.append({
                'beir_qid': beir_qid,
                'corpus_id': corpus_id,
                'flashrag_id': item.get('flashrag_id', ''),
                'flashrag_question': item.get('flashrag_question', ''),
                'beir_question': item.get('beir_question', ''),
                'similarity': item.get('similarity', 0.0)
            })
            
            total_queries += 1
            
            # 当批次满了，进行检索
            if len(batch_queries) >= batch_size:
                batch_results = process_batch(
                    batch_queries, 
                    batch_metadata, 
                    retriever, 
                    top_k
                )
                results.extend(batch_results)
                
                # 如果指定了输出文件，实时写入
                if output_file:
                    write_batch_results(batch_results, output_file, mode='a')
                
                batch_queries = []
                batch_metadata = []
    
    # 处理剩余的批次
    if batch_queries:
        batch_results = process_batch(
            batch_queries, 
            batch_metadata, 
            retriever, 
            top_k
        )
        results.extend(batch_results)
        
        if output_file:
            write_batch_results(batch_results, output_file, mode='a')
    
    # 打印统计信息
    print("\n" + "=" * 80)
    print("检索完成统计")
    print("=" * 80)
    print(f"总查询数: {total_queries}")
    print(f"成功检索: {len(results)}")
    print(f"缺失 qrels: {missing_qrels}")
    print(f"缺失 corpus: {missing_corpus}")
    
    return results


def process_batch(
    batch_queries: List[str],
    batch_metadata: List[Dict],
    retriever,
    top_k: int
) -> List[Dict]:
    """
    处理一个批次的查询
    
    Args:
        batch_queries: 查询文本列表
        batch_metadata: 查询元数据列表
        retriever: 检索器实例
        top_k: 返回 top-k 结果
        
    Returns:
        结果列表
    """
    # 批量检索
    search_results = retriever.search(batch_queries, top_k=top_k, return_scores=True)
    
    # 构建结果
    results = []
    for i, metadata in enumerate(batch_metadata):
        top_docs = search_results[i]
        
        result = {
            **metadata,
            'top50_docs': [
                {
                    'doc_id': doc['doc_id'],
                    'score': doc['score'],
                    'rank': doc['rank']
                }
                for doc in top_docs
            ]
        }
        results.append(result)
    
    return results


def write_batch_results(results: List[Dict], output_file: str, mode: str = 'w'):
    """
    写入批次结果到文件
    
    Args:
        results: 结果列表
        output_file: 输出文件路径
        mode: 写入模式 ('w' 或 'a')
    """
    with open(output_file, mode, encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description='使用现有模型索引检索TOP50文档')
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.json',
        help='配置文件路径 (默认: config.json)'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='指定使用的模型 (e5, bge, bge_m3)，覆盖配置文件'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        help='返回 top-k 结果，覆盖配置文件'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help='批处理大小，覆盖配置文件'
    )
    parser.add_argument(
        '--encode-batch-size',
        type=int,
        help='向量编码 batch size（最关键的显存参数），覆盖配置文件'
    )
    parser.add_argument(
        '--faiss-nprobe',
        type=int,
        help='FAISS IVF nprobe 参数（速度/召回权衡），覆盖配置文件'
    )
    
    args = parser.parse_args()
    
    # 加载配置文件
    config_path = Path(args.config)
    if not config_path.is_absolute():
        script_dir = Path(__file__).parent
        config_path = script_dir / args.config
    
    if not config_path.exists():
        print(f"❌ 错误: 配置文件不存在: {config_path}")
        return
    
    print(f"加载配置文件: {config_path}")
    config = load_config(str(config_path))
    
    # 获取配置参数
    data_paths = config['data_paths']
    model_configs = config['model_configs']
    retrieval_config = config['retrieval_config']
    output_config = config['output_config']
    
    # 命令行参数覆盖配置文件
    top_k = args.top_k if args.top_k else retrieval_config['top_k']
    batch_size = args.batch_size if args.batch_size else retrieval_config['batch_size']
    encode_batch_size_cfg = retrieval_config.get('encode_batch_size', 128)
    faiss_nprobe = args.faiss_nprobe if args.faiss_nprobe is not None else retrieval_config.get('faiss_nprobe')
    
    # 确定使用哪些模型
    if args.model:
        selected_models = [args.model]
    else:
        selected_models = retrieval_config['selected_models']
    
    print("\n" + "=" * 80)
    print("检索配置")
    print("=" * 80)
    print(f"问题匹配文件: {data_paths['question_match_file']}")
    print(f"Qrels 文件: {data_paths['qrels_file']}")
    print(f"Corpus 文件: {data_paths['corpus_file']}")
    print(f"Top-K: {top_k}")
    print(f"Batch Size: {batch_size}")
    print(f"选择的模型: {', '.join(selected_models)}")
    print("=" * 80)
    
    # 创建输出目录
    output_dir = Path(output_config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置日志文件路径
    log_dir = output_dir / "log_retriever"
    log_dir.mkdir(parents=True, exist_ok=True)
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"retrieval_log_{current_time}.txt"

    # 配置 Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger.info("=" * 60)
    logger.info(f"开始新的检索任务 | 时间: {current_time}")
    logger.info(f"日志文件: {log_file}")
    logger.info("=" * 60)
    logger.info(f"系统初始状态: {get_system_status()}")
    
    # Step 1: 加载数据
    logger.info("\nStep 1: 加载数据")
    logger.info("-" * 40)
    
    logger.info(f"加载问题匹配文件: {data_paths['question_match_file']}")
    question_matches = load_question_matches(data_paths['question_match_file'])
    
    logger.info(f"加载 qrels 文件: {data_paths['qrels_file']}")
    qrels = load_qrels(data_paths['qrels_file'])
    
    logger.info(f"加载 BEIR corpus: {data_paths['corpus_file']}")
    corpus = load_corpus(data_paths['corpus_file'])

    logger.info(f"数据加载完成: Docs={len(corpus)}, Qrels={len(qrels)}, Matches={len(question_matches)}")
    logger.info(f"资源状态: {get_system_status()}")

    
    # Step 2: 对每个选择的模型进行检索
    for model_name in selected_models:
        if model_name not in model_configs:
            print(f"\n⚠️  警告: 模型 {model_name} 不在配置中，跳过")
            continue
        
        model_config = model_configs[model_name]
        
        if not model_config.get('enabled', True):
            print(f"\n⚠️  模型 {model_name} 未启用，跳过")
            continue
        
        logger.info("\n" + "=" * 80)
        logger.info(f"Step 2: 使用 {model_name} 模型检索")
        logger.info("=" * 80)
        
        # 初始化检索器
        try:
            if isinstance(encode_batch_size_cfg, dict):
                encode_batch_size = int(encode_batch_size_cfg.get(model_name, 128))
            else:
                encode_batch_size = int(encode_batch_size_cfg)

            if args.encode_batch_size is not None:
                encode_batch_size = int(args.encode_batch_size)

            logger.info(f"初始化检索器... (Encode Batch Size: {encode_batch_size})")
            retriever = initialize_retriever(
                model_config,
                device=retrieval_config['device'],
                use_gpu_for_search=retrieval_config['use_gpu_for_search'],
                encode_batch_size=encode_batch_size,
                faiss_nprobe=faiss_nprobe,
            )
        except Exception as e:
            logger.error(f"❌ 错误: 无法初始化 {model_name} 检索器: {e}")
            continue
        
        # 生成输出文件名
        output_file = output_dir / output_config['output_file_pattern'].format(
            model_name=model_name
        )
        
        # 清空输出文件（如果存在）
        if output_file.exists():
            output_file.unlink()
        
        logger.info(f"输出文件: {output_file}")
        logger.info(f"检索前资源状态: {get_system_status()}")
        
        # 检索
        start_time = time.time()
        results = retrieve_top50(
            question_matches=question_matches,
            qrels=qrels,
            corpus=corpus,
            retriever=retriever,
            top_k=top_k,
            batch_size=batch_size,
            output_file=str(output_file)
        )
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"\n✓ {model_name} 检索完成")
        logger.info(f"  耗时: {duration:.1f} 秒")
        logger.info(f"  结果数: {len(results)}")
        logger.info(f"  输出文件: {output_file}")
        logger.info(f"检索后资源状态: {get_system_status()}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ 所有检索任务完成!")
    logger.info("=" * 80)



if __name__ == '__main__':
    main()
