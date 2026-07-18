#!/usr/bin/env python3
"""
AUC-PR优化的融合实验

对每个retriever-scorer组合:
1. 在β∈{0.0,0.1,...,1.0}共11个值上扫描
2. 对每个β，计算完整PR曲线（θ_high从0到1）
3. 计算AUC-PR，选择最高的β作为最优融合权重
4. 保存最优β下的完整PR曲线数据和图表
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import logging
import sys

CONFIG_FILE = "config.json"


def load_config(config_path: str) -> Dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def setup_logging(log_file: str, level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def load_gold_docs(gold_file: Path) -> Dict[str, set]:
    """从TSV文件加载人工标注的gold标注

    格式: flashrag_qid\tbeir_qid\tbeir_docid\twiki_100w_id\tis_annotated\tRelevance
    Relevant doc: is_annotated > 0 且 Relevance == 1
    """
    gold_docs = {}
    with open(gold_file, 'r', encoding='utf-8') as f:
        header = f.readline()  # 跳过表头
        for line in f:
            line = line.strip().replace('\r', '')
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 6:
                continue
            flashrag_qid = parts[0]
            wiki_id = parts[3]  # wiki_100w_id 是文档ID
            is_annotated = float(parts[4]) if parts[4] else 0
            relevance = int(parts[5])

            if is_annotated > 0 and relevance == 1:
                if flashrag_qid not in gold_docs:
                    gold_docs[flashrag_qid] = set()
                gold_docs[flashrag_qid].add(wiki_id)

    return gold_docs


def load_retrieval_scores(retrieval_file: Path, logger) -> Dict[str, List[Dict]]:
    """加载检索分数，支持top200_docs和fused_docs两种格式"""
    qid_docs = defaultdict(list)
    with open(retrieval_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                qid = data.get('flashrag_id')
                # 兼容两种格式
                docs = data.get('top200_docs') or data.get('fused_docs', [])
                for doc in docs:
                    doc_id = str(doc.get('doc_id', ''))
                    score = float(doc.get('score', 0.0))
                    rank = int(doc.get('rank', 0))
                    if qid and doc_id:
                        qid_docs[qid].append({
                            'doc_id': doc_id,
                            'ret_score': score,
                            'rank': rank
                        })
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error at line {line_num}: {e}")
    logger.info(f"Loaded retrieval scores for {len(qid_docs)} queries")
    return qid_docs


def load_rerank_scores(rerank_file: Path, logger) -> Dict[str, Dict[str, float]]:
    """加载重排分数"""
    qid_doc_scores = defaultdict(dict)
    with open(rerank_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for item in data:
        qid = item.get('qid')
        reranked_docs = item.get('reranked_docs_with_scores', [])
        for doc in reranked_docs:
            doc_id = str(doc.get('doc_id', ''))
            score = float(doc.get('rerank_score', 0.0))
            if qid and doc_id:
                qid_doc_scores[qid][doc_id] = score
    logger.info(f"Loaded rerank scores for {len(qid_doc_scores)} queries")
    return qid_doc_scores


def fuse_scores(retrieval_docs: List[Dict], rerank_scores: Dict[str, float], beta: float) -> List[Dict]:
    """融合分数并排序"""
    fused = []
    for doc in retrieval_docs:
        doc_id = doc['doc_id']
        ret_score = doc['ret_score']
        rer_score = rerank_scores.get(doc_id, 0.0)
        fused_score = beta * ret_score + (1 - beta) * rer_score
        fused.append({
            'doc_id': doc_id,
            'fused_score': fused_score,
            'ret_score': ret_score,
            'rer_score': rer_score,
            'rank': doc['rank']
        })
    fused.sort(key=lambda x: x['fused_score'], reverse=True)
    for i, doc in enumerate(fused, 1):
        doc['new_rank'] = i
    return fused


def min_max_normalize(docs: List[Dict]) -> List[Dict]:
    """Min-max归一化"""
    if not docs:
        return docs
    scores = [doc['fused_score'] for doc in docs]
    min_score, max_score = min(scores), max(scores)
    if max_score == min_score:
        for doc in docs:
            doc['normalized_score'] = 0.5
    else:
        for doc in docs:
            doc['normalized_score'] = (doc['fused_score'] - min_score) / (max_score - min_score)
    return docs


def compute_pr_at_theta(docs: List[Dict], K: int, theta_high: float, gold: set) -> Tuple[float, float]:
    """计算某个theta_high下的Precision和Recall"""
    docs_k = docs[:K]
    docs_k = min_max_normalize(docs_k)

    relevant = set()
    for doc in docs_k:
        if doc['new_rank'] <= K and doc['normalized_score'] >= theta_high:
            relevant.add(doc['doc_id'])

    if len(relevant) == 0:
        precision = 0.0
    else:
        hits = len(relevant & gold)
        precision = hits / len(relevant)

    if len(gold) == 0:
        recall = 0.0
    else:
        hits = len(relevant & gold)
        recall = hits / len(gold)

    return precision, recall


def compute_pr_curve_at_beta(
    retrieval_data: Dict[str, List[Dict]],
    rerank_data: Dict[str, Dict[str, float]],
    gold_docs: Dict[str, set],
    beta: float,
    K: int,
    theta_low: float
) -> Tuple[List[float], List[float]]:
    """计算某个beta下的完整PR曲线

    Returns:
        (precisions, recalls) - 两个等长列表，101个点
    """
    theta_highs = [round(t, 2) for t in np.arange(0.00, 1.01, 0.01)]
    common_qids = set(retrieval_data.keys()) & set(rerank_data.keys()) & set(gold_docs.keys())

    precisions_all = []
    recalls_all = []

    for theta_high in theta_highs:
        p_sum, r_sum = 0.0, 0.0
        count = 0
        for qid in common_qids:
            gold = gold_docs[qid]
            retrieval_docs = retrieval_data[qid]
            rerank_scores = rerank_data[qid]
            fused = fuse_scores(retrieval_docs, rerank_scores, beta)
            p, r = compute_pr_at_theta(fused, K, theta_high, gold)
            p_sum += p
            r_sum += r
            count += 1
        precisions_all.append(p_sum / count if count > 0 else 0.0)
        recalls_all.append(r_sum / count if count > 0 else 0.0)

    return precisions_all, recalls_all


def compute_auc_pr(precisions: List[float], recalls: List[float]) -> float:
    """计算AUC-PR (precision-recall曲线下的面积)"""
    # PR曲线下面积，使用梯形法则
    # 需要按recall排序（从小到大）
    # 但我们的theta_high是从高到低扫描的，所以需要反转
    recalls_rev = recalls[::-1]
    precisions_rev = precisions[::-1]

    # 确保recall是递增的
    auc = 0.0
    for i in range(1, len(recalls_rev)):
        w = recalls_rev[i] - recalls_rev[i-1]
        h = (precisions_rev[i] + precisions_rev[i-1]) / 2
        auc += w * h
    return max(0.0, auc)


def run_optimization_for_combination(
    retriever: str,
    reranker: str,
    retrieval_file: Path,
    rerank_file: Path,
    gold_docs: Dict[str, set],
    config: Dict,
    logger
) -> Dict[str, Any]:
    """对单个组合运行AUC-PR优化"""
    K = config['parameters']['K']
    theta_low = config['parameters']['theta_low']
    beta_start = config['parameters']['beta_range']['start']
    beta_end = config['parameters']['beta_range']['end']
    beta_step = config['parameters']['beta_range']['step']

    betas = [round(b, 2) for b in np.arange(beta_start, beta_end + beta_step, beta_step)]
    theta_highs = [round(t, 2) for t in np.arange(0.00, 1.01, 0.01)]

    logger.info(f"Loading retrieval scores from {retrieval_file}")
    retrieval_data = load_retrieval_scores(retrieval_file, logger)
    logger.info(f"Loading rerank scores from {rerank_file}")
    rerank_data = load_rerank_scores(rerank_file, logger)

    common_qids = set(retrieval_data.keys()) & set(rerank_data.keys()) & set(gold_docs.keys())
    logger.info(f"Common queries: {len(common_qids)}")

    # 存储每个beta的PR曲线和AUC-PR
    beta_results = []

    for beta in betas:
        precisions, recalls = compute_pr_curve_at_beta(
            retrieval_data, rerank_data, gold_docs, beta, K, theta_low
        )
        auc_pr = compute_auc_pr(precisions, recalls)
        beta_results.append({
            'beta': beta,
            'precisions': precisions,
            'recalls': recalls,
            'auc_pr': auc_pr
        })
        logger.info(f"  β={beta:.1f}, AUC-PR={auc_pr:.6f}")

    # 找最优beta
    best_result = max(beta_results, key=lambda x: x['auc_pr'])

    return {
        'retriever': retriever,
        'reranker': reranker,
        'betas': betas,
        'theta_highs': theta_highs,
        'beta_results': beta_results,
        'best_beta': best_result['beta'],
        'best_auc_pr': best_result['auc_pr'],
        'best_precisions': best_result['precisions'],
        'best_recalls': best_result['recalls']
    }


def main():
    config = load_config(CONFIG_FILE)
    logger = setup_logging(config['logging']['log_file'], config['logging']['level'])

    logger.info("=" * 80)
    logger.info("AUC-PR Optimization for Score Fusion")
    logger.info("=" * 80)

    gold_docs = load_gold_docs(Path(config['input']['gold_file']))
    logger.info(f"Loaded {len(gold_docs)} queries with gold docs")

    output_dir = Path(config['output']['base_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建子文件夹保存PR曲线数据
    pr_curves_dir = output_dir / 'pr_curves_data'
    pr_curves_dir.mkdir(exist_ok=True)

    # 所有retriever和reranker组合
    retrievers = ['bm25', 'bge', 'e5', 'e5_large', 'fusion_interleaving']
    rerankers = ['monot5_base', 'cross_encoder', 'ms_marco_l12', 'nli', 'no_rerank', 'rouge_l']

    # 特殊文件路径映射
    retrieval_file_map = {
        'fusion_interleaving': Path("/home/asus/projects_hll/windsurf_projects/ATM_2080/all_scores_202603/stage1_retriever/fusion_results/fusion_bm25_bge_interleaving_top200.jsonl")
    }
    # rerank文件名映射 (retriever名称 -> rerank文件中的名称)
    rerank_name_map = {
        'e5': 'e5_small'  # e5检索器的rerank文件用的是e5_small
    }

    all_results = []
    best_combination = None
    best_auc_pr_global = -1.0

    total = len(retrievers) * len(rerankers)
    current = 0

    for retriever in retrievers:
        for reranker in rerankers:
            current += 1
            logger.info(f"\n[{current}/{total}] Processing: {retriever} + {reranker}")

            # 构建文件路径
            if retriever in retrieval_file_map:
                retrieval_file = retrieval_file_map[retriever]
            else:
                retrieval_file = Path(config['input']['retrieval_scores_dir']) / f"{retriever}_top200_results.jsonl"
            # 使用rerank名称映射
            rerank_name = rerank_name_map.get(retriever, retriever)
            rerank_file = Path(config['input']['rerank_results_dir']) / f"{rerank_name}_{reranker}_detailed_results.json"

            if not retrieval_file.exists():
                logger.warning(f"Retrieval file not found: {retrieval_file}, skipping")
                continue

            if not rerank_file.exists():
                logger.warning(f"Rerank file not found: {rerank_file}, skipping")
                continue

            try:
                result = run_optimization_for_combination(
                    retriever, reranker, retrieval_file, rerank_file, gold_docs, config, logger
                )

                all_results.append({
                    'retriever': result['retriever'],
                    'reranker': result['reranker'],
                    'best_beta': result['best_beta'],
                    'best_auc_pr': result['best_auc_pr'],
                    'theta_highs': result['theta_highs'],
                    'best_precisions': result['best_precisions'],
                    'best_recalls': result['best_recalls']
                })

                # 保存该组合的PR曲线数据
                pr_df = pd.DataFrame({
                    'theta_high': result['theta_highs'],
                    'precision': result['best_precisions'],
                    'recall': result['best_recalls']
                })
                pr_df.to_csv(pr_curves_dir / f"{retriever}_{reranker}_pr_curve.csv", index=False)

                # 保存该组合所有beta的AUC-PR值
                auc_df = pd.DataFrame([{
                    'beta': br['beta'],
                    'auc_pr': br['auc_pr']
                } for br in result['beta_results']])
                auc_df.to_csv(pr_curves_dir / f"{retriever}_{reranker}_auc_pr_by_beta.csv", index=False)

                if result['best_auc_pr'] > best_auc_pr_global:
                    best_auc_pr_global = result['best_auc_pr']
                    best_combination = f"{result['retriever']}_{result['reranker']}"

            except Exception as e:
                logger.error(f"Error processing {retriever} + {reranker}: {e}")

    # 创建汇总DataFrame
    summary_data = []
    for r in all_results:
        summary_data.append({
            'retriever': r['retriever'],
            'reranker': r['reranker'],
            'best_beta': r['best_beta'],
            'best_auc_pr': r['best_auc_pr']
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('best_auc_pr', ascending=False)
    summary_df = summary_df.reset_index(drop=True)
    summary_df['is_best'] = summary_df.apply(
        lambda row: f"{row['retriever']}_{row['reranker']}" == best_combination, axis=1
    )

    summary_file = output_dir / 'auc_pr_optimization_summary.csv'
    summary_df.to_csv(summary_file, index=False, encoding='utf-8')
    logger.info(f"\nSaved summary to {summary_file}")

    # 绘制最佳组合的PR曲线
    best_result = [r for r in all_results if f"{r['retriever']}_{r['reranker']}" == best_combination][0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # PR曲线
    ax1 = axes[0]
    ax1.plot(best_result['best_recalls'], best_result['best_precisions'], 'b-', linewidth=2)
    ax1.fill_between(best_result['best_recalls'], best_result['best_precisions'], alpha=0.2)
    ax1.set_xlabel('Recall', fontsize=12)
    ax1.set_ylabel('Precision', fontsize=12)
    ax1.set_title(f'Best PR Curve: {best_combination}\n(β={best_result["best_beta"]:.1f}, AUC-PR={best_result["best_auc_pr"]:.4f})', fontsize=11)
    ax1.grid(True, alpha=0.3)

    # F1 vs theta_high曲线
    ax2 = axes[1]
    f1s = [2*p*r/(p+r) if (p+r) > 0 else 0 for p, r in zip(best_result['best_precisions'], best_result['best_recalls'])]
    ax2.plot(best_result['theta_highs'], f1s, 'g-', linewidth=2)
    best_idx = np.argmax(f1s)
    ax2.axvline(best_result['theta_highs'][best_idx], color='red', linestyle='--', alpha=0.7)
    ax2.scatter([best_result['theta_highs'][best_idx]], [f1s[best_idx]], color='red', s=100, zorder=5)
    ax2.set_xlabel('θ_high', fontsize=12)
    ax2.set_ylabel('F1 Score', fontsize=12)
    ax2.set_title(f'F1 vs θ_high (Best θ={best_result["theta_highs"][best_idx]:.2f}, F1={f1s[best_idx]:.4f})', fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'best_combination_pr_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved best combination PR curve")

    # 绘制Top 10组合的PR曲线对比
    top10 = summary_df.head(10)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for i, (_, row) in enumerate(top10.iterrows()):
        combo_name = f"{row['retriever']}_{row['reranker']}"
        result = [r for r in all_results if f"{r['retriever']}_{r['reranker']}" == combo_name][0]
        label = f"{combo_name} (β={row['best_beta']:.1f}, AUC={row['best_auc_pr']:.4f})"
        ax.plot(result['best_recalls'], result['best_precisions'],
                color=colors[i], label=label, linewidth=1.5)

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Top 10 Combinations - PR Curves at Optimal β', fontsize=12)
    ax.legend(loc='lower left', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'top10_pr_curves_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved top 10 PR curves comparison")

    # 生成报告
    report = []
    report.append("=" * 80)
    report.append("AUC-PR OPTIMIZATION REPORT")
    report.append("=" * 80)
    report.append("")
    report.append(f"Parameters: K={config['parameters']['K']}, θ_low={config['parameters']['theta_low']}")
    report.append(f"β range: {config['parameters']['beta_range']['start']} to {config['parameters']['beta_range']['end']}, step={config['parameters']['beta_range']['step']}")
    report.append(f"θ_high range: {config['parameters']['theta_high_range']['start']} to {config['parameters']['theta_high_range']['end']}, step={config['parameters']['theta_high_range']['step']}")
    report.append(f"Total combinations evaluated: {len(all_results)}")
    report.append("")
    report.append("-" * 80)
    report.append("SUMMARY TABLE (sorted by best AUC-PR, descending)")
    report.append("-" * 80)
    report.append(f"{'Retriever':<20} {'Reranker':<15} {'Best β':<10} {'AUC-PR':<15} {'Best'}")
    report.append("-" * 80)

    for _, row in summary_df.iterrows():
        is_best = " ★ BEST" if row['is_best'] else ""
        report.append(f"{row['retriever']:<20} {row['reranker']:<15} {row['best_beta']:<10.1f} {row['best_auc_pr']:<15.6f}{is_best}")

    report.append("")
    report.append("-" * 80)
    report.append("BEST PIPELINE CONFIGURATION")
    report.append("-" * 80)
    best_row = summary_df[summary_df['is_best']].iloc[0]
    report.append(f"Combination: {best_row['retriever']} + {best_row['reranker']}")
    report.append(f"Optimal β: {best_row['best_beta']:.1f}")
    report.append(f"AUC-PR: {best_row['best_auc_pr']:.6f}")

    report_text = "\n".join(report)
    with open(output_dir / 'auc_pr_optimization_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)

    logger.info(f"\nSaved report to {output_dir / 'auc_pr_optimization_report.txt'}")

    print("\n" + "=" * 80)
    print("BEST PIPELINE CONFIGURATION")
    print("=" * 80)
    print(f"Combination: {best_row['retriever']} + {best_row['reranker']}")
    print(f"Optimal β: {best_row['best_beta']:.1f}")
    print(f"AUC-PR: {best_row['best_auc_pr']:.6f}")
    print("\n" + "=" * 80)
    print("TOP 10 COMBINATIONS")
    print("=" * 80)
    print(summary_df.head(10).to_string(index=False))
    print(f"\nAll PR curve data saved to {pr_curves_dir}")


if __name__ == '__main__':
    main()