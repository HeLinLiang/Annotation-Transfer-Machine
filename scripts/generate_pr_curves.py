#!/usr/bin/env python3
"""
使用最优β生成PR曲线，提取操作点数据

对每个组合:
1. 使用融合阶段确定的最优β
2. 固定融合得分，扫描θ_high从0到1得到完整PR曲线
3. 提取操作点: P=0.8,0.7,0.6,0.5时的R值，以及R=0.5时的P值
4. 计算每个操作点下平均Relevant文档数
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("/home/asus/projects_hll/windsurf_projects/ATM_2080/all_scores_202603/stage4_jueceshuchu/以AUC_PR为优化目标的输出")
GOLD_FILE = "/home/asus/projects_hll/windsurf_projects/ATM_2080/retriever_top50_3models/results_2026_TOP50/biaozhu_results/gold_doc.tsv"
RETRIEVAL_DIR = Path("/home/asus/projects_hll/windsurf_projects/ATM_2080/all_scores_202603/stage1_retriever/normalize_202603")
RERANK_DIR = Path("/home/asus/projects_hll/windsurf_projects/ATM_2080/all_scores_202603/stage2_dafen")
FUSION_RESULTS_DIR = Path("/home/asus/projects_hll/windsurf_projects/ATM_2080/all_scores_202603/stage3_ronghe/detail_results")

# 最优β值（从融合阶段确定）
OPTIMAL_BETAS = {
    ('bm25', 'monot5_base'): 0.8,
    ('bm25', 'ms_marco_l12'): 0.8,
    ('bm25', 'cross_encoder'): 0.9,
    ('bm25', 'rouge_l'): 0.5,
    ('bm25', 'nli'): 1.0,
    ('bm25', 'no_rerank'): 1.0,
    ('bge', 'monot5_base'): 0.8,
    ('bge', 'ms_marco_l12'): 0.8,
    ('bge', 'cross_encoder'): 0.9,
    ('bge', 'rouge_l'): 0.4,
    ('bge', 'nli'): 1.0,
    ('bge', 'no_rerank'): 1.0,
    ('e5_small', 'monot5_base'): 0.0,
    ('e5_small', 'ms_marco_l12'): 0.3,
    ('e5_small', 'cross_encoder'): 0.7,
    ('e5_small', 'rouge_l'): 0.1,
    ('e5_small', 'nli'): 1.0,
    ('e5_small', 'no_rerank'): 1.0,
    ('e5_large', 'monot5_base'): 0.0,
    ('e5_large', 'ms_marco_l12'): 0.1,
    ('e5_large', 'cross_encoder'): 0.8,
    ('e5_large', 'rouge_l'): 0.1,
    ('e5_large', 'nli'): 1.0,
    ('e5_large', 'no_rerank'): 1.0,
    ('fusion_interleaving', 'monot5_base'): 0.5,
    ('fusion_interleaving', 'ms_marco_l12'): 0.4,
    ('fusion_interleaving', 'cross_encoder'): 0.8,
    ('fusion_interleaving', 'rouge_l'): 0.2,
    ('fusion_interleaving', 'nli'): 0.9,
    ('fusion_interleaving', 'no_rerank'): 0.8,
}

# 特殊文件路径映射
RETRIEVAL_FILE_MAP = {
    'fusion_interleaving': Path("/home/asus/projects_hll/windsurf_projects/ATM_2080/all_scores_202603/stage1_retriever/fusion_results/fusion_bm25_bge_interleaving_top200.jsonl")
}
RERANK_NAME_MAP = {
    'e5': 'e5_small'
}


def load_gold_docs() -> Dict[str, set]:
    """加载gold标注"""
    gold_docs = {}
    with open(GOLD_FILE, 'r', encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            line = line.strip().replace('\r', '')
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 6:
                continue
            flashrag_qid = parts[0]
            wiki_id = parts[3]
            is_annotated = float(parts[4]) if parts[4] else 0
            relevance = int(parts[5])
            if is_annotated > 0 and relevance == 1:
                if flashrag_qid not in gold_docs:
                    gold_docs[flashrag_qid] = set()
                gold_docs[flashrag_qid].add(wiki_id)
    logger.info(f"Loaded gold docs for {len(gold_docs)} queries")
    return gold_docs


def load_fusion_scores(combo_name: str) -> Dict[str, List[Dict]]:
    """加载融合结果分数

    Returns:
        {qid: [{doc_id, score, rank}, ...]}
    """
    file_path = FUSION_RESULTS_DIR / f"fusion_{combo_name}.jsonl"
    qid_docs = defaultdict(list)

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                qid = data.get('flashrag_id')
                docs = data.get('top200_docs', [])

                for doc in docs:
                    doc_id = str(doc.get('doc_id', ''))
                    score = float(doc.get('score', 0.0))
                    rank = int(doc.get('rank', 0))
                    if qid and doc_id:
                        qid_docs[qid].append({
                            'doc_id': doc_id,
                            'score': score,
                            'rank': rank
                        })
            except json.JSONDecodeError:
                pass

    logger.info(f"Loaded {len(qid_docs)} queries from {combo_name}")
    return qid_docs


def min_max_normalize(docs: List[Dict]) -> List[Dict]:
    """Min-max归一化"""
    if not docs:
        return docs
    scores = [doc['score'] for doc in docs]
    min_score, max_score = min(scores), max(scores)
    if max_score == min_score:
        for doc in docs:
            doc['normalized_score'] = 0.5
    else:
        for doc in docs:
            doc['normalized_score'] = (doc['score'] - min_score) / (max_score - min_score)
    return docs


def compute_pr_and_count_at_theta(
    docs: List[Dict],
    K: int,
    theta_high: float,
    gold: set
) -> Tuple[float, float, int]:
    """计算某个theta_high下的Precision、Recall和Relevant文档数

    Returns:
        (precision, recall, relevant_count)
    """
    docs_k = docs[:K]
    docs_k = min_max_normalize(docs_k)

    relevant = set()
    for doc in docs_k:
        if doc['new_rank'] <= K and doc['normalized_score'] >= theta_high:
            relevant.add(doc['doc_id'])

    relevant_count = len(relevant)

    if relevant_count == 0:
        precision = 0.0
    else:
        hits = len(relevant & gold)
        precision = hits / relevant_count

    if len(gold) == 0:
        recall = 0.0
    else:
        hits = len(relevant & gold)
        recall = hits / len(gold)

    return precision, recall, relevant_count


def compute_pr_curve(
    fusion_data: Dict[str, List[Dict]],
    gold_docs: Dict[str, set],
    K: int,
    theta_low: float
) -> Tuple[List[float], List[float], List[int], List[float]]:
    """计算完整PR曲线

    Returns:
        (precisions, recalls, relevant_counts, theta_highs)
    """
    theta_highs = [round(t, 2) for t in np.arange(0.00, 1.01, 0.01)]
    common_qids = set(fusion_data.keys()) & set(gold_docs.keys())

    precisions_all = []
    recalls_all = []
    relevant_counts_all = []

    for theta_high in theta_highs:
        p_sum, r_sum, c_sum = 0.0, 0.0, 0
        count = 0

        for qid in common_qids:
            gold = gold_docs[qid]
            docs = fusion_data[qid].copy()

            # 按rank排序
            docs.sort(key=lambda x: x['rank'])
            for i, doc in enumerate(docs, 1):
                doc['new_rank'] = i

            p, r, c = compute_pr_and_count_at_theta(docs, K, theta_high, gold)
            p_sum += p
            r_sum += r
            c_sum += c
            count += 1

        precisions_all.append(p_sum / count if count > 0 else 0.0)
        recalls_all.append(r_sum / count if count > 0 else 0.0)
        relevant_counts_all.append(c_sum / count if count > 0 else 0.0)

    return precisions_all, recalls_all, relevant_counts_all, theta_highs


def find_operation_points(
    precisions: List[float],
    recalls: List[float],
    relevant_counts: List[int],
    theta_highs: List[float]
) -> Dict[str, Any]:
    """提取操作点数据

    Returns:
        dict containing operation point data
    """
    # 找到最接近目标precision的recall
    target_precisions = [0.8, 0.7, 0.6, 0.5]
    results = {}

    for target_p in target_precisions:
        # 找到所有precision >= target_p的点
        candidates = [(i, p, r, c, t) for i, (p, r, c, t) in
                     enumerate(zip(precisions, recalls, relevant_counts, theta_highs))
                     if p >= target_p]
        if candidates:
            # 选择recall最高的
            best = max(candidates, key=lambda x: x[2])
            results[f'P{target_p}'] = {
                'idx': best[0],
                'precision': best[1],
                'recall': best[2],
                'relevant_count': best[3],
                'theta_high': best[4]
            }
        else:
            results[f'P{target_p}'] = None

    # 找到最接近recall=0.5的precision
    candidates = [(i, p, r, c, t) for i, (p, r, c, t) in
                 enumerate(zip(precisions, recalls, relevant_counts, theta_highs))
                 if r >= 0.5]
    if candidates:
        best = min(candidates, key=lambda x: abs(x[2] - 0.5))
        results['R0.5'] = {
            'idx': best[0],
            'precision': best[1],
            'recall': best[2],
            'relevant_count': best[3],
            'theta_high': best[4]
        }
    else:
        results['R0.5'] = None

    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gold_docs = load_gold_docs()
    K = 100
    theta_low = 0.2
    gold_avg = 17.81

    # 处理所有25个组合
    all_combos = [
        ('bm25', 'monot5_base'),
        ('bm25', 'ms_marco_l12'),
        ('bm25', 'cross_encoder'),
        ('bm25', 'rouge_l'),
        ('bm25', 'nli'),
        ('bm25', 'no_rerank'),
        ('bge', 'monot5_base'),
        ('bge', 'ms_marco_l12'),
        ('bge', 'cross_encoder'),
        ('bge', 'rouge_l'),
        ('bge', 'nli'),
        ('bge', 'no_rerank'),
        ('e5_small', 'monot5_base'),
        ('e5_small', 'ms_marco_l12'),
        ('e5_small', 'cross_encoder'),
        ('e5_small', 'rouge_l'),
        ('e5_small', 'nli'),
        ('e5_small', 'no_rerank'),
        ('e5_large', 'monot5_base'),
        ('e5_large', 'ms_marco_l12'),
        ('e5_large', 'cross_encoder'),
        ('e5_large', 'rouge_l'),
        ('e5_large', 'nli'),
        ('e5_large', 'no_rerank'),
        ('fusion_interleaving', 'monot5_base'),
        ('fusion_interleaving', 'ms_marco_l12'),
        ('fusion_interleaving', 'cross_encoder'),
        ('fusion_interleaving', 'rouge_l'),
        ('fusion_interleaving', 'nli'),
        ('fusion_interleaving', 'no_rerank'),
    ]

    top_combos = all_combos

    all_results = []
    all_curves = {}

    for retriever, reranker in top_combos:
        combo_name = f"{retriever}_{reranker}"
        logger.info(f"Processing: {combo_name}")

        beta = OPTIMAL_BETAS.get((retriever, reranker), 0.5)
        logger.info(f"  Using β={beta}")

        fusion_data = load_fusion_scores(combo_name)

        precisions, recalls, relevant_counts, theta_highs = compute_pr_curve(
            fusion_data, gold_docs, K, theta_low
        )

        all_curves[combo_name] = {
            'theta_highs': theta_highs,
            'precisions': precisions,
            'recalls': recalls,
            'relevant_counts': relevant_counts,
            'beta': beta
        }

        # 保存PR曲线数据
        curve_df = pd.DataFrame({
            'theta_high': theta_highs,
            'precision': precisions,
            'recall': recalls,
            'relevant_count': relevant_counts
        })
        curve_df.to_csv(OUTPUT_DIR / f'{combo_name}_pr_curve.csv', index=False)

        # 提取操作点
        op_points = find_operation_points(precisions, recalls, relevant_counts, theta_highs)

        result = {
            'retriever': retriever,
            'reranker': reranker,
            'beta': beta
        }

        for target_p in [0.8, 0.7, 0.6, 0.5]:
            key = f'P{target_p}'
            if op_points[key]:
                result[f'{key}_recall'] = op_points[key]['recall']
                result[f'{key}_relevant_count'] = op_points[key]['relevant_count']
                result[f'{key}_theta_high'] = op_points[key]['theta_high']
            else:
                result[f'{key}_recall'] = np.nan
                result[f'{key}_relevant_count'] = np.nan
                result[f'{key}_theta_high'] = np.nan

        if op_points['R0.5']:
            result['R0.5_precision'] = op_points['R0.5']['precision']
            result['R0.5_relevant_count'] = op_points['R0.5']['relevant_count']
            result['R0.5_theta_high'] = op_points['R0.5']['theta_high']
        else:
            result['R0.5_precision'] = np.nan
            result['R0.5_relevant_count'] = np.nan
            result['R0.5_theta_high'] = np.nan

        all_results.append(result)
        logger.info(f"  P0.8: R={result.get('P0.8_recall', 'N/A'):.4f}, "
                   f"P0.5: R={result.get('P0.5_recall', 'N/A'):.4f}, "
                   f"R0.5: P={result.get('R0.5_precision', 'N/A'):.4f}")

    # 保存汇总结果
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / 'operation_points_summary.csv', index=False)
    logger.info(f"\nSaved summary to {OUTPUT_DIR / 'operation_points_summary.csv'}")

    # 绘制PR曲线对比图 (分5组，每组5条线)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    combo_list = list(all_curves.items())
    n_groups = 5
    combos_per_group = 6

    for gi in range(n_groups):
        ax = axes[gi]
        start_idx = gi * combos_per_group
        end_idx = min(start_idx + combos_per_group, len(combo_list))
        group_combos = combo_list[start_idx:end_idx]

        colors = plt.cm.tab10(np.linspace(0, 1, len(group_combos)))
        for i, (combo_name, data) in enumerate(group_combos):
            beta = data['beta']
            label = f"{combo_name} (β={beta:.1f})"
            ax.plot(data['recalls'], data['precisions'],
                    color=colors[i], label=label, linewidth=1.5)

        ax.set_xlabel('Recall', fontsize=10)
        ax.set_ylabel('Precision', fontsize=10)
        ax.set_title(f'PR Curves (Group {gi+1})', fontsize=11)
        ax.legend(loc='lower left', fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    # 隐藏多余的子图
    axes[5].axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'all25_pr_curves_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved PR curves comparison")

    # 绘制每个组合的单独PR曲线
    for combo_name, data in all_curves.items():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # PR曲线
        ax1 = axes[0]
        ax1.plot(data['recalls'], data['precisions'], 'b-', linewidth=2)
        ax1.fill_between(data['recalls'], data['precisions'], alpha=0.2)
        ax1.set_xlabel('Recall', fontsize=11)
        ax1.set_ylabel('Precision', fontsize=11)
        ax1.set_title(f'{combo_name} PR Curve (β={data["beta"]:.1f})', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])

        # Relevant文档数曲线
        ax2 = axes[1]
        ax2.plot(data['theta_highs'], data['relevant_counts'], 'g-', linewidth=2)
        ax2.axhline(y=gold_avg, color='red', linestyle='--', label=f'Gold avg ({gold_avg})')
        ax2.set_xlabel('θ_high', fontsize=11)
        ax2.set_ylabel('Avg Relevant Count', fontsize=11)
        ax2.set_title(f'{combo_name} - Relevant Docs per Query', fontsize=11)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'{combo_name}_pr_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved {combo_name}_pr_curve.png")

    # 打印汇总
    print("\n" + "=" * 100)
    print("OPERATION POINTS SUMMARY")
    print("=" * 100)
    print(f"Gold标注平均值: {gold_avg} docs/query")
    print()
    print(results_df.to_string(index=False))

    return results_df


if __name__ == '__main__':
    main()