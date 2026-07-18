#!/bin/bash
# 监视重排进度脚本

LOG_FILE="results/reranked_top200/rerank_evaluation_top200.log"
RESULT_DIR="results/reranked_top200"

echo "=========================================="
echo "重排任务进度监视"
echo "=========================================="
echo ""

# 检查进程是否在运行
if ps aux | grep -E "rerank_and_evaluate" | grep -v grep > /dev/null; then
    echo "✅ 重排任务正在运行中"
    echo ""
else
    echo "❌ 重排任务未运行"
    echo ""
    exit 1
fi

# 显示最新的日志
echo "📋 最新日志（最后20行）："
echo "----------------------------------------"
tail -20 "$LOG_FILE"
echo ""

# 统计已完成的重排器
echo "📊 已完成的重排结果："
echo "----------------------------------------"
ls -1 "$RESULT_DIR"/bge_bm25_mixed_*_report.txt 2>/dev/null | while read file; do
    basename "$file" | sed 's/bge_bm25_mixed_//' | sed 's/_report.txt//'
done
echo ""

# 显示文件数量
total_files=$(ls -1 "$RESULT_DIR"/bge_bm25_mixed_* 2>/dev/null | wc -l)
echo "📁 已生成文件数：$total_files"
echo ""

# 显示最近修改的文件
echo "🕐 最近修改的文件（最新5个）："
echo "----------------------------------------"
ls -lht "$RESULT_DIR"/bge_bm25_mixed_* 2>/dev/null | head -5 | awk '{print $9, "(" $6, $7, $8 ")"}'
echo ""

echo "=========================================="
echo "提示：运行 'bash monitor_rerank_progress.sh' 查看最新进度"
echo "=========================================="
