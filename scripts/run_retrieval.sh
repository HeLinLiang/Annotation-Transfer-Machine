#!/bin/bash
# 快速启动脚本 - 使用现有模型索引检索TOP50文档

echo "=========================================="
echo "Retriever TOP50 - 多模型检索系统"
echo "=========================================="
echo ""

# 检查 Python 环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到 python3"
    exit 1
fi

# 切换到脚本所在目录
cd "$(dirname "$0")"

# 显示使用帮助
show_help() {
    echo "使用方法:"
    echo "  ./run_retrieval.sh [选项]"
    echo ""
    echo "选项:"
    echo "  --model MODEL      指定模型 (e5, bge, bge_m3)"
    echo "  --top-k K          返回 top-k 结果 (默认: 50)"
    echo "  --batch-size N     批处理大小 (默认: 32)"
    echo "  --config FILE      配置文件路径 (默认: config.json)"
    echo "  --help             显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  ./run_retrieval.sh                    # 使用默认配置"
    echo "  ./run_retrieval.sh --model e5         # 只使用 E5 模型"
    echo "  ./run_retrieval.sh --top-k 100        # 检索 TOP100"
    echo ""
}

# 解析命令行参数
ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_help
            exit 0
            ;;
        --model|--top-k|--batch-size|--config)
            ARGS="$ARGS $1 $2"
            shift 2
            ;;
        *)
            echo "❌ 未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 运行检索脚本
echo "启动检索任务..."
echo "Python environment: ../miniconda/bin/python"
../miniconda/bin/python --version
echo ""
../miniconda/bin/python retrieve_top50.py $ARGS

# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ 检索任务完成!"
    echo "=========================================="
    echo ""
    echo "结果文件位于: results_2026_TOP50/"
    echo ""
    echo "查看结果:"
    echo "  head -5 results_2026_TOP50/e5_top50_results.jsonl"
    echo "  head -5 results_2026_TOP50/bge_top50_results.jsonl"
else
    echo ""
    echo "=========================================="
    echo "❌ 检索任务失败"
    echo "=========================================="
    exit 1
fi
