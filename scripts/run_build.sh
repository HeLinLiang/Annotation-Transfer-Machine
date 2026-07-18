#!/bin/bash
# FAISS 索引构建快速启动脚本（建议在 screen 中运行）

set -e

echo "=========================================="
echo "FAISS Dense Retrieval Index Builder"
echo "=========================================="

# 检查 GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Info:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
fi

# 定义 Python 解释器路径
PYTHON_EXEC="../miniconda/bin/python"

# 检查 Python 环境
echo "Python path: $PYTHON_EXEC"
echo "Python version:"
$PYTHON_EXEC --version
echo ""

# 检查依赖
echo "Checking dependencies..."
$PYTHON_EXEC -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
$PYTHON_EXEC -c "import sentence_transformers; print(f'SentenceTransformers: {sentence_transformers.__version__}')"
$PYTHON_EXEC -c "import faiss; print(f'FAISS: {faiss.__version__}, GPU: {faiss.get_num_gpus()}')"
echo ""

# 选择构建模式
echo "Select build mode:"
echo "1) E5 small only"
echo "2) E5 large only"
echo "3) BGE only"
echo "4) All models (default)"
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        ENCODER="e5"
        ;;
    2)
        ENCODER="e5_large"
        ;;
    3)
        ENCODER="bge"
        ;;
    *)
        ENCODER="all"
        ;;
esac

echo ""
echo "Building $ENCODER index(es)..."
echo "This will take approximately 4-7 hours for 21M documents"
echo "Press Ctrl+C to interrupt (progress will be saved)"
echo ""

# 运行构建
$PYTHON_EXEC build_index.py --encoder $ENCODER

echo ""
echo "=========================================="
echo "Build completed!"
echo "=========================================="
