#!/bin/bash
# DeFRCN Complete Training Startup Script
# Starts training, W&B logger, and monitoring

set -e

echo "========================================"
echo "🚀 DeFRCN Training Startup"
echo "========================================"
echo ""

# Navigate to DeFRCN directory
cd "${HOME}/DeFRCN"

# 1. Stop any existing training
echo "1️⃣  Stopping any existing training..."
pkill -f 'python.*main.py' 2>/dev/null || true
pkill -f 'wandb_auto_logger_defrcn' 2>/dev/null || true
sleep 2
echo "✅ Stopped old processes"
echo ""

# 2. Clean outputs (optional)
if [ -d "outputs_defrcn" ]; then
    echo "2️⃣  Found existing outputs directory"
    ls -l outputs_defrcn/ | head -5
    echo ""
    read -p "Delete old outputs? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Backing up to outputs_defrcn_backup..."
        mv outputs_defrcn outputs_defrcn_backup_$(date +%Y%m%d_%H%M%S)
        echo "✅ Backed up old outputs"
    else
        echo "⚠️  Keeping existing outputs (will be overwritten)"
    fi
else
    echo "2️⃣  No previous outputs found"
fi
echo ""

# 3. Start W&B logger
echo "3️⃣  Starting W&B Auto-Logger..."
chmod +x wandb_auto_logger_defrcn.py
nohup python3 wandb_auto_logger_defrcn.py > wandb_logger_defrcn.log 2>&1 &
sleep 3

# Check if logger started
if ps aux | grep 'wandb_auto_logger_defrcn' | grep -v grep > /dev/null; then
    LOGGER_PID=$(ps aux | grep 'wandb_auto_logger_defrcn' | grep -v grep | awk '{print $2}')
    echo "✅ W&B Logger started (PID: $LOGGER_PID)"
else
    echo "❌ W&B Logger failed to start!"
    exit 1
fi
echo ""

# 4. Start training
echo "4️⃣  Starting DeFRCN Training..."
chmod +x train_vizwiz_fewshot_codetr_2x.sh
nohup bash train_vizwiz_fewshot_codetr_2x.sh > training_gcp_DFRCN.log 2>&1 &
sleep 5

# Check if training started
if ps aux | grep 'python.*main.py' | grep -v grep > /dev/null; then
    TRAIN_PID=$(ps aux | grep 'python.*main.py' | grep -v grep | awk '{print $2}' | head -1)
    echo "✅ Training started (PID: $TRAIN_PID)"
else
    echo "⚠️  Training initializing... (check log in 30 seconds)"
fi
echo ""

# 5. Summary
echo "========================================"
echo "✅ DeFRCN TRAINING STARTED!"
echo "========================================"
echo ""
echo "📊 Configuration:"
echo "   - Project: gcp_DFRCN"
echo "   - Folds: OD25_0, OD25_1, OD25_2, OD25_3"
echo "   - K-shots: 1, 3, 5, 10"
echo "   - Estimated time: ~8-10 hours"
echo ""
echo "📝 Log files:"
echo "   - Training: training_gcp_DFRCN.log"
echo "   - W&B Logger: wandb_logger_defrcn.log"
echo ""
echo "🔗 W&B Dashboard:"
echo "   https://wandb.ai/sturjem000-the-city-university-of-new-york/gcp_DFRCN"
echo ""
echo "📊 To monitor training:"
echo "   python3 monitor_defrcn_comprehensive.py"
echo ""
echo "   OR tail the log:"
echo "   tail -f training_gcp_DFRCN.log"
echo ""
echo "========================================"

