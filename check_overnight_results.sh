#!/bin/bash

# Quick morning check for DeFRCN results

echo "========================================================================"
echo "🌅 DeFRCN Overnight Results Check"
echo "========================================================================"
echo ""

# Check if training is still running
if ps aux | grep "python.*train_net.py" | grep -v grep > /dev/null; then
    echo "⏳ Status: STILL RUNNING"
    echo ""
    echo "Current progress:"
    tail -100 ~/DeFRCN/training_gcp_DFRCN.log | grep -E "(STARTING|complete|FOLD)" | tail -5
else
    echo "✅ Status: COMPLETED (or stopped)"
fi

echo ""
echo "========================================================================"
echo "📊 Completed Stages:"
echo "========================================================================"
echo ""

for FOLD in OD25_0 OD25_1 OD25_2 OD25_3; do
    echo "🗂️  $FOLD:"
    
    # Check base training
    if [ -f ~/DeFRCN/outputs_defrcn/$FOLD/base/model_final.pth ]; then
        echo "  ✅ Base training complete"
    else
        echo "  ⏳ Base training incomplete"
    fi
    
    # Check novel training
    for K in 1 3 5 10; do
        if [ -f ~/DeFRCN/outputs_defrcn/$FOLD/novel_${K}shot/model_final.pth ]; then
            echo "  ✅ ${K}-shot complete"
        else
            echo "  ⏳ ${K}-shot incomplete"
        fi
    done
    echo ""
done

echo "========================================================================"
echo "🎯 Latest mAP Values (THESE WILL BE WRONG DUE TO SPARSE IDs):"
echo "========================================================================"
echo ""
grep "copypaste:" ~/DeFRCN/training_gcp_DFRCN.log | tail -20

echo ""
echo "========================================================================"
echo "🔗 W&B Dashboard:"
echo "https://wandb.ai/sturjem000-the-city-university-of-new-york/gcp_DFRCN"
echo "========================================================================"
echo ""
echo "⚠️  IMPORTANT: mAP values are WRONG due to sparse category IDs"
echo "    The models ARE learning, but evaluation is broken"
echo "    We'll fix this in the morning and re-evaluate!"
echo ""
