#!/usr/bin/env python3
"""
DeFRCN Training Monitor - GCP Instance
Monitors DeFRCN training progress for VizWiz few-shot detection
"""

import subprocess
import time
from datetime import datetime

print("🔴 LIVE DEFRCN TRAINING MONITOR")
print("="*70)
print("Press Ctrl+C to stop monitoring")
print("="*70)
print()

try:
    while True:
        # Clear screen (optional)
        print("\033[H\033[J", end="")  # ANSI clear screen
        
        print("🔴 LIVE DeFRCN Training Monitor (GCP)")
        print("="*70)
        print(f"⏰ Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        print()
        
        # Get process status
        result = subprocess.run(
            "ps aux | grep 'python.*main.py' | grep -v grep | head -1",
            shell=True, capture_output=True, text=True
        )
        
        if result.stdout.strip():
            parts = result.stdout.split()
            pid = parts[1]
            cpu = parts[2]
            mem = parts[3]
            runtime = parts[9]
            
            print(f"✅ Status: RUNNING")
            print(f"   PID: {pid} | CPU: {cpu}% | Memory: {mem}% | Time: {runtime}")
        else:
            print("❌ Status: NOT RUNNING")
            print("⚠️  Training may have stopped or not started yet")
        
        print()
        print("📊 Latest Training Progress:")
        print("-"*70)
        
        # Get last 5 training iterations
        result = subprocess.run(
            "tail -200 ~/DeFRCN/training_resume.log 2>/dev/null | grep 'd2.utils.events' | tail -5",
            shell=True, capture_output=True, text=True
        )
        
        if result.stdout.strip():
            for line in result.stdout.strip().split('\n'):
                # Parse DeFRCN log format: [timestamp d2.utils.events]:  eta: X:XX  iter: XXX  total_loss: X.XXX
                if 'eta:' in line and 'iter:' in line and 'total_loss:' in line:
                    parts = line.split()
                    
                    eta = iter_num = total_loss = "N/A"
                    
                    for i, part in enumerate(parts):
                        if 'eta:' == part and i+1 < len(parts):
                            eta = parts[i+1]
                        if 'iter:' == part and i+1 < len(parts):
                            iter_num = parts[i+1]
                        if 'total_loss:' == part and i+1 < len(parts):
                            total_loss = parts[i+1]
                    
                    print(f"   [Iter {iter_num}] | Loss: {total_loss} | ETA: {eta}")
        else:
            print("   ⏳ Waiting for training to start...")
        
        print()
        print("🎯 Training Configuration:")
        print("-"*70)
        print("   📁 Dataset: VizWiz OD25_1 (2275 base images, 75 classes)")
        print("   🎯 Stage: Base Training (3472 iterations)")
        print("   📊 Output: ~/DeFRCN/outputs_codetr_2x/OD25_1/")
        print("   🔗 W&B: defrcn_gcp_cursor")
        
        print()
        print("📝 Training Pipeline:")
        print("-"*70)
        print("   1. ✅ Base Training (3472 iter) - IN PROGRESS")
        print("   2. ⏳ Model Surgery (remove classification head)")
        print("   3. ⏳ Few-shot Fine-tuning (1-shot: 1400 iter)")
        print("   4. ⏳ Few-shot Fine-tuning (10-shot: 1400 iter)")
        print("   5. ⏳ Repeat for OD25_2 and OD25_3 folds")
        
        print()
        print("💡 Press Ctrl+C to stop monitoring (training continues)")
        print("="*70)
        
        # Update every 10 seconds
        time.sleep(10)
        
except KeyboardInterrupt:
    print("\n\n✅ Monitoring stopped. Training continues in background!")

