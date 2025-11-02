#!/usr/bin/env python3
"""
Comprehensive DeFRCN Live Monitor
Shows training progress, W&B status, and fold completion
"""

import subprocess
import time
import sys
from datetime import datetime

def clear_screen():
    print("\033[2J\033[H", end="")

def get_process_status():
    """Check if training is running"""
    result = subprocess.run(
        "ps aux | grep 'python.*main.py' | grep -v grep",
        shell=True, capture_output=True, text=True
    )
    return result.stdout.strip()

def get_wandb_status():
    """Check if W&B logger is running"""
    result = subprocess.run(
        "ps aux | grep 'wandb_auto_logger_defrcn' | grep -v grep",
        shell=True, capture_output=True, text=True
    )
    return result.stdout.strip()

def get_current_fold_stage():
    """Detect current fold and stage from running process"""
    result = subprocess.run(
        "ps aux | grep 'python.*main.py' | grep -v grep",
        shell=True, capture_output=True, text=True
    )
    
    if not result.stdout.strip():
        return None, None
    
    # Look for fold and stage in command line
    cmd = result.stdout
    fold = None
    stage = None
    
    if 'OD25_0' in cmd:
        fold = 'OD25_0'
    elif 'OD25_1' in cmd:
        fold = 'OD25_1'
    elif 'OD25_2' in cmd:
        fold = 'OD25_2'
    elif 'OD25_3' in cmd:
        fold = 'OD25_3'
    
    if 'train_base' in cmd or 'base_model' in cmd:
        stage = 'base'
    elif '1shot' in cmd:
        stage = '1shot'
    elif '3shot' in cmd:
        stage = '3shot'
    elif '5shot' in cmd:
        stage = '5shot'
    elif '10shot' in cmd:
        stage = '10shot'
    
    return fold, stage

def get_latest_metrics():
    """Get latest training metrics from log"""
    result = subprocess.run(
        "find outputs_defrcn -name 'log.txt' -type f -printf '%T@ %p\\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-",
        shell=True, capture_output=True, text=True
    )
    
    if not result.stdout.strip():
        return []
    
    log_file = result.stdout.strip()
    result = subprocess.run(
        f"tail -50 '{log_file}' 2>/dev/null | grep 'iter:' | tail -3",
        shell=True, capture_output=True, text=True
    )
    
    return result.stdout.strip().split('\n') if result.stdout.strip() else []

def get_fold_completion():
    """Check which folds/stages are complete"""
    folds = ['OD25_0', 'OD25_1', 'OD25_2', 'OD25_3']
    status = {}
    
    for fold in folds:
        status[fold] = {
            'base': False,
            '1shot': False,
            '3shot': False,
            '5shot': False,
            '10shot': False
        }
        
        # Check base
        result = subprocess.run(
            f"test -f outputs_defrcn/{fold}/base_model/model_final.pth && echo 'YES' || echo 'NO'",
            shell=True, capture_output=True, text=True
        )
        status[fold]['base'] = result.stdout.strip() == 'YES'
        
        # Check few-shots
        for k in [1, 3, 5, 10]:
            result = subprocess.run(
                f"test -f outputs_defrcn/{fold}/novel_{k}shot/model_final.pth && echo 'YES' || echo 'NO'",
                shell=True, capture_output=True, text=True
            )
            status[fold][f'{k}shot'] = result.stdout.strip() == 'YES'
    
    return status

def monitor():
    """Main monitoring loop"""
    print("🔴 LIVE DeFRCN Training Monitor")
    print("=" * 80)
    print("Press Ctrl+C to stop monitoring (training continues)")
    print("=" * 80)
    print()
    time.sleep(2)
    
    try:
        while True:
            clear_screen()
            
            print("🔴 LIVE DeFRCN Multi-Fold Training Monitor (4 Folds x 5 Stages)")
            print("=" * 80)
            print(f"⏰ Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)
            print()
            
            # 1. Training Process Status
            process_status = get_process_status()
            if process_status:
                parts = process_status.split()
                print(f"✅ Training: RUNNING")
                print(f"   PID: {parts[1]} | CPU: {parts[2]}% | Memory: {parts[3]}% | Time: {parts[9]}")
            else:
                print("⚠️  Training: NOT RUNNING (may be between stages)")
            
            # 2. W&B Logger Status
            wandb_status = get_wandb_status()
            if wandb_status:
                parts = wandb_status.split()
                print(f"✅ W&B Logger: RUNNING (PID: {parts[1]})")
            else:
                print("❌ W&B Logger: NOT RUNNING")
            
            print()
            print("📂 Current Progress:")
            print("-" * 80)
            
            # 3. Current fold/stage
            fold, stage = get_current_fold_stage()
            if fold and stage:
                print(f"   Currently training: {fold} / {stage}")
            else:
                print("   ⏳ Waiting for training to start or between stages...")
            
            print()
            print("📊 Latest Training Iterations:")
            print("-" * 80)
            
            # 4. Latest metrics
            metrics = get_latest_metrics()
            if metrics and metrics[0]:
                for line in metrics:
                    if line.strip():
                        print(f"   {line}")
            else:
                print("   ⏳ No training metrics yet...")
            
            print()
            print("🎯 Fold Completion Status:")
            print("-" * 80)
            
            # 5. Fold completion
            completion = get_fold_completion()
            for fold, stages in completion.items():
                base_mark = "✅" if stages['base'] else "⏳"
                shots_done = [k for k, v in stages.items() if k != 'base' and v]
                shot_status = f"✅[{','.join(shots_done)}]" if shots_done else "⏳"
                print(f"   {fold}: Base {base_mark} | Few-shot {shot_status}")
            
            print()
            print("-" * 80)
            print("🔗 W&B Dashboard:")
            print("   https://wandb.ai/sturjem000-the-city-university-of-new-york/gcp_DFRCN")
            print()
            print("📝 Training Details:")
            print("   ✅ Pipeline: 4 folds (OD25_0, 1, 2, 3)")
            print("   ✅ Each fold: Base + 1/3/5/10-shot")
            print("   ✅ Backbone: ResNet101")
            print("   ⏱️  Total time: ~8-10 hours")
            print()
            print("💡 Press Ctrl+C to stop monitoring (training + W&B continue)")
            
            # Update every 15 seconds
            time.sleep(15)
            
    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped. Training + W&B logging continue in background!")
        sys.exit(0)

if __name__ == "__main__":
    monitor()

