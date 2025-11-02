#!/usr/bin/env python3
"""
Auto-switching W&B logger for DeFRCN that tracks all folds and stages
"""
import wandb
import json
import time
import os
from pathlib import Path
import subprocess

def find_active_training_dirs(base_output_dir):
    """
    Find all active training directories by checking which have recent metrics.json
    """
    base_path = Path(base_output_dir)
    active_dirs = []
    
    # Search pattern: outputs_codetr_2x/OD25_*/*/metrics.json
    for metrics_file in base_path.glob("OD25_*/*/metrics.json"):
        # Check if file was modified in last 2 minutes (active training)
        if time.time() - metrics_file.stat().st_mtime < 120:  # 2 minutes
            output_dir = metrics_file.parent
            
            # Parse fold and stage from path
            parts = str(output_dir).split('/')
            fold = parts[-2]  # e.g., OD25_0
            stage = parts[-1]  # e.g., base_model, novel_1shot
            
            active_dirs.append({
                'path': output_dir,
                'metrics_file': metrics_file,
                'fold': fold,
                'stage': stage,
                'mtime': metrics_file.stat().st_mtime
            })
    
    return active_dirs

def get_run_name(fold, stage):
    """
    Generate W&B run name from fold and stage
    """
    # Map stage directory names to clean run names
    stage_map = {
        'base_model': 'base_FIXED',
        'novel_1shot': 'novel_1shot_FIXED',
        'novel_10shot': 'novel_10shot_FIXED',
    }
    
    stage_clean = stage_map.get(stage, stage)
    return f"{fold}_{stage_clean}"

def log_training_to_wandb(
    base_output_dir,
    project_name="defrcn_gpu_c",
    update_interval=30
):
    """
    Auto-switching W&B logger that tracks all active training stages
    """
    print("="*70)
    print("AUTO-SWITCHING W&B LOGGER FOR DeFRCN")
    print("="*70)
    print(f"Project: {project_name}")
    print(f"Monitoring: {base_output_dir}")
    print(f"Update interval: {update_interval}s")
    print()
    
    active_runs = {}  # track active W&B runs
    logged_steps = {}  # track logged iterations per directory
    
    try:
        while True:
            # Find active training directories
            active_dirs = find_active_training_dirs(base_output_dir)
            
            if not active_dirs:
                print(f"[{time.strftime('%H:%M:%S')}] No active training detected, waiting...")
                time.sleep(update_interval)
                continue
            
            # Log each active directory
            for dir_info in active_dirs:
                dir_path = str(dir_info['path'])
                metrics_file = dir_info['metrics_file']
                fold = dir_info['fold']
                stage = dir_info['stage']
                run_name = get_run_name(fold, stage)
                
                # Initialize W&B run if not exists
                if dir_path not in active_runs:
                    print(f"\n🚀 Starting W&B run: {run_name}")
                    
                    # Finish old run if switching
                    if active_runs:
                        old_path = list(active_runs.keys())[0]
                        print(f"   Finishing old run: {old_path}")
                        active_runs[old_path].finish()
                        del active_runs[old_path]
                    
                    # Start new run
                    run = wandb.init(
                        project=project_name,
                        name=run_name,
                        config={
                            "fold": fold,
                            "stage": stage,
                            "output_dir": dir_path,
                            "framework": "DeFRCN",
                        },
                        reinit=True
                    )
                    active_runs[dir_path] = run
                    logged_steps[dir_path] = 0
                    print(f"   ✅ Run URL: {run.url}")
                
                # Read and log new metrics
                if not metrics_file.exists():
                    continue
                
                with open(metrics_file, 'r') as f:
                    lines = f.readlines()
                
                # Log new lines
                for line in lines[logged_steps[dir_path]:]:
                    try:
                        metric = json.loads(line.strip())
                        
                        # Log to W&B
                        step = metric.get('iteration', None)
                        active_runs[dir_path].log(metric, step=step)
                        
                        logged_steps[dir_path] += 1
                        
                    except json.JSONDecodeError:
                        continue
                
                if logged_steps[dir_path] % 10 == 0:
                    print(f"[{time.strftime('%H:%M:%S')}] {run_name}: Logged {logged_steps[dir_path]} steps", end='\r')
            
            time.sleep(update_interval)
            
    except KeyboardInterrupt:
        print("\n\n✅ Stopping W&B logger...")
        for run in active_runs.values():
            run.finish()
        print("✅ All runs finished")

if __name__ == "__main__":
    import sys
    
    base_output_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs_codetr_2x"
    project_name = sys.argv[2] if len(sys.argv) > 2 else "DeFRCN_GCP_T"
    
    log_training_to_wandb(base_output_dir, project_name)

