#!/usr/bin/env python3
"""
DeFRCN W&B Auto-Logger with Auto-Switching
Automatically detects training stage and switches W&B runs
"""

import wandb
import time
import json
import subprocess
import re
from pathlib import Path

def monitor_defrcn_wandb(project_name="gcp_DFRCN"):
    """Auto-switching W&B logger for DeFRCN multi-fold training"""
    
    current_run = None
    current_fold = None
    current_stage = None
    current_kshot = None
    
    print("=" * 70)
    print("DeFRCN W&B Auto-Logger with Auto-Switching")
    print("=" * 70)
    print(f"Project: {project_name}")
    print()
    
    while True:
        try:
            # Detect active training
            result = subprocess.run(
                "ps aux | grep 'python.*main.py' | grep -v grep",
                shell=True, capture_output=True, text=True
            )
            
            if not result.stdout.strip():
                print("No training detected, waiting...")
                if current_run:
                    print(f"Finishing run for {current_fold}/{current_stage}")
                    current_run.finish()
                    current_run = None
                    current_fold = None
                    current_stage = None
                    current_kshot = None
                time.sleep(30)
                continue
            
            # Find most recent metrics.json file
            result = subprocess.run(
                "find outputs_defrcn -name 'metrics.json' -type f -printf '%T@ %p\\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-",
                shell=True, capture_output=True, text=True, cwd=str(Path.home() / "DeFRCN")
            )
            
            if not result.stdout.strip():
                time.sleep(10)
                continue
            
            metrics_file = Path.home() / "DeFRCN" / result.stdout.strip()
            
            # Detect fold, stage, and k-shot from path
            path_parts = str(metrics_file).split('/')
            fold = None
            stage = None
            kshot = None
            
            for part in path_parts:
                if part.startswith('OD25_'):
                    fold = part
                if part == 'base_model':
                    stage = 'base'
                elif 'novel' in part:
                    # Extract k-shot from "novel_1shot", "novel_10shot", etc.
                    match = re.search(r'novel_(\d+)shot', part)
                    if match:
                        kshot = match.group(1)
                        stage = f'novel_{kshot}shot'
            
            # Check if we need to switch runs
            if fold != current_fold or stage != current_stage:
                if current_run:
                    print(f"\nFinishing run for {current_fold}/{current_stage}")
                    current_run.finish()
                
                current_fold = fold
                current_stage = stage
                current_kshot = kshot
                
                run_name = f"{fold}_{stage}" if fold and stage else "training"
                print(f"\nStarting new W&B run: {run_name}")
                print("=" * 70)
                
                config_dict = {
                    "fold": fold,
                    "stage": stage,
                    "framework": "DeFRCN",
                    "backbone": "ResNet101",
                }
                
                if kshot:
                    config_dict["k_shot"] = int(kshot)
                    config_dict["num_classes"] = 100  # base + novel
                else:
                    config_dict["num_classes"] = 75  # base only
                
                current_run = wandb.init(
                    project=project_name,
                    name=run_name,
                    config=config_dict,
                    reinit=True
                )
            
            # Log metrics from JSON file
            if metrics_file.exists() and current_run:
                with open(metrics_file) as f:
                    lines = f.readlines()
                    if lines:
                        # Read last line
                        last_line = lines[-1].strip()
                        try:
                            metrics = json.loads(last_line)
                            
                            # Filter and log metrics
                            log_dict = {}
                            for key, value in metrics.items():
                                if isinstance(value, (int, float)):
                                    # Clean up key names for better visualization
                                    clean_key = key.replace('total_', '').replace('bbox/', '')
                                    log_dict[clean_key] = value
                            
                            if log_dict:
                                current_run.log(log_dict)
                                print(f"Logged {len(log_dict)} metrics for {fold}/{stage}", end='\r')
                        except json.JSONDecodeError:
                            pass
            
            time.sleep(10)
            
        except KeyboardInterrupt:
            print("\n\nStopping logger...")
            if current_run:
                current_run.finish()
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    monitor_defrcn_wandb()

