#!/usr/bin/env python3
"""
Simple W&B logger for DeFRCN that reads metrics.json and logs to W&B
"""
import wandb
import json
import time
import os
from pathlib import Path

def log_defrcn_to_wandb(
    output_dir,
    project_name="defrcn_gcp_c",
    run_name=None,
    update_interval=30
):
    """
    Monitor DeFRCN metrics.json and log to W&B
    
    Args:
        output_dir: Path to DeFRCN output directory
        project_name: W&B project name
        run_name: W&B run name
        update_interval: How often to check for updates (seconds)
    """
    output_dir = Path(output_dir)
    metrics_file = output_dir / "metrics.json"
    
    if not metrics_file.exists():
        print(f"⏳ Waiting for metrics file: {metrics_file}")
        while not metrics_file.exists():
            time.sleep(5)
    
    # Initialize W&B
    print(f"🚀 Initializing W&B...")
    print(f"   Project: {project_name}")
    print(f"   Run: {run_name}")
    
    wandb.init(
        project=project_name,
        name=run_name,
        config={
            "output_dir": str(output_dir),
            "framework": "DeFRCN",
        }
    )
    
    print(f"✅ W&B initialized!")
    print(f"📊 Dashboard: {wandb.run.url}")
    
    last_size = 0
    logged_lines = 0
    
    try:
        while True:
            # Check if file has grown
            current_size = metrics_file.stat().st_size
            
            if current_size > last_size:
                # Read new lines
                with open(metrics_file, 'r') as f:
                    lines = f.readlines()
                
                # Log new metrics
                for line in lines[logged_lines:]:
                    try:
                        metric = json.loads(line.strip())
                        
                        # Log to W&B
                        if 'iteration' in metric:
                            step = metric['iteration']
                        else:
                            step = None
                        
                        wandb.log(metric, step=step)
                        logged_lines += 1
                        
                    except json.JSONDecodeError:
                        continue
                
                last_size = current_size
            
            time.sleep(update_interval)
            
    except KeyboardInterrupt:
        print("\n✅ W&B logging stopped")
        wandb.finish()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python wandb_logger.py <output_dir> [project_name] [run_name]")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    project_name = sys.argv[2] if len(sys.argv) > 2 else "defrcn_gcp_c"
    run_name = sys.argv[3] if len(sys.argv) > 3 else None
    
    log_defrcn_to_wandb(output_dir, project_name, run_name)

