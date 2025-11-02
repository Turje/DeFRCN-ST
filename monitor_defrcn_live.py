#!/usr/bin/env python3
"""
Live DeFRCN Training Monitor
"""
import subprocess
import time
import os

def clear_screen():
    os.system('clear' if os.name == 'posix' else 'cls')

def monitor_defrcn():
    """Monitor DeFRCN training with live updates"""
    print("🔴 LIVE DeFRCN Training Monitor (GCP)")
    print("="*70)
    print("Press Ctrl+C to stop monitoring (training continues)")
    print("="*70)
    print()
    time.sleep(2)
    
    try:
        while True:
            clear_screen()
            
            print("🔴 LIVE DeFRCN Training Monitor (GCP - num_classes=75)")
            print("="*70)
            print(f"⏰ Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*70)
            print()
            
            # Get process status
            result = subprocess.run(
                "ps aux | grep 'python.*main.py' | grep -v grep | head -1",
                shell=True, capture_output=True, text=True
            )
            
            if result.stdout.strip():
                parts = result.stdout.split()
                print(f"✅ Status: RUNNING")
                print(f"   PID: {parts[1]} | CPU: {parts[2]}% | Memory: {parts[3]}% | Time: {parts[9]}")
            else:
                print("❌ Status: NOT RUNNING")
                print("⚠️  Training may have stopped")
            
            print()
            print("📊 Latest Training Progress:")
            print("-"*70)
            
            # Get last 5 training iterations
            result = subprocess.run(
                "tail -100 ~/DeFRCN/training_DeFRCN_GCP_T_continued.log 2>/dev/null | grep 'iter:' | tail -5",
                shell=True, capture_output=True, text=True
            )
            
            if result.stdout.strip():
                for line in result.stdout.strip().split('\n'):
                    # Parse DeFRCN log format
                    if 'iter:' in line and 'total_loss:' in line:
                        parts = line.split()
                        iter_val = loss_val = eta = "N/A"
                        
                        for i, part in enumerate(parts):
                            if 'iter:' in part and i+1 < len(parts):
                                iter_val = parts[i+1]
                            if 'total_loss:' in part and i+1 < len(parts):
                                loss_val = parts[i+1]
                            if 'eta:' in part and i+1 < len(parts):
                                eta = parts[i+1]
                        
                        print(f"   [Iter {iter_val}] | Loss: {loss_val} | ETA: {eta}")
            else:
                print("   ⏳ Waiting for training to start...")
            
            # Get mAP evaluations
            print()
            print("🎯 mAP Evaluations (FIXED - num_classes=75!):")
            print("-"*70)
            
            result = subprocess.run(
                "tail -500 ~/DeFRCN/training_DeFRCN_GCP_T_continued.log 2>/dev/null | grep -E 'Average Precision|bbox/AP' | tail -10",
                shell=True, capture_output=True, text=True
            )
            
            if result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                
                # Look for the main AP line
                for line in lines:
                    if 'Average Precision' in line and 'IoU=0.50:0.95' in line:
                        # Extract AP value
                        parts = line.split('=')
                        if len(parts) >= 2:
                            ap_val = parts[-1].strip()
                            try:
                                ap_float = float(ap_val)
                                if ap_float > 0.15:
                                    marker = "✅"
                                elif ap_float > 0.05:
                                    marker = "⚠️"
                                else:
                                    marker = "❌"
                                print(f"   {marker} mAP: {ap_float:.4f} ({ap_float*100:.2f}%)")
                            except:
                                print(f"   📊 mAP: {ap_val}")
                    elif 'bbox/AP:' in line:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if 'bbox/AP:' in part and i+1 < len(parts):
                                ap_val = parts[i+1]
                                try:
                                    ap_float = float(ap_val)
                                    if ap_float > 15:
                                        marker = "✅"
                                    elif ap_float > 5:
                                        marker = "⚠️"
                                    else:
                                        marker = "❌"
                                    print(f"   {marker} bbox/AP: {ap_float:.2f}%")
                                except:
                                    print(f"   📊 bbox/AP: {ap_val}")
            else:
                print("   ⏳ No evaluations yet (happens at end of base training)")
            
            print()
            print("-"*70)
            print("🔗 W&B: https://wandb.ai/sturjem000-the-city-university-of-new-york/DeFRCN_GCP_T")
            print()
            
            print("📝 Training Configuration:")
            print("   ✅ Fold: OD25_0 (base training)")
            print("   ✅ Config: NUM_CLASSES=75 (FIXED!)")
            print("   ✅ Data: 75 base classes with sparse IDs")
            print("   ✅ Detectron2 handles sparse→contiguous mapping")
            print("   ✅ Total iterations: 3472")
            print()
            
            print("🎯 Expected mAP after base training: 15-40% (if sparse ID handling works!)")
            print()
            print("💡 Press Ctrl+C to stop monitoring (training continues)")
            print("="*70)
            
            # Update every 10 seconds
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped. Training continues in background!")
        print(f"📊 View W&B: https://wandb.ai/sturjem000-the-city-university-of-new-york/DeFRCN_GCP_T")

if __name__ == "__main__":
    monitor_defrcn()

