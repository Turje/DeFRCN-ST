import os
import re
import pandas as pd

def extract_metrics_from_log(log_path):
    """Extract AP metrics from a log file."""
    metrics = {}
    try:
        with open(log_path, 'r') as f:
            content = f.read()
            # Find all evaluation results
            pattern = r'copypaste: ([bn]AP,[bn]AP50,[bn]AP75,[bn]APs,[bn]APm,[bn]APl,AP)\n.*?copypaste: ([\d\.-]+,[\d\.-]+,[\d\.-]+,[\d\.-]+,[\d\.-]+,[\d\.-]+,[\d\.-]+)'
            matches = re.findall(pattern, content, re.MULTILINE)
            
            # Get the last evaluation results (most recent)
            if matches:
                last_match = matches[-1]
                metric_names = last_match[0].split(',')
                metric_values = [float(x) for x in last_match[1].split(',')]
                metrics = dict(zip(metric_names, metric_values))
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
    return metrics

# Define the folds and shots to process
folds = ["OD25_0", "OD25_1", "OD25_2"]
shots = ["base", "1shot", "3shot", "5shot", "10shot"]

# Initialize results storage
results = []

# Process each fold and shot combination
for fold in folds:
    for shot in shots:
        # Determine the log file path based on shot type
        if shot == "base":
            log_path = f"outputs_defrcn/{fold}/base_model/log.txt"
        else:
            log_path = f"outputs_defrcn/{fold}/novel_{shot}/log.txt"
        
        if os.path.exists(log_path):
            metrics = extract_metrics_from_log(log_path)
            if metrics:
                row = {
                    "Fold": fold,
                    "Shot": shot,
                    "AP": metrics.get("AP", None),
                    "AP50": metrics.get(f"{'b' if shot=='base' else 'n'}AP50", None),
                    "AP75": metrics.get(f"{'b' if shot=='base' else 'n'}AP75", None),
                    "APs": metrics.get(f"{'b' if shot=='base' else 'n'}APs", None),
                    "APm": metrics.get(f"{'b' if shot=='base' else 'n'}APm", None),
                    "APl": metrics.get(f"{'b' if shot=='base' else 'n'}APl", None)
                }
                results.append(row)

# Convert to DataFrame and save to Excel
df = pd.DataFrame(results)

# Reorder columns for better readability
df = df[["Fold", "Shot", "AP", "AP50", "AP75", "APs", "APm", "APl"]]

# Save to Excel with some formatting
with pd.ExcelWriter('defrcn_evaluation_results.xlsx', engine='openpyxl') as writer:
    df.to_excel(writer, index=False, sheet_name='Evaluation Results')
    
    # Get the worksheet
    worksheet = writer.sheets['Evaluation Results']
    
    # Format headers
    for col in range(len(df.columns)):
        cell = worksheet.cell(row=1, column=col+1)
        cell.font = cell.font.copy(bold=True)
    
    # Auto-adjust column widths
    for col in worksheet.columns:
        max_length = 0
        column = col[0].column_letter
        for cell in col:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = (max_length + 2)
        worksheet.column_dimensions[column].width = adjusted_width

print("Results saved to defrcn_evaluation_results.xlsx")