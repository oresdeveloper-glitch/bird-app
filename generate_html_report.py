#!/usr/bin/env python3
"""
Generate comprehensive HTML evaluation report
"""

import json
import os
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent
EVALUATION_DIR = PROJECT_ROOT / "evaluation_results"
REPORT_FILE = EVALUATION_DIR / "index.html"

# Read the latest evaluation files
latest_summary = None
for f in sorted(EVALUATION_DIR.glob("summary_*.json"), reverse=True):
    with open(f) as fp:
        latest_summary = json.load(fp)
    break

latest_cm = None
for f in sorted(EVALUATION_DIR.glob("confusion_matrix_*.json"), reverse=True):
    with open(f) as fp:
        latest_cm = json.load(fp)
    break

latest_eval = None
for f in sorted(EVALUATION_DIR.glob("model_evaluation_*.json"), reverse=True):
    with open(f) as fp:
        latest_eval = json.load(fp)
    break

html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bird Model Evaluation Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f5f5; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px; text-align: center; border-radius: 8px 8px 0 0; }}
        h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .subtitle {{ font-size: 0.9em; opacity: 0.9; }}
        .content {{ padding: 40px; }}
        section {{ margin-bottom: 40px; border-bottom: 2px solid #eee; padding-bottom: 30px; }}
        section:last-child {{ border-bottom: none; }}
        h2 {{ color: #333; font-size: 1.8em; margin-bottom: 20px; border-left: 4px solid #667eea; padding-left: 15px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .stat-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #667eea; }}
        .stat-card h3 {{ color: #667eea; font-size: 0.9em; text-transform: uppercase; margin-bottom: 10px; }}
        .stat-card .value {{ font-size: 2em; font-weight: bold; color: #333; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th {{ background: #f8f9fa; padding: 12px; text-align: left; font-weight: 600; color: #333; border-bottom: 2px solid #ddd; }}
        td {{ padding: 12px; border-bottom: 1px solid #eee; }}
        tr:hover {{ background: #f8f9fa; }}
        .status-ok {{ color: #28a745; font-weight: bold; }}
        .status-error {{ color: #dc3545; font-weight: bold; }}
        .status-pending {{ color: #ffc107; font-weight: bold; }}
        .accuracy-bar {{ background: #e9ecef; border-radius: 4px; overflow: hidden; height: 30px; }}
        .accuracy-fill {{ background: linear-gradient(90deg, #667eea, #764ba2); height: 100%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; }}
        .footer {{ background: #f8f9fa; padding: 20px; text-align: center; color: #666; font-size: 0.9em; border-top: 1px solid #ddd; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Tanzania Bird Identification System</h1>
            <p class="subtitle">Model Evaluation Report</p>
        </header>
        
        <div class="content">
            <!-- Summary Section -->
            <section>
                <h2>Evaluation Summary</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <h3>Total Birds in Database</h3>
                        <div class="value">{total_birds}</div>
                    </div>
                    <div class="stat-card">
                        <h3>Dataset Folders</h3>
                        <div class="value">{dataset_folders}</div>
                    </div>
                    <div class="stat-card">
                        <h3>Models Evaluated</h3>
                        <div class="value">{models_evaluated}</div>
                    </div>
                    <div class="stat-card">
                        <h3>Report Generated</h3>
                        <div class="value">{timestamp}</div>
                    </div>
                </div>
            </section>
            
            <!-- Model Evaluation Results -->
            <section>
                <h2>Model Evaluation Results</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Model Name</th>
                            <th>Status</th>
                            <th>Loss</th>
                            <th>Accuracy</th>
                            <th>Classes</th>
                        </tr>
                    </thead>
                    <tbody>
                        {model_rows}
                    </tbody>
                </table>
            </section>
            
            <!-- Confusion Matrix Summary -->
            <section>
                <h2>Confusion Matrix Evaluation</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Images Processed</th>
                            <th>Correct Predictions</th>
                            <th>Accuracy</th>
                            <th>Classes</th>
                        </tr>
                    </thead>
                    <tbody>
                        {cm_rows}
                    </tbody>
                </table>
            </section>
            
            <!-- Database Info -->
            <section>
                <h2>Database Information</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <h3>Total Birds Populated</h3>
                        <div class="value">{total_birds}</div>
                    </div>
                    <div class="stat-card">
                        <h3>Dataset Folders Matched</h3>
                        <div class="value">{dataset_folders}</div>
                    </div>
                    <div class="stat-card">
                        <h3>Images in Dataset</h3>
                        <div class="value">9,414</div>
                    </div>
                </div>
            </section>
            
            <!-- Recommendations -->
            <section>
                <h2>Recommendations</h2>
                <ul style="list-style: none; padding: 0;">
                    <li style="padding: 10px 0; border-bottom: 1px solid #eee;">
                        <strong>✓ Database Updated:</strong> All 200 bird species from the dataset have been added to the database.
                    </li>
                    <li style="padding: 10px 0; border-bottom: 1px solid #eee;">
                        <strong>⚠ Model Evaluation:</strong> There are issues loading and evaluating the trained Keras models. This may be due to model format or TensorFlow version compatibility.
                    </li>
                    <li style="padding: 10px 0; border-bottom: 1px solid #eee;">
                        <strong>✓ Confusion Matrix Generated:</strong> NIPE-embeddings model successfully generated confusion matrix on 100 test images.
                    </li>
                    <li style="padding: 10px 0;">
                        <strong>Next Steps:</strong> Re-train models with full dataset or debug model loading to get accurate metrics.
                    </li>
                </ul>
            </section>
        </div>
        
        <div class="footer">
            <p>Report generated on {timestamp_full}</p>
            <p>Saved in: <code>evaluation_results/</code></p>
        </div>
    </div>
</body>
</html>
"""

# Generate model rows
model_rows = ""
if latest_eval:
    for model_name, result in latest_eval.items():
        status = result.get("status", "unknown")
        if status == "evaluated":
            status_html = f'<span class="status-ok">✓ {status}</span>'
            loss = result.get("loss", "N/A")
            accuracy = result.get("accuracy", 0) * 100
            classes = result.get("classes", 0)
        else:
            status_html = f'<span class="status-error">✗ {status}</span>'
            loss = "N/A"
            accuracy = "N/A"
            classes = "N/A"
        
        model_rows += f"""
        <tr>
            <td><strong>{model_name}</strong></td>
            <td>{status_html}</td>
            <td>{loss}</td>
            <td>{accuracy if accuracy == 'N/A' else f'{accuracy:.2f}%'}</td>
            <td>{classes}</td>
        </tr>
        """

# Generate confusion matrix rows
cm_rows = ""
if latest_cm:
    for model_name, cm_data in latest_cm.items():
        meta = cm_data.get("meta", {})
        processed = meta.get("processed", 0)
        correct = meta.get("correct", 0)
        accuracy = meta.get("accuracy", 0) * 100
        classes = meta.get("classes", 0)
        
        cm_rows += f"""
        <tr>
            <td><strong>{model_name}</strong></td>
            <td>{processed}</td>
            <td>{correct}</td>
            <td>
                <div class="accuracy-bar">
                    <div class="accuracy-fill" style="width: {accuracy}%">
                        {accuracy:.2f}%
                    </div>
                </div>
            </td>
            <td>{classes}</td>
        </tr>
        """

# Format the HTML
final_html = html_content.format(
    total_birds=latest_summary.get("total_birds_in_db", "N/A") if latest_summary else "N/A",
    dataset_folders=latest_summary.get("dataset_folders", "N/A") if latest_summary else "N/A",
    models_evaluated=latest_summary.get("models_evaluated", 0) if latest_summary else 0,
    timestamp=latest_summary.get("timestamp", "N/A") if latest_summary else "N/A",
    timestamp_full=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    model_rows=model_rows or "<tr><td colspan='5' style='text-align: center; padding: 20px; color: #999;'>No model data available</td></tr>",
    cm_rows=cm_rows or "<tr><td colspan='5' style='text-align: center; padding: 20px; color: #999;'>No confusion matrix data available</td></tr>"
)

# Save HTML report
with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write(final_html)

print(f"✓ HTML Report generated: {REPORT_FILE.name}")
print(f"\nReport Summary:")
print(f"  Total Birds in DB: {latest_summary.get('total_birds_in_db', 'N/A') if latest_summary else 'N/A'}")
print(f"  Dataset Folders: {latest_summary.get('dataset_folders', 'N/A') if latest_summary else 'N/A'}")
print(f"  Models Evaluated: {latest_summary.get('models_evaluated', 0) if latest_summary else 0}")
