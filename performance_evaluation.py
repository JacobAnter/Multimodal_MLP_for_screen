import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, precision_recall_curve

KS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]

def _load_and_merge_data(preds_tsv, labels_tsv):
    """Loads prediction and label TSV files and merges them on the 'gene' column."""
    preds_df = pd.read_csv(preds_tsv, sep='\t')
    labels_df = pd.read_csv(labels_tsv, sep='\t')
    
    if 'gene' not in preds_df.columns or 'gene' not in labels_df.columns:
        raise ValueError("Both TSV files must contain a 'gene' column.")
    if 'probability' not in preds_df.columns:
        raise ValueError("Prediction TSV must contain a 'probability' column.")
    if 'label' not in labels_df.columns:
        raise ValueError("Label TSV must contain a 'label' column.")
        
    merged_df = pd.merge(preds_df, labels_df, on='gene')
    return merged_df

def plot_auroc(preds_tsv, labels_tsv, output_path):
    """Calculates AUROC, plots the ROC curve, and saves it to disk."""
    df = _load_and_merge_data(preds_tsv, labels_tsv)
    y_true = df['label'].values
    y_scores = df['probability'].values
    
    auroc = roc_auc_score(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUROC = {auroc:.4f}')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (AUROC: {auroc:.4f})')
    plt.legend(loc='lower right')
    base_path, _ = os.path.splitext(output_path)
    plt.savefig(f"{base_path}.png", dpi=300)
    plt.savefig(f"{base_path}.svg")
    plt.close()
    
    return auroc

def plot_pr_auc(preds_tsv, labels_tsv, output_path):
    """Calculates PR AUC, plots the Precision-Recall curve, and saves it to disk."""
    df = _load_and_merge_data(preds_tsv, labels_tsv)
    y_true = df['label'].values
    y_scores = df['probability'].values
    
    pr_auc = average_precision_score(y_true, y_scores)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    
    plt.figure()
    plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (PR AUC: {pr_auc:.4f})')
    plt.legend(loc='lower left')
    base_path, _ = os.path.splitext(output_path)
    plt.savefig(f"{base_path}.png", dpi=300)
    plt.savefig(f"{base_path}.svg")
    plt.close()
    
    return pr_auc

def _compute_top_k(df, k):
    """Helper function to compute top-k metrics given a DataFrame."""
    df_sorted = df.sort_values(by='probability', ascending=False).head(k)
    num_pos_topk = df_sorted['label'].sum()
    total_positives = df['label'].sum()
    num_samples = len(df)
    
    recall_k = num_pos_topk / total_positives if total_positives > 0 else 0.0
    precision_k = num_pos_topk / k if k > 0 else 0.0
    baseline_rate = total_positives / num_samples if num_samples > 0 else 0.0
    enrichment_k = precision_k / baseline_rate if baseline_rate > 0 else 0.0
    
    return num_pos_topk, recall_k, precision_k, enrichment_k

def plot_top_k_metrics(preds_tsv, labels_tsv, metric_type, output_path):
    """Calculates and plots a @k metric (recall, precision, or enrichment)."""
    df = _load_and_merge_data(preds_tsv, labels_tsv)
    
    metrics = []
    for k in KS:
        actual_k = min(k, len(df))
        _, recall_k, precision_k, enrichment_k = _compute_top_k(df, actual_k)
        
        if metric_type == 'recall':
            metrics.append(recall_k)
        elif metric_type == 'precision':
            metrics.append(precision_k)
        elif metric_type == 'enrichment':
            metrics.append(enrichment_k)
        else:
            raise ValueError(f"Unknown metric_type: {metric_type}")
            
    plt.figure()
    plt.plot(KS, metrics, marker='o')
    plt.xlabel('k')
    plt.ylabel(metric_type.capitalize())
    plt.title(f'{metric_type.capitalize()} @ k Curve')
    plt.grid(True)
    base_path, _ = os.path.splitext(output_path)
    plt.savefig(f"{base_path}.png", dpi=300)
    plt.savefig(f"{base_path}.svg")
    plt.close()
    
    return dict(zip(KS, metrics))

def evaluate_all_metrics(preds_tsv, labels_tsv, output_dir, table_output_path):
    """
    Computes all metrics, generates all plots, and saves a summary table.
    
    Parameters:
    - preds_tsv: Path to predictions TSV.
    - labels_tsv: Path to labels TSV.
    - output_dir: Directory to save the plot images.
    - table_output_path: File path to save the generated metrics table.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all plots
    plot_auroc(preds_tsv, labels_tsv, os.path.join(output_dir, 'auroc.png'))
    plot_pr_auc(preds_tsv, labels_tsv, os.path.join(output_dir, 'pr_auc.png'))
    plot_top_k_metrics(preds_tsv, labels_tsv, 'recall', os.path.join(output_dir, 'recall_at_k.png'))
    plot_top_k_metrics(preds_tsv, labels_tsv, 'precision', os.path.join(output_dir, 'precision_at_k.png'))
    plot_top_k_metrics(preds_tsv, labels_tsv, 'enrichment', os.path.join(output_dir, 'enrichment_at_k.png'))
    
    # Generate summary table
    df = _load_and_merge_data(preds_tsv, labels_tsv)
    records = []
    
    for k in KS:
        actual_k = min(k, len(df))
        num_pos_topk, recall_k, precision_k, enrichment_k = _compute_top_k(df, actual_k)
        records.append({
            'k': k,
            '#positives': num_pos_topk,
            'recall@k': recall_k,
            'precision@k': precision_k,
            'enrichment@k': enrichment_k
        })
        
    results_df = pd.DataFrame(records)
    results_df.to_csv(table_output_path, sep='\t', index=False)
