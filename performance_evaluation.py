import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, precision_recall_curve, auc
from decimal import Decimal, ROUND_HALF_UP

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

def plot_auroc(preds_tsvs, labels_tsvs, descriptions, output_path):
    """Calculates AUROC, plots the ROC curves, and saves it to disk."""
    labels_tsv_list = [labels_tsvs] * len(preds_tsvs) if isinstance(labels_tsvs, str) else labels_tsvs
    
    plt.figure()
    colors = plt.get_cmap('tab10').colors
    
    aurocs = []
    for idx, (preds_tsv, labels_tsv, desc) in enumerate(zip(preds_tsvs, labels_tsv_list, descriptions)):
        df = _load_and_merge_data(preds_tsv, labels_tsv)
        y_true = df['label'].values
        y_scores = df['probability'].values
        
        auroc = roc_auc_score(y_true, y_scores)
        aurocs.append(auroc)
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        
        auroc_rounded = Decimal(str(auroc)).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)
        color = colors[idx % len(colors)]
        plt.plot(fpr, tpr, label=f'{desc} (AUROC = {auroc_rounded})', color=color)
        
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    leg = plt.legend(loc='lower right')
    if leg and aurocs:
        best_idx = aurocs.index(max(aurocs))
        leg.get_texts()[best_idx].set_fontweight('bold')
    base_path, _ = os.path.splitext(output_path)
    plt.savefig(f"{base_path}.png", dpi=300)
    plt.savefig(f"{base_path}.svg")
    plt.close()
    
    return aurocs

def plot_pr_auc(preds_tsvs, labels_tsvs, descriptions, output_path):
    """Calculates PR AUC, plots the Precision-Recall curves, and saves it to disk."""
    labels_tsv_list = [labels_tsvs] * len(preds_tsvs) if isinstance(labels_tsvs, str) else labels_tsvs
    
    plt.figure()
    colors = plt.get_cmap('tab10').colors
    
    pr_aucs = []
    for idx, (preds_tsv, labels_tsv, desc) in enumerate(zip(preds_tsvs, labels_tsv_list, descriptions)):
        df = _load_and_merge_data(preds_tsv, labels_tsv)
        y_true = df['label'].values
        y_scores = df['probability'].values
        
        pr_auc = average_precision_score(y_true, y_scores)
        pr_aucs.append(pr_auc)
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        
        pr_auc_rounded = Decimal(str(pr_auc)).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)
        color = colors[idx % len(colors)]
        plt.plot(recall, precision, label=f'{desc} (PR AUC = {pr_auc_rounded})', color=color)
        
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    leg = plt.legend(loc='lower left')
    if leg and pr_aucs:
        best_idx = pr_aucs.index(max(pr_aucs))
        leg.get_texts()[best_idx].set_fontweight('bold')
    base_path, _ = os.path.splitext(output_path)
    plt.savefig(f"{base_path}.png", dpi=300)
    plt.savefig(f"{base_path}.svg")
    plt.close()
    
    return pr_aucs

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

def plot_top_k_metrics(preds_tsvs, labels_tsvs, descriptions, metric_type, output_path):
    """Calculates and plots a @k metric (recall, precision, or enrichment)."""
    labels_tsv_list = [labels_tsvs] * len(preds_tsvs) if isinstance(labels_tsvs, str) else labels_tsvs
    
    plt.figure()
    colors = plt.get_cmap('tab10').colors
    
    all_metrics = []
    aucs = []
    for idx, (preds_tsv, labels_tsv, desc) in enumerate(zip(preds_tsvs, labels_tsv_list, descriptions)):
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
                
        all_metrics.append(dict(zip(KS, metrics)))
        aucs.append(auc(KS, metrics))
        color = colors[idx % len(colors)]
        plt.plot(KS, metrics, marker='o', label=desc, color=color)
            
    plt.xlabel('k')
    plt.ylabel(metric_type.capitalize())
    plt.title(f'{metric_type.capitalize()} @ k Curve')
    leg = plt.legend()
    if leg and aucs:
        best_idx = aucs.index(max(aucs))
        leg.get_texts()[best_idx].set_fontweight('bold')
    plt.grid(True)
    base_path, _ = os.path.splitext(output_path)
    plt.savefig(f"{base_path}.png", dpi=300)
    plt.savefig(f"{base_path}.svg")
    plt.close()
    
    return all_metrics

def evaluate_all_metrics(preds_tsvs, labels_tsvs, descriptions, output_dir):
    """
    Computes all metrics, generates all plots, and saves summary tables.
    
    Parameters:
    - preds_tsvs: Iterable of paths to predictions TSVs.
    - labels_tsvs: Iterable of paths to labels TSVs, or a single path.
    - descriptions: Iterable of descriptions for the models.
    - output_dir: Directory to save the plot images and tables.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all plots
    plot_auroc(preds_tsvs, labels_tsvs, descriptions, os.path.join(output_dir, 'auroc.png'))
    plot_pr_auc(preds_tsvs, labels_tsvs, descriptions, os.path.join(output_dir, 'pr_auc.png'))
    plot_top_k_metrics(preds_tsvs, labels_tsvs, descriptions, 'recall', os.path.join(output_dir, 'recall_at_k.png'))
    plot_top_k_metrics(preds_tsvs, labels_tsvs, descriptions, 'precision', os.path.join(output_dir, 'precision_at_k.png'))
    plot_top_k_metrics(preds_tsvs, labels_tsvs, descriptions, 'enrichment', os.path.join(output_dir, 'enrichment_at_k.png'))
    
    # Generate summary tables
    labels_tsv_list = [labels_tsvs] * len(preds_tsvs) if isinstance(labels_tsvs, str) else labels_tsvs
    for preds_tsv, labels_tsv, desc in zip(preds_tsvs, labels_tsv_list, descriptions):
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
        table_output_path = os.path.join(output_dir, f"{desc}_performance_top_k_comparison.tsv")
        results_df.to_csv(table_output_path, sep='\t', index=False)

def evaluate_cv_performance(preds_template, labels_tsv, k, start_idx=0):
    """
    Evaluates AUROC and PR AUC across k cross-validation splits or random seeds.
    
    Parameters:
    - preds_template: A template string for the predictions file path, e.g., "dir/preds_seed_{i}.tsv".
    - labels_tsv: Path to the labels file, or a template string if labels vary per split.
    - k: Number of splits/seeds.
    - start_idx: The starting index for `i` (usually 0 or 1).
    """
    aurocs = []
    pr_aucs = []
    
    for i in range(start_idx, start_idx + k):
        preds_path = preds_template.format(i=i)
        
        # Check if labels_tsv is a template
        if "{i}" in labels_tsv:
            labels_path = labels_tsv.format(i=i)
        else:
            labels_path = labels_tsv
            
        df = _load_and_merge_data(preds_path, labels_path)
        y_true = df['label'].values
        y_scores = df['probability'].values
        
        auroc = roc_auc_score(y_true, y_scores)
        pr_auc = average_precision_score(y_true, y_scores)
        
        aurocs.append(auroc)
        pr_aucs.append(pr_auc)
        
    auroc_mean = np.mean(aurocs)
    auroc_std = np.std(aurocs)
    pr_auc_mean = np.mean(pr_aucs)
    pr_auc_std = np.std(pr_aucs)
    
    print(f"--- Cross-Validation / Multi-Seed Performance (k={k}) ---")
    print(f"AUROC:  {auroc_mean:.4f} ± {auroc_std:.4f}")
    print(f"PR AUC: {pr_auc_mean:.4f} ± {pr_auc_std:.4f}")
    
    return {
        'auroc_mean': auroc_mean,
        'auroc_std': auroc_std,
        'pr_auc_mean': pr_auc_mean,
        'pr_auc_std': pr_auc_std,
        'aurocs': aurocs,
        'pr_aucs': pr_aucs
    }
