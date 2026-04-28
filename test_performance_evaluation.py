import os
import unittest
import pandas as pd
from performance_evaluation import (
    plot_auroc,
    plot_pr_auc,
    plot_top_k_metrics,
    evaluate_all_metrics
)

class TestPerformanceEvaluation(unittest.TestCase):
    def setUp(self):
        self.preds_tsv = 'test_preds.tsv'
        self.labels_tsv = 'test_labels.tsv'
        self.output_dir = 'test_output_dir'
        self.table_output_path = 'test_output_table.tsv'
        
        # Create a dataset of 250 items to properly test k up to 200
        # The first 50 are positives with high probabilities, ensuring perfect separation.
        preds_data = {
            'gene': [f'gene_{i}' for i in range(250)],
            'probability': [0.9 if i < 50 else 0.1 for i in range(250)]
        }
        labels_data = {
            'gene': [f'gene_{i}' for i in range(250)],
            'label': [1 if i < 50 else 0 for i in range(250)]
        }
        
        pd.DataFrame(preds_data).to_csv(self.preds_tsv, sep='\t', index=False)
        pd.DataFrame(labels_data).to_csv(self.labels_tsv, sep='\t', index=False)

    def tearDown(self):
        if os.path.exists(self.preds_tsv):
            os.remove(self.preds_tsv)
        if os.path.exists(self.labels_tsv):
            os.remove(self.labels_tsv)
        if os.path.exists(self.table_output_path):
            os.remove(self.table_output_path)
            
        if os.path.exists(self.output_dir):
            for file in os.listdir(self.output_dir):
                os.remove(os.path.join(self.output_dir, file))
            os.rmdir(self.output_dir)

    def test_plot_auroc(self):
        output_plot = os.path.join(self.output_dir, 'test_auroc.png')
        os.makedirs(self.output_dir, exist_ok=True)
        
        auroc = plot_auroc(self.preds_tsv, self.labels_tsv, output_plot)
        self.assertEqual(auroc, 1.0) # perfect separation
        self.assertTrue(os.path.exists(output_plot))

    def test_plot_pr_auc(self):
        output_plot = os.path.join(self.output_dir, 'test_pr_auc.png')
        os.makedirs(self.output_dir, exist_ok=True)
        
        pr_auc = plot_pr_auc(self.preds_tsv, self.labels_tsv, output_plot)
        self.assertEqual(pr_auc, 1.0) # perfect separation
        self.assertTrue(os.path.exists(output_plot))

    def test_plot_top_k_metrics(self):
        output_plot = os.path.join(self.output_dir, 'test_recall.png')
        os.makedirs(self.output_dir, exist_ok=True)
        
        metrics = plot_top_k_metrics(self.preds_tsv, self.labels_tsv, 'recall', output_plot)
        
        # Top 10 will have 10 positives, total positives = 50. Recall @ 10 = 10/50 = 0.2
        self.assertAlmostEqual(metrics[10], 0.2)
        # Top 50 will have 50 positives, recall = 1.0
        self.assertAlmostEqual(metrics[50], 1.0)
        
        self.assertTrue(os.path.exists(output_plot))

    def test_evaluate_all_metrics(self):
        evaluate_all_metrics(self.preds_tsv, self.labels_tsv, self.output_dir, self.table_output_path)
        
        # Check files exist
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'auroc.png')))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'pr_auc.png')))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'recall_at_k.png')))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'precision_at_k.png')))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'enrichment_at_k.png')))
        self.assertTrue(os.path.exists(self.table_output_path))
        
        # Check table
        df = pd.read_csv(self.table_output_path, sep='\t')
        self.assertIn('k', df.columns)
        self.assertIn('#positives', df.columns)
        self.assertIn('recall@k', df.columns)
        self.assertIn('precision@k', df.columns)
        self.assertIn('enrichment@k', df.columns)
        self.assertEqual(len(df), 20) # 20 k values
        
if __name__ == '__main__':
    unittest.main()
