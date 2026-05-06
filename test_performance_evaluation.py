import os
import unittest
import pandas as pd
from performance_evaluation import (
    plot_auroc,
    plot_pr_auc,
    plot_top_k_metrics,
    evaluate_all_metrics,
    evaluate_cv_performance
)

class TestPerformanceEvaluation(unittest.TestCase):
    def setUp(self):
        self.preds_tsv = 'test_preds.tsv'
        self.labels_tsv = 'test_labels.tsv'
        self.output_dir = 'test_output_dir'
        
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
            
        # Clean up evaluate_cv_performance test files
        for i in range(2):
            if os.path.exists(f"test_preds_seed_{i}.tsv"):
                os.remove(f"test_preds_seed_{i}.tsv")
            
        if os.path.exists(self.output_dir):
            for file in os.listdir(self.output_dir):
                os.remove(os.path.join(self.output_dir, file))
            os.rmdir(self.output_dir)

    def test_plot_auroc(self):
        output_plot = os.path.join(self.output_dir, 'test_auroc.png')
        os.makedirs(self.output_dir, exist_ok=True)
        
        aurocs = plot_auroc([self.preds_tsv], [self.labels_tsv], ['Test Model'], output_plot)
        self.assertEqual(aurocs[0], 1.0) # perfect separation
        self.assertTrue(os.path.exists(output_plot))

    def test_plot_pr_auc(self):
        output_plot = os.path.join(self.output_dir, 'test_pr_auc.png')
        os.makedirs(self.output_dir, exist_ok=True)
        
        pr_aucs = plot_pr_auc([self.preds_tsv], [self.labels_tsv], ['Test Model'], output_plot)
        self.assertEqual(pr_aucs[0], 1.0) # perfect separation
        self.assertTrue(os.path.exists(output_plot))

    def test_plot_top_k_metrics(self):
        output_plot = os.path.join(self.output_dir, 'test_recall.png')
        os.makedirs(self.output_dir, exist_ok=True)
        
        metrics = plot_top_k_metrics([self.preds_tsv], [self.labels_tsv], ['Test Model'], 'recall', output_plot)
        
        # Top 10 will have 10 positives, total positives = 50. Recall @ 10 = 10/50 = 0.2
        self.assertAlmostEqual(metrics[0][10], 0.2)
        # Top 50 will have 50 positives, recall = 1.0
        self.assertAlmostEqual(metrics[0][50], 1.0)
        
        self.assertTrue(os.path.exists(output_plot))

    def test_evaluate_all_metrics(self):
        evaluate_all_metrics([self.preds_tsv], [self.labels_tsv], ['Test Model'], self.output_dir)
        
        # Check files exist
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'auroc.png')))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'pr_auc.png')))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'recall_at_k.png')))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'precision_at_k.png')))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'enrichment_at_k.png')))

    def test_evaluate_cv_performance(self):
        # Create a couple of splits
        preds_data1 = {
            'gene': [f'gene_{i}' for i in range(250)],
            'probability': [0.9 if i < 50 else 0.1 for i in range(250)]
        }
        preds_data2 = {
            'gene': [f'gene_{i}' for i in range(250)],
            'probability': [0.8 if i < 50 else 0.2 for i in range(250)]
        }
        pd.DataFrame(preds_data1).to_csv('test_preds_seed_0.tsv', sep='\t', index=False)
        pd.DataFrame(preds_data2).to_csv('test_preds_seed_1.tsv', sep='\t', index=False)
        
        results = evaluate_cv_performance(
            preds_template='test_preds_seed_{i}.tsv',
            labels_tsv=self.labels_tsv,
            k=2,
            start_idx=0,
            output_dir=self.output_dir,
            desc='cv_test'
        )
        
        self.assertAlmostEqual(results['auroc_mean'], 1.0)
        self.assertAlmostEqual(results['auroc_std'], 0.0)
        self.assertAlmostEqual(results['pr_auc_mean'], 1.0)
        self.assertAlmostEqual(results['pr_auc_std'], 0.0)
        
        table_path = os.path.join(self.output_dir, 'cv_test_performance_top_k_comparison.tsv')
        self.assertTrue(os.path.exists(table_path))
        
        df = pd.read_csv(table_path, sep='\t')
        self.assertIn('k', df.columns)
        self.assertIn('#positives_mean', df.columns)
        self.assertIn('recall@k_mean', df.columns)
        self.assertIn('precision@k_std', df.columns)
        self.assertEqual(len(df), 20)

if __name__ == '__main__':
    unittest.main()
