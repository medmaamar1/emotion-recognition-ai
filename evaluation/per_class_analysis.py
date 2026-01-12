"""
Per-class metrics analysis for detailed model evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_recall_fscore_support, roc_auc_score, roc_curve, auc
)
import torch


class PerClassAnalyzer:
    """
    Per-class metrics analyzer for emotion recognition.
    
    Args:
        num_classes: Number of classes (default: 14)
        class_names: List of class names (default: None)
    """
    def __init__(self, num_classes=14, class_names=None):
        self.num_classes = num_classes
        
        if class_names is None:
            self.class_names = [
                "Happily surprised", "Happily disgusted", "Sadly fearful", "Sadly angry",
                "Sadly surprised", "Sadly disgusted", "Fearfully angry", "Fearfully surprised",
                "Fearfully disgusted", "Angrily surprised", "Angrily disgusted",
                "Disgustedly surprised", "Happily fearful", "Happily sad"
            ]
        else:
            self.class_names = class_names
    
    def analyze(self, y_true, y_pred, y_prob=None):
        """
        Perform comprehensive per-class analysis.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional, for ROC-AUC)
        
        Returns:
            Dictionary with per-class metrics
        """
        # Compute per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Compute per-class accuracy
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        per_class_acc = np.nan_to_num(per_class_acc, nan=0.0)
        
        # Compute ROC-AUC if probabilities provided
        roc_auc_scores = None
        if y_prob is not None:
            try:
                # One-hot encode true labels
                y_true_onehot = np.zeros((len(y_true), self.num_classes))
                y_true_onehot[np.arange(len(y_true)), y_true] = 1
                
                # Compute ROC-AUC for each class
                roc_auc_scores = []
                for i in range(self.num_classes):
                    if len(np.unique(y_true_onehot[:, i])) > 1:
                        auc_score = roc_auc_score(y_true_onehot[:, i], y_prob[:, i])
                        roc_auc_scores.append(auc_score)
                    else:
                        roc_auc_scores.append(np.nan)
                
                roc_auc_scores = np.array(roc_auc_scores)
            except:
                roc_auc_scores = None
        
        # Compile results
        results = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
            'accuracy': per_class_acc,
            'confusion_matrix': cm,
            'roc_auc': roc_auc_scores
        }
        
        return results
    
    def get_summary_table(self, y_true, y_pred, y_prob=None):
        """
        Get summary table of per-class metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
        
        Returns:
            Dictionary with summary table
        """
        results = self.analyze(y_true, y_pred, y_prob)
        
        # Create summary table
        summary = {
            'Class': self.class_names,
            'Precision': results['precision'],
            'Recall': results['recall'],
            'F1-Score': results['f1'],
            'Accuracy': results['accuracy'],
            'Support': results['support']
        }
        
        if results['roc_auc'] is not None:
            summary['ROC-AUC'] = results['roc_auc']
        
        return summary
    
    def plot_per_class_metrics(self, y_true, y_pred, save_path='per_class_metrics.png'):
        """
        Plot per-class metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save plot (default: 'per_class_metrics.png')
        """
        results = self.analyze(y_true, y_pred)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Per-Class Metrics Analysis', fontsize=16)
        
        # Precision, Recall, F1
        x = np.arange(self.num_classes)
        width = 0.25
        
        axes[0, 0].bar(x - width, results['precision'], width, label='Precision', alpha=0.8)
        axes[0, 0].bar(x, results['recall'], width, label='Recall', alpha=0.8)
        axes[0, 0].bar(x + width, results['f1'], width, label='F1-Score', alpha=0.8)
        axes[0, 0].set_xlabel('Class')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Precision, Recall, and F1-Score')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(self.class_names, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].bar(x, results['accuracy'], color='green', alpha=0.8)
        axes[0, 1].set_xlabel('Class')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Per-Class Accuracy')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(self.class_names, rotation=45, ha='right')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Support
        axes[1, 0].bar(x, results['support'], color='orange', alpha=0.8)
        axes[1, 0].set_xlabel('Class')
        axes[1, 0].set_ylabel('Support')
        axes[1, 0].set_title('Class Support (Number of Samples)')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(self.class_names, rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Confusion Matrix
        sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', 
                   xticklabels=self.class_names, yticklabels=self.class_names, 
                   cmap='Blues', ax=axes[1, 1])
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')
        axes[1, 1].set_title('Confusion Matrix')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curves(self, y_true, y_prob, save_path='roc_curves.png'):
        """
        Plot ROC curves for each class.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            save_path: Path to save plot (default: 'roc_curves.png')
        """
        # One-hot encode true labels
        y_true_onehot = np.zeros((len(y_true), self.num_classes))
        y_true_onehot[np.arange(len(y_true)), y_true] = 1
        
        # Create subplots
        n_rows = (self.num_classes + 3) // 4
        fig, axes = plt.subplots(n_rows, 4, figsize=(20, 5 * n_rows))
        fig.suptitle('ROC Curves for Each Class', fontsize=16)
        
        axes = axes.flatten()
        
        # Plot ROC curve for each class
        for i in range(self.num_classes):
            if len(np.unique(y_true_onehot[:, i])) > 1:
                fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                
                axes[i].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
                axes[i].plot([0, 1], [0, 1], 'k--', linewidth=1)
                axes[i].set_xlim([0.0, 1.0])
                axes[i].set_ylim([0.0, 1.05])
                axes[i].set_xlabel('False Positive Rate')
                axes[i].set_ylabel('True Positive Rate')
                axes[i].set_title(f'{self.class_names[i]}')
                axes[i].legend(loc='lower right')
                axes[i].grid(True, alpha=0.3)
            else:
                axes[i].text(0.5, 0.5, 'Not enough samples', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{self.class_names[i]}')
        
        # Hide unused subplots
        for i in range(self.num_classes, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def get_class_balance_analysis(self, y_true):
        """
        Analyze class balance in the dataset.
        
        Args:
            y_true: True labels
        
        Returns:
            Dictionary with class balance information
        """
        unique, counts = np.unique(y_true, return_counts=True)
        total = len(y_true)
        
        class_balance = {}
        for i in range(self.num_classes):
            if i in unique:
                count = counts[unique == i][0]
                percentage = (count / total) * 100
            else:
                count = 0
                percentage = 0.0
            
            class_balance[self.class_names[i]] = {
                'count': int(count),
                'percentage': percentage,
                'is_imbalanced': percentage < (100 / self.num_classes) * 0.5
            }
        
        return class_balance
    
    def plot_class_balance(self, y_true, save_path='class_balance.png'):
        """
        Plot class balance.
        
        Args:
            y_true: True labels
            save_path: Path to save plot (default: 'class_balance.png')
        """
        class_balance = self.get_class_balance_analysis(y_true)
        
        classes = list(class_balance.keys())
        counts = [class_balance[c]['count'] for c in classes]
        percentages = [class_balance[c]['percentage'] for c in classes]
        is_imbalanced = [class_balance[c]['is_imbalanced'] for c in classes]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar plot
        colors = ['red' if imb else 'blue' for imb in is_imbalanced]
        ax1.bar(classes, counts, color=colors, alpha=0.7)
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Count')
        ax1.set_title('Class Distribution')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='blue', alpha=0.7, label='Balanced'),
                        Patch(facecolor='red', alpha=0.7, label='Imbalanced')]
        ax1.legend(handles=legend_elements)
        
        # Pie chart
        ax2.pie(percentages, labels=classes, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Class Percentage')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def get_worst_performing_classes(self, y_true, y_pred, metric='f1', top_k=5):
        """
        Get worst performing classes based on a metric.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            metric: Metric to use ('precision', 'recall', 'f1', 'accuracy')
            top_k: Number of worst classes to return (default: 5)
        
        Returns:
            List of tuples (class_name, metric_value)
        """
        results = self.analyze(y_true, y_pred)
        
        if metric not in results:
            raise ValueError(f"Unknown metric: {metric}")
        
        metric_values = results[metric]
        
        # Sort by metric value
        sorted_indices = np.argsort(metric_values)
        
        worst_classes = []
        for i in sorted_indices[:top_k]:
            worst_classes.append((self.class_names[i], metric_values[i]))
        
        return worst_classes
    
    def get_best_performing_classes(self, y_true, y_pred, metric='f1', top_k=5):
        """
        Get best performing classes based on a metric.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            metric: Metric to use ('precision', 'recall', 'f1', 'accuracy')
            top_k: Number of best classes to return (default: 5)
        
        Returns:
            List of tuples (class_name, metric_value)
        """
        results = self.analyze(y_true, y_pred)
        
        if metric not in results:
            raise ValueError(f"Unknown metric: {metric}")
        
        metric_values = results[metric]
        
        # Sort by metric value (descending)
        sorted_indices = np.argsort(metric_values)[::-1]
        
        best_classes = []
        for i in sorted_indices[:top_k]:
            best_classes.append((self.class_names[i], metric_values[i]))
        
        return best_classes


def generate_evaluation_report(y_true, y_pred, y_prob=None, 
                          class_names=None, output_dir='./evaluation'):
    """
    Generate comprehensive evaluation report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        class_names: List of class names (default: None)
        output_dir: Directory to save report (default: './evaluation')
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Create analyzer
    analyzer = PerClassAnalyzer(num_classes=14, class_names=class_names)
    
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=analyzer.class_names, 
                              output_dict=True, zero_division=0)
    
    # Save classification report
    import json
    with open(os.path.join(output_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
    
    # Plot per-class metrics
    analyzer.plot_per_class_metrics(
        y_true, y_pred, 
        save_path=os.path.join(output_dir, 'per_class_metrics.png')
    )
    
    # Plot ROC curves if probabilities provided
    if y_prob is not None:
        analyzer.plot_roc_curves(
            y_true, y_prob,
            save_path=os.path.join(output_dir, 'roc_curves.png')
        )
    
    # Plot class balance
    analyzer.plot_class_balance(
        y_true,
        save_path=os.path.join(output_dir, 'class_balance.png')
    )
    
    # Get worst and best performing classes
    worst_classes = analyzer.get_worst_performing_classes(y_true, y_pred, metric='f1', top_k=5)
    best_classes = analyzer.get_best_performing_classes(y_true, y_pred, metric='f1', top_k=5)
    
    # Create summary text file
    summary_path = os.path.join(output_dir, 'evaluation_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("EVALUATION SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        f.write("Overall Metrics:\n")
        f.write(f"  Accuracy: {report['accuracy']:.4f}\n")
        f.write(f"  Macro Avg Precision: {report['macro avg']['precision']:.4f}\n")
        f.write(f"  Macro Avg Recall: {report['macro avg']['recall']:.4f}\n")
        f.write(f"  Macro Avg F1: {report['macro avg']['f1-score']:.4f}\n")
        f.write(f"  Weighted Avg Precision: {report['weighted avg']['precision']:.4f}\n")
        f.write(f"  Weighted Avg Recall: {report['weighted avg']['recall']:.4f}\n")
        f.write(f"  Weighted Avg F1: {report['weighted avg']['f1-score']:.4f}\n\n")
        
        f.write("Best Performing Classes (F1-Score):\n")
        for i, (class_name, f1) in enumerate(best_classes, 1):
            f.write(f"  {i}. {class_name}: {f1:.4f}\n")
        f.write("\n")
        
        f.write("Worst Performing Classes (F1-Score):\n")
        for i, (class_name, f1) in enumerate(worst_classes, 1):
            f.write(f"  {i}. {class_name}: {f1:.4f}\n")
    
    print(f"Evaluation report saved to: {output_dir}")


if __name__ == "__main__":
    # Test per-class analysis
    print("Testing per-class metrics analysis...")
    
    # Generate dummy data
    np.random.seed(42)
    n_samples = 1000
    num_classes = 14
    
    y_true = np.random.randint(0, num_classes, n_samples)
    y_pred = y_true.copy()
    
    # Add some noise to predictions
    noise_indices = np.random.choice(n_samples, size=100, replace=False)
    y_pred[noise_indices] = np.random.randint(0, num_classes, 100)
    
    # Generate dummy probabilities
    y_prob = np.random.dirichlet(np.ones(num_classes), size=n_samples)
    
    # Test PerClassAnalyzer
    print("\nTesting PerClassAnalyzer...")
    analyzer = PerClassAnalyzer(num_classes=num_classes)
    
    results = analyzer.analyze(y_true, y_pred, y_prob)
    print(f"Precision shape: {results['precision'].shape}")
    print(f"Recall shape: {results['recall'].shape}")
    print(f"F1 shape: {results['f1'].shape}")
    print(f"Confusion matrix shape: {results['confusion_matrix'].shape}")
    print(f"ROC-AUC shape: {results['roc_auc'].shape if results['roc_auc'] is not None else 'None'}")
    
    # Test get_summary_table
    print("\nTesting get_summary_table...")
    summary = analyzer.get_summary_table(y_true, y_pred, y_prob)
    print(f"Summary keys: {summary.keys()}")
    
    # Test worst/best performing classes
    print("\nTesting worst/best performing classes...")
    worst = analyzer.get_worst_performing_classes(y_true, y_pred, metric='f1', top_k=3)
    best = analyzer.get_best_performing_classes(y_true, y_pred, metric='f1', top_k=3)
    print(f"Worst performing: {worst}")
    print(f"Best performing: {best}")
    
    # Test class balance analysis
    print("\nTesting class balance analysis...")
    class_balance = analyzer.get_class_balance_analysis(y_true)
    print(f"Number of classes: {len(class_balance)}")
    print(f"Sample class balance: {list(class_balance.items())[0]}")
    
    # Test generate_evaluation_report
    print("\nTesting generate_evaluation_report...")
    generate_evaluation_report(y_true, y_pred, y_prob, output_dir='./test_evaluation')
    
    print("\nAll per-class analysis utilities tested successfully!")
