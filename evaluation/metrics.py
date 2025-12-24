import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

class FEREvaluator:
    def __init__(self, num_classes=14):
        self.num_classes = num_classes
        self.emotion_names = [
            "Happily surprised", "Happily disgusted", "Sadly fearful", "Sadly angry",
            "Sadly surprised", "Sadly disgusted", "Fearfully angry", "Fearfully surprised",
            "Fearfully disgusted", "Angrily surprised", "Angrily disgusted",
            "Disgustedly surprised", "Happily fearful", "Happily sad"
        ]

    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate Accuracy, Macro F1, and Weighted F1.
        """
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        
        report = classification_report(y_true, y_pred, target_names=self.emotion_names, output_dict=True)
        
        return {
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "full_report": report
        }

    def plot_confusion_matrix(self, y_true, y_pred, save_path="confusion_matrix.png"):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=self.emotion_names, yticklabels=self.emotion_names, cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def evaluate_au_coverage(self, y_true, y_pred, au_labels):
        """
        Logic to evaluate if predicted emotion matches the AU activation.
        This is a placeholder for the multi-task analysis.
        """
        # TODO: Implement correlation between predicted emotion labels and AUs
        pass

if __name__ == "__main__":
    # Test metrics
    evaluator = FEREvaluator()
    y_true = [0, 1, 2, 5, 10]
    y_pred = [0, 1, 3, 5, 10]
    metrics = evaluator.calculate_metrics(y_true, y_pred)
    print(f"Test F1 Macro: {metrics['f1_macro']:.4f}")
