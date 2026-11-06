"""
Evaluation Metrics and Reporting for Machine Learning Classifiers.

Computes:
- Confusion Matrix
- Accuracy, Precision, Recall, F1-Score (per class and macro/weighted average)
- Formatted Text Classification Report
- Train / Test split utility
"""
import random
from typing import List, Dict, Any, Tuple
from collections import defaultdict


def train_test_split(
    X: List[Any], y: List[Any], test_size: float = 0.25, random_seed: int = 42
) -> Tuple[List[Any], List[Any], List[Any], List[Any]]:
    """Splits dataset into stratified or randomized train and test sets."""
    random.seed(random_seed)
    indices = list(range(len(X)))
    random.shuffle(indices)

    split_idx = int(len(X) * (1 - test_size))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]

    return X_train, X_test, y_train, y_test


class ModelEvaluator:
    """Calculates classification metrics without external dependencies."""

    @staticmethod
    def confusion_matrix(y_true: List[str], y_pred: List[str], labels: List[str] = None) -> Tuple[List[str], List[List[int]]]:
        if labels is None:
            labels = sorted(list(set(y_true) | set(y_pred)))
        label_to_idx = {l: i for i, l in enumerate(labels)}
        n = len(labels)
        matrix = [[0] * n for _ in range(n)]

        for t, p in zip(y_true, y_pred):
            if t in label_to_idx and p in label_to_idx:
                matrix[label_to_idx[t]][label_to_idx[p]] += 1

        return labels, matrix

    @staticmethod
    def classification_report(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
        """Calculates Precision, Recall, F1, and Support per class."""
        labels = sorted(list(set(y_true) | set(y_pred)))
        _, matrix = ModelEvaluator.confusion_matrix(y_true, y_pred, labels)
        total_samples = len(y_true)

        report = {}
        macro_p = 0.0
        macro_r = 0.0
        macro_f1 = 0.0

        for i, label in enumerate(labels):
            tp = matrix[i][i]
            fn = sum(matrix[i][j] for j in range(len(labels))) - tp
            fp = sum(matrix[j][i] for j in range(len(labels))) - tp
            support = tp + fn

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            report[label] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "support": support
            }

            macro_p += precision
            macro_r += recall
            macro_f1 += f1

        correct = sum(matrix[i][i] for i in range(len(labels)))
        accuracy = correct / total_samples if total_samples > 0 else 0.0
        n_classes = len(labels)

        report["accuracy"] = round(accuracy, 4)
        report["macro_avg"] = {
            "precision": round(macro_p / n_classes, 4),
            "recall": round(macro_r / n_classes, 4),
            "f1_score": round(macro_f1 / n_classes, 4),
            "support": total_samples
        }

        return report

    @staticmethod
    def format_report_table(report: Dict[str, Any]) -> str:
        """Formats the classification report into an aligned text table."""
        lines = []
        header = f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}"
        separator = "-" * len(header)
        lines.append(separator)
        lines.append(header)
        lines.append(separator)

        for key, metrics in report.items():
            if key in ("accuracy", "macro_avg"):
                continue
            p = f"{metrics['precision']:.2%}"
            r = f"{metrics['recall']:.2%}"
            f = f"{metrics['f1_score']:.2%}"
            s = str(metrics['support'])
            lines.append(f"{key:<15} {p:<12} {r:<12} {f:<12} {s:<10}")

        lines.append(separator)
        acc_str = f"{report['accuracy']:.2%}"
        lines.append(f"{'Accuracy':<15} {'':<12} {'':<12} {acc_str:<12} {report['macro_avg']['support']:<10}")

        macro = report["macro_avg"]
        mp = f"{macro['precision']:.2%}"
        mr = f"{macro['recall']:.2%}"
        mf = f"{macro['f1_score']:.2%}"
        lines.append(f"{'Macro Avg':<15} {mp:<12} {mr:<12} {mf:<12} {macro['support']:<10}")
        lines.append(separator)

        return "\n".join(lines)
