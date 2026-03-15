"""
Classification metrics: precision, recall, F1, confusion matrix.
"""


def compute_metrics(confusion: dict) -> dict:
    """Confusion matrix에서 precision, recall, F1 계산.

    Args:
        confusion: {"LONG": {"LONG": TP, "SHORT": .., "NEUTRAL": ..}, ...}
                   rows=actual, cols=predicted

    Returns:
        per_class metrics + macro averages
    """
    labels = ["LONG", "SHORT", "NEUTRAL"]
    metrics = {}

    for label in labels:
        tp = confusion[label][label]
        # FP: 다른 actual인데 이 label로 예측한 것
        fp = sum(confusion[other][label] for other in labels if other != label)
        # FN: 이 label이 actual인데 다른 것으로 예측한 것
        fn = sum(confusion[label][other] for other in labels if other != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = sum(confusion[label].values())

        metrics[label] = {
            "precision": round(precision * 100, 1),
            "recall": round(recall * 100, 1),
            "f1": round(f1 * 100, 1),
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    # Macro average
    macro_precision = sum(m["precision"] for m in metrics.values()) / len(labels)
    macro_recall = sum(m["recall"] for m in metrics.values()) / len(labels)
    macro_f1 = sum(m["f1"] for m in metrics.values()) / len(labels)

    metrics["macro_avg"] = {
        "precision": round(macro_precision, 1),
        "recall": round(macro_recall, 1),
        "f1": round(macro_f1, 1),
    }

    return metrics


def print_classification_report(confusion: dict, name: str = ""):
    """Classification report 출력."""
    metrics = compute_metrics(confusion)
    labels = ["LONG", "SHORT", "NEUTRAL"]

    if name:
        print(f"\n  --- {name} Classification Report ---")
    else:
        print(f"\n  --- Classification Report ---")

    print(f"  {'':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"  {'-'*50}")
    for label in labels:
        m = metrics[label]
        print(f"  {label:>10} {m['precision']:>9.1f}% {m['recall']:>9.1f}% {m['f1']:>9.1f}% {m['support']:>10}")

    print(f"  {'-'*50}")
    ma = metrics["macro_avg"]
    total_support = sum(metrics[l]["support"] for l in labels)
    print(f"  {'macro avg':>10} {ma['precision']:>9.1f}% {ma['recall']:>9.1f}% {ma['f1']:>9.1f}% {total_support:>10}")

    # Confusion matrix
    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"  {'':>10} {'LONG':>8} {'SHORT':>8} {'NEUTRAL':>8}")
    for actual in labels:
        row = confusion[actual]
        print(f"  {actual:>10} {row['LONG']:>8} {row['SHORT']:>8} {row['NEUTRAL']:>8}")

    return metrics
