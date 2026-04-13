import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_score,
    f1_score,
)

from .adapters import Adapter


def evaluate(adapter: Adapter, target_emb: np.ndarray, other_emb: np.ndarray) -> dict:
    """Evaluate a fitted adapter on held-out test embeddings.

    Args:
        adapter: A fitted adapter (threshold already calibrated during fit).
        target_emb: Test embeddings of the target (normal/in-class) speaker.
            Samples here should be *unseen* during fit. Label 0 internally.
        other_emb: Test embeddings of non-target (anomaly/out-of-class) speakers.
            Label 1 internally.

    Returns:
        Dict with ``m_``-prefixed metric keys: recall, precision, f1,
        false_alarm_rate, accuracy, auc, auprc, eer, threshold.
    """
    scores_target = adapter.score(target_emb)
    scores_other = adapter.score(other_emb)

    preds_target = scores_target > adapter.threshold
    preds_other = scores_other > adapter.threshold

    n_target = len(target_emb)
    n_other = len(other_emb)
    false_alarms = preds_target.sum()
    hits = preds_other.sum()

    # label 0 = target (normal), 1 = other (anomaly)
    labels = np.concatenate([np.zeros(n_target), np.ones(n_other)])
    preds = np.concatenate([preds_target, preds_other]).astype(int)
    scores = np.concatenate([scores_target, scores_other])

    auc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)

    # EER: operating point where FAR == FRR (1 - recall)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2)
    threshold_eer = float(thresholds[eer_idx])

    # ACC at target FAR (5%)
    target_far = 0.05
    far_idx = np.argmin(np.abs(fpr - target_far))
    tpr_at_far = tpr[far_idx]
    fpr_at_far = fpr[far_idx]
    acc_at_far = (tpr_at_far * n_other + (1 - fpr_at_far) * n_target) / (n_target + n_other)
    threshold_at_far5 = float(thresholds[far_idx])

    precision = precision_score(labels, preds, zero_division=0)
    recall = hits / n_other
    f1 = f1_score(labels, preds, zero_division=0)

    return {
        "m_recall": recall,
        "m_precision": precision,
        "m_f1": f1,
        "m_false_alarm_rate": false_alarms / n_target,
        "m_accuracy": (hits + n_target - false_alarms) / (n_target + n_other),
        "m_auc": auc,
        "m_auprc": auprc,
        "m_eer": eer,
        "m_acc_at_far5": acc_at_far,
        "m_threshold": adapter.threshold,
        "m_threshold_eer": threshold_eer,
        "m_threshold_at_far5": threshold_at_far5,
        "m_avg_ll": getattr(adapter, "avg_log_likelihood", None),
        "m_n_iter": getattr(getattr(adapter, "_gmm", None), "n_iter_", None),
        "m_inference_macs": adapter.inference_macs(),
        "m_training_macs": adapter.training_macs(),
        "m_inference_flops": adapter.inference_flops(),
        "m_training_flops": adapter.training_flops(),
        **{
            f"m_val_loss_{i+1}": v
            for i, v in enumerate(getattr(adapter, "val_loss_checkpoints", []))
        },
        **{
            f"m_train_loss_{i+1}": v
            for i, v in enumerate(getattr(adapter, "train_loss_checkpoints", []))
        },
    }
