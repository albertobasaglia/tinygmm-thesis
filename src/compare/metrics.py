import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from .adapters import Adapter


def evaluate(adapter: Adapter, target_emb: np.ndarray, other_emb: np.ndarray) -> dict:
    """Evaluate a fitted adapter on held-out test embeddings.

    Reports only threshold-free metrics: AUC, AUPRC, EER, and accuracy at a
    fixed 5% FAR operating point read off the test ROC.

    Args:
        adapter: A fitted adapter.
        target_emb: Test embeddings of the target (normal/in-class). Label 0.
        other_emb:  Test embeddings of non-target (anomaly/out-of-class). Label 1.
    """
    scores_target = adapter.score(target_emb)
    scores_other = adapter.score(other_emb)

    n_target = len(target_emb)
    n_other = len(other_emb)

    # label 0 = target (normal), 1 = other (anomaly)
    labels = np.concatenate([np.zeros(n_target), np.ones(n_other)])
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

    return {
        "m_auc": auc,
        "m_auprc": auprc,
        "m_eer": eer,
        "m_acc_at_far5": acc_at_far,
        "m_threshold_eer": threshold_eer,
        "m_threshold_at_far5": threshold_at_far5,
        "m_avg_ll": getattr(adapter, "avg_log_likelihood", None),
        "m_n_iter": getattr(getattr(adapter, "_gmm", None), "n_iter_", None),
        "m_inference_flops": adapter.inference_flops(),
        "m_training_flops": adapter.training_flops(),
        "m_parameters": adapter.parameters(),
        **{
            f"m_val_loss_{i+1}": v
            for i, v in enumerate(getattr(adapter, "val_loss_checkpoints", []))
        },
        **{
            f"m_train_loss_{i+1}": v
            for i, v in enumerate(getattr(adapter, "train_loss_checkpoints", []))
        },
    }
