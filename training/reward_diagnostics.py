"""Read-only metrics and gates for reward/termination diagnostic datasets."""
from __future__ import annotations
import numpy as np


def horizon_pairs(transition_count: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """One episode, including its final successor; never crosses a reset.

    With N actions and N+1 observations, action[target] labels node[target+1].
    In particular, the final true terminal is retained at every feasible H.
    """
    if horizon < 1 or transition_count < 0:
        raise ValueError('Expected positive horizon and nonnegative transition count')
    start = np.arange(max(0, transition_count - horizon + 1), dtype=np.int64)
    return start, start + horizon - 1


def reward_terminal_metrics(reward, terminal, reward_symlog_prediction, continuation,
                            train_symlog_mean: float) -> dict:
    reward, terminal, prediction, continuation = [np.asarray(x).reshape(-1) for x in
        [reward, terminal, reward_symlog_prediction, continuation]]
    if not (len(reward) == len(terminal) == len(prediction) == len(continuation)) or not len(reward):
        raise ValueError('Metrics require nonempty aligned arrays')
    if not all(np.isfinite(x).all() for x in [reward, prediction, continuation]):
        raise ValueError('Metrics cannot hide non-finite predictions')
    if not np.isin(terminal, [0, 1]).all() or ((continuation < 0) | (continuation > 1)).any():
        raise ValueError('Expected binary terminals and probability of continuation')
    terminal = terminal.astype(bool)
    target = np.sign(reward) * np.log1p(np.abs(reward))
    mse = np.mean((prediction - target)**2)
    baseline = np.mean((train_symlog_mean - target)**2)
    failure = terminal & (reward < 0)
    predict_terminal = continuation <= .5
    def rate(value, mask):return float(np.mean(value[mask])) if mask.any() else None
    return dict(n=len(reward),terminal_n=int(terminal.sum()),nonterminal_n=int((~terminal).sum()),
                negative_failure_terminal_n=int(failure.sum()),symlog_mse=float(mse),
                mean_baseline_ratio=float(mse / baseline) if baseline > 0 else None,
                terminal_recall=rate(predict_terminal,terminal),
                false_positive_rate=rate(predict_terminal,~terminal),
                negative_failure_terminal_recall=rate(prediction < 0,failure),
                continuation_brier=float(np.mean((continuation-(~terminal))**2)))


def reward_terminal_gate(metrics: dict) -> bool:
    """Frozen report thresholds; absent classes do not count as a pass."""
    fields=['mean_baseline_ratio','terminal_recall','false_positive_rate','negative_failure_terminal_recall']
    if any(metrics.get(k) is None for k in fields):return False
    return (metrics['mean_baseline_ratio'] < 1 and metrics['terminal_recall'] >= .9
            and metrics['false_positive_rate'] <= .05 and metrics['negative_failure_terminal_recall'] >= .9)
