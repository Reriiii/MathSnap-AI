"""
Metrics for HMER evaluation.

Implements ExpRate (Expression Recognition Rate) and BLEU score.
"""

from typing import List
from collections import Counter


def compute_exprate(
    predictions: List[str],
    targets: List[str],
) -> dict:
    """
    Compute Expression Recognition Rate (ExpRate) and variants.

    ExpRate: exact match rate (whole expression correct)
    ExpRate@1: at most 1 token wrong
    ExpRate@2: at most 2 tokens wrong

    Args:
        predictions: list of predicted LaTeX strings
        targets: list of ground truth LaTeX strings

    Returns:
        dict with 'exprate', 'exprate_1', 'exprate_2'
    """
    exact = 0
    within_1 = 0
    within_2 = 0
    total = len(predictions)

    if total == 0:
        return {'exprate': 0.0, 'exprate_1': 0.0, 'exprate_2': 0.0}

    for pred, tgt in zip(predictions, targets):
        pred_tokens = pred.split()
        tgt_tokens = tgt.split()

        if pred_tokens == tgt_tokens:
            exact += 1
            within_1 += 1
            within_2 += 1
        else:
            # Count edit distance (simple token-level)
            dist = _edit_distance(pred_tokens, tgt_tokens)
            if dist <= 1:
                within_1 += 1
            if dist <= 2:
                within_2 += 1

    return {
        'exprate': exact / total * 100,
        'exprate_1': within_1 / total * 100,
        'exprate_2': within_2 / total * 100,
    }


def _edit_distance(seq1: list, seq2: list) -> int:
    """Compute token-level edit distance (Levenshtein)."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]


def compute_bleu(
    predictions: List[str],
    targets: List[str],
    max_n: int = 4,
) -> dict:
    """
    Compute corpus-level BLEU score.

    Args:
        predictions: list of predicted LaTeX strings
        targets: list of ground truth LaTeX strings
        max_n: maximum n-gram order

    Returns:
        dict with 'bleu', 'bleu_1', 'bleu_2', 'bleu_3', 'bleu_4',
              'brevity_penalty'
    """
    import math

    if not predictions or not targets:
        return {f'bleu_{i}': 0.0 for i in range(1, max_n + 1)}

    clipped_counts = [0] * max_n
    total_counts = [0] * max_n
    total_pred_len = 0
    total_ref_len = 0

    for pred, ref in zip(predictions, targets):
        pred_tokens = pred.split()
        ref_tokens = ref.split()

        total_pred_len += len(pred_tokens)
        total_ref_len += len(ref_tokens)

        for n in range(1, max_n + 1):
            pred_ngrams = _get_ngrams(pred_tokens, n)
            ref_ngrams = _get_ngrams(ref_tokens, n)

            # Clipped count
            for ngram, count in pred_ngrams.items():
                clipped_counts[n - 1] += min(count, ref_ngrams.get(ngram, 0))
            total_counts[n - 1] += sum(pred_ngrams.values())

    # Compute BLEU components
    precisions = []
    for n in range(max_n):
        if total_counts[n] == 0:
            precisions.append(0.0)
        else:
            precisions.append(clipped_counts[n] / total_counts[n])

    # Brevity penalty
    if total_pred_len == 0:
        bp = 0.0
    elif total_pred_len >= total_ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - total_ref_len / total_pred_len)

    # Compute BLEU scores
    result = {'brevity_penalty': bp}

    for n in range(1, max_n + 1):
        # BLEU-n: geometric mean of precisions 1..n
        log_avg = 0.0
        valid = True
        for i in range(n):
            if precisions[i] == 0:
                valid = False
                break
            log_avg += math.log(precisions[i])

        if valid:
            log_avg /= n
            result[f'bleu_{n}'] = bp * math.exp(log_avg) * 100
        else:
            result[f'bleu_{n}'] = 0.0

    # Main BLEU = BLEU-4
    result['bleu'] = result[f'bleu_{max_n}']

    return result


def _get_ngrams(tokens: list, n: int) -> Counter:
    """Extract n-gram counts from a token list."""
    ngrams = Counter()
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        ngrams[ngram] += 1
    return ngrams


def compute_token_accuracy(
    predictions: List[str],
    targets: List[str],
) -> float:
    """
    Compute token-level accuracy.

    For each position, check if predicted token matches target token.
    """
    correct = 0
    total = 0

    for pred, tgt in zip(predictions, targets):
        pred_tokens = pred.split()
        tgt_tokens = tgt.split()

        max_len = max(len(pred_tokens), len(tgt_tokens))
        for i in range(max_len):
            pt = pred_tokens[i] if i < len(pred_tokens) else ""
            tt = tgt_tokens[i] if i < len(tgt_tokens) else ""
            if pt == tt:
                correct += 1
            total += 1

    return (correct / total * 100) if total > 0 else 0.0
