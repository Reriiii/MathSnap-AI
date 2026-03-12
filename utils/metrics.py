from collections import defaultdict


def _edit_dist(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        ndp = [i] + [0] * n
        for j in range(1, n + 1):
            ndp[j] = dp[j-1] if a[i-1] == b[j-1] else 1 + min(dp[j], ndp[j-1], dp[j-1])
        dp = ndp
    return dp[n]


def compute_exprate(preds, gts):
    ex = l1 = l2 = 0
    for p, g in zip(preds, gts):
        d = _edit_dist(p, g)
        if d == 0: ex += 1
        if d <= 1: l1 += 1
        if d <= 2: l2 += 1
    N = max(len(preds), 1)
    return ex / N, l1 / N, l2 / N


def _path_selection(token_ids, E, none_idx, sos=1, eos=2):
    """
    Graph path selection using argmax right-neighbor pointers.

    token_ids : list[int]  corrected token ids from SCH (length N)
    E         : np.ndarray [N, N]  edge_scores = right_scores + left_scores.T
                right_scores[i,j] = P(right neighbor of i is j)  (softmax, sums to 1)
                E[i,j] is highest when both i thinks j is its right and j thinks i is its left.

    WHY NOT eps threshold:
      right_scores is a softmax over N tokens.  For N=20, uniform ≈ 0.05.
      eps=0.5 would filter virtually every edge → Bellman-Ford sees empty graph
      → always falls back to column-sorted list → PGD completely bypassed.

    CORRECT approach: use argmax right-pointer to build a deterministic chain.
      right_ptr[i] = argmax_j E[i,j]   (best right neighbor of i)
      Then trace: SOS → right_ptr[SOS] → ... → EOS
    """
    N = len(token_ids)
    if N == 0:
        return []

    sos_nodes = [i for i, t in enumerate(token_ids) if t == sos]
    eos_nodes = [i for i, t in enumerate(token_ids) if t == eos]

    if not sos_nodes or not eos_nodes:
        return [t for t in token_ids if t not in (none_idx, sos, eos)]

    start = sos_nodes[0]
    end   = eos_nodes[0]

    # Build right-pointer chain from argmax of edge matrix
    right_ptr = E.argmax(axis=1)   # [N]  for each node, best right neighbor

    path = []
    cur  = start
    seen = set()
    while cur != end and cur not in seen and len(path) < N + 2:
        seen.add(cur)
        path.append(cur)
        cur = int(right_ptr[cur])

    if cur == end:
        path.append(end)

    # Valid path must start at SOS and end at EOS
    if len(path) >= 2 and path[0] == start and path[-1] == end:
        return [token_ids[i] for i in path if token_ids[i] not in (sos, eos, none_idx)]

    # Fallback: column-sorted order (same as before, but now only triggers on cycle)
    return [t for t in token_ids if t not in (none_idx, sos, eos)]