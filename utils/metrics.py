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


def _path_selection(token_ids, E, none_idx, eps=0.5, sos=1, eos=2):
    """
    DAG longest-path selection.

    token_ids: list of corrected token ids from SCH
    E:         [N, N] numpy edge score matrix
    none_idx:  index of ∅ (background) class
    eps:       edge threshold for adjacency
    """
    N = len(token_ids)
    if N == 0:
        return []

    adj = defaultdict(list)
    for i in range(N):
        for j in range(N):
            if i != j and float(E[i, j]) >= eps:
                adj[i].append((j, float(E[i, j])))

    sos_nodes = [i for i, t in enumerate(token_ids) if t == sos]
    eos_nodes = [i for i, t in enumerate(token_ids) if t == eos]

    if not sos_nodes or not eos_nodes:
        return [t for t in token_ids if t not in (none_idx, sos, eos)]

    start, end = sos_nodes[0], eos_nodes[0]
    
    # Bellman-Ford style Longest Path (max N edges)
    # Since PGD edge predictions can (and will) contain cycles, 
    # Topo Sort DP is INVALID. We must relax edges N times.
    dist = [-1e9] * N
    prev = [-1] * N
    dist[start] = 0.0

    for _ in range(N - 1):
        updated = False
        for u in range(N):
            if dist[u] == -1e9: continue
            for v, w in adj[u]:
                if dist[u] + w > dist[v]:
                    dist[v] = dist[u] + w
                    prev[v] = u
                    updated = True
        if not updated:
            break

    # Traceback
    path, cur, seen = [], end, set()
    while cur != -1 and cur not in seen:
        seen.add(cur)
        path.append(cur)
        cur = prev[cur]
    path.reverse()

    if not path or path[0] != start:
        return [t for t in token_ids if t not in (none_idx, sos, eos)]

    return [token_ids[i] for i in path if token_ids[i] not in (sos, eos, none_idx)]
