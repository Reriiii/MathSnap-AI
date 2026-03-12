from collections import defaultdict
import numpy as np


def _path_selection(token_ids, E, none_idx, eps=0.5, sos=1, eos=2):
    """
    DAG longest-path selection via topological sort DP.

    Paper Sec 3.4:
      E_ij = right_score(i→j) + left_score(j→i)  — already computed by caller
      Delete edges below eps ONLY IF removing them does not disconnect
      SOS from EOS. Then find longest path via topological sort DP.

    token_ids : list[int]  corrected token ids (length N), includes SOS/EOS
    E         : np.ndarray [N, N]  edge score matrix (from compute_scores)
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

    def _build_adj(threshold):
        """Build adjacency list with edges >= threshold."""
        adj = defaultdict(list)
        for i in range(N):
            for j in range(N):
                if i != j and float(E[i, j]) >= threshold:
                    adj[i].append((j, float(E[i, j])))
        return adj

    def _has_path(adj, src, dst):
        """BFS reachability check."""
        visited = set()
        queue   = [src]
        while queue:
            u = queue.pop()
            if u == dst:
                return True
            if u in visited:
                continue
            visited.add(u)
            for v, _ in adj[u]:
                queue.append(v)
        return False

    # Build full graph first, then prune edges below eps
    # but ONLY if the pruned graph still connects SOS → EOS (paper condition)
    adj_full = _build_adj(0.0)
    adj_pruned = _build_adj(eps)

    adj = adj_pruned if _has_path(adj_pruned, start, end) else adj_full

    # Topological sort DP for longest path (paper: O(V+E), DAG assumed)
    # Use DFS-based topological ordering
    visited   = set()
    topo      = []

    def _dfs(u):
        visited.add(u)
        for v, _ in adj[u]:
            if v not in visited:
                _dfs(v)
        topo.append(u)

    for u in range(N):
        if u not in visited:
            _dfs(u)
    topo.reverse()   # topological order

    # DP: dist[v] = longest path weight from start to v
    dist = {u: float('-inf') for u in range(N)}
    prev = {u: -1 for u in range(N)}
    dist[start] = 0.0

    for u in topo:
        if dist[u] == float('-inf'):
            continue
        for v, w in adj[u]:
            if dist[u] + w > dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u

    # Traceback from end
    if dist[end] == float('-inf'):
        # No path found — fallback to column-sorted visible tokens
        return [t for t in token_ids if t not in (none_idx, sos, eos)]

    path, cur, seen = [], end, set()
    while cur != -1 and cur not in seen:
        seen.add(cur)
        path.append(cur)
        cur = prev[cur]
    path.reverse()

    if not path or path[0] != start:
        return [t for t in token_ids if t not in (none_idx, sos, eos)]

    return [token_ids[i] for i in path if token_ids[i] not in (sos, eos, none_idx)]


def compute_exprate(preds, gts):
    """
    ExpRate, ExpRate≤1, ExpRate≤2.
    preds: list of predicted token lists (decoded)
    gts:   list of ground-truth token lists
    """
    assert len(preds) == len(gts)
    exact = lev1 = lev2 = 0
    for p, g in zip(preds, gts):
        p = list(p); g = list(g)
        if p == g:
            exact += 1; lev1 += 1; lev2 += 1
        elif _edit_distance(p, g) <= 1:
            lev1 += 1; lev2 += 1
        elif _edit_distance(p, g) <= 2:
            lev2 += 1
    n = len(preds)
    return exact / n, lev1 / n, lev2 / n


def _edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]; dp[0] = i
        for j in range(1, n + 1):
            tmp = dp[j]
            dp[j] = prev if s1[i-1] == s2[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = tmp
    return dp[n]