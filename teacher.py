"""比較強的 expectimax 老師。

和 play_2048.policy_expectimax 同樣的蛇形評分，但:
  - 盤面用 16 格 tuple 表示，列的滑動結果有快取，快很多
  - 同一步內重複出現的盤面只算一次（transposition table）
  - 機率很低的分支直接停止展開，省下來的時間拿去算更深
"""
from functools import lru_cache

DIRS = ["up", "down", "left", "right"]  # 與 QUESTIONS 的 criteria 順序相同

# 每個方向的 4 條線，每條線的格子依「滑向的那一端」排序
LINES = {
    "left": [[r * 4 + c for c in range(4)] for r in range(4)],
    "right": [[r * 4 + c for c in (3, 2, 1, 0)] for r in range(4)],
    "up": [[r * 4 + c for r in range(4)] for c in range(4)],
    "down": [[r * 4 + c for r in (3, 2, 1, 0)] for c in range(4)],
}

_SNAKE = [[4 ** 15, 4 ** 14, 4 ** 13, 4 ** 12],
          [4 ** 8, 4 ** 9, 4 ** 10, 4 ** 11],
          [4 ** 7, 4 ** 6, 4 ** 5, 4 ** 4],
          [4 ** 0, 4 ** 1, 4 ** 2, 4 ** 3]]
WEIGHTS = [w for row in _SNAKE for w in row]
PROB_CUTOFF = 1e-4


@lru_cache(maxsize=None)
def _slide(line):
    tiles = [v for v in line if v]
    out, i = [], 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            out.append(tiles[i] * 2)
            i += 2
        else:
            out.append(tiles[i])
            i += 1
    return tuple(out + [0] * (4 - len(out)))


def move(b, d):
    out = list(b)
    for idx in LINES[d]:
        for i, v in zip(idx, _slide(tuple(b[i] for i in idx))):
            out[i] = v
    return tuple(out)


def heuristic(b):
    return sum(w * v for w, v in zip(WEIGHTS, b))


def _max_node(b, depth, prob, cache):
    best = 0.0  # 無路可走 = 輸，給 0 分
    for d in DIRS:
        nb = move(b, d)
        if nb != b:
            best = max(best, _chance_node(nb, depth, prob, cache))
    return best


def _chance_node(b, depth, prob, cache):
    if depth == 0 or prob < PROB_CUTOFF:
        return heuristic(b)
    key = (b, depth)
    if key in cache:
        return cache[key]
    empties = [i for i, v in enumerate(b) if v == 0]
    if not empties:
        return heuristic(b)
    n = len(empties)
    total = 0.0
    lb = list(b)
    for i in empties:
        for tile, p in ((2, 0.9), (4, 0.1)):
            lb[i] = tile
            total += p * _max_node(tuple(lb), depth - 1, prob * p / n, cache)
        lb[i] = 0
    v = total / n
    cache[key] = v
    return v


def search_depth(b):
    # 深度 = 往後看幾次「隨機出新方塊」。實測每步: 深度 2 約 18ms、3 約 290ms、4 約 2 秒。
    # 盤面擠的時候最容易走錯，也因為空格少、分支少，才花得起多看一層。
    return 3 if b.count(0) <= 4 else 2


def teacher_scores(grid):
    """grid: 4x4 list → {方向: 期望分數}，只含合法方向。"""
    b = tuple(v for row in grid for v in row)
    depth = search_depth(b)
    cache = {}
    return {d: _chance_node(move(b, d), depth, 1.0, cache) for d in DIRS if move(b, d) != b}


def teacher_move(grid):
    s = teacher_scores(grid)
    return max(s, key=s.get) if s else "up"
