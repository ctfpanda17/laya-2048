"""用 convaiinnovations/laya 模型自動玩網頁版 2048。

流程：在本機架起原版 2048 網頁 (game2048/，gabrielecirulli/2048，MIT) → 用 Playwright 開瀏覽器
→ 從 localStorage 讀盤面 → 把盤面寫成文字丟給 laya 做四選一 (up/down/left/right)
→ 依機率由高到低嘗試按方向鍵，盤面沒變就換下一個方向。

用法:
    python play_2048.py              # 開瀏覽器視窗看它玩
    python play_2048.py --games 3    # 連玩 3 局
    python play_2048.py --headless --delay 0
"""
import argparse
import functools
import http.server
import json
import os
import threading
import time

import laya
from huggingface_hub import snapshot_download

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "laya_model")
GAME_DIR = os.path.join(HERE, "game2048")
KEYS = {"up": "ArrowUp", "down": "ArrowDown", "left": "ArrowLeft", "right": "ArrowRight"}

QUESTIONS = {
    "move": {
        "type": "choice",
        "instructions": (
            "You are playing 2048. Tiles with equal numbers merge when they slide into each other. "
            "Keep the largest tile in a corner and keep empty cells. Which move is best?"
        ),
        "criteria": {
            "up": "slide all tiles up",
            "down": "slide all tiles down",
            "left": "slide all tiles left",
            "right": "slide all tiles right",
        },
    }
}


def load_model(model_dir=MODEL_DIR):
    # 直接下載到本機資料夾，避開 Windows 上 HF 快取需要 symlink 權限的問題
    if model_dir == MODEL_DIR and not os.path.exists(os.path.join(MODEL_DIR, "model.safetensors")):
        snapshot_download(
            "convaiinnovations/laya",
            local_dir=MODEL_DIR,
            allow_patterns=["*.json", "*.py", "model.safetensors", "encoder/*", "tokenizer/*"],
        )
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    print(f"載入 laya 模型 {os.path.basename(model_dir)} ({device}) ...")
    return laya.load(model_dir, device=device)


def start_game_server():
    """在背景用隨機埠提供 game2048/ 靜態網頁，回傳網址。"""
    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    handler = functools.partial(QuietHandler, directory=GAME_DIR)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}/index.html"


def read_grid(page):
    """回傳 (grid[row][col], score)；grid 裡 0 代表空格。遊戲結束後原版會清掉存檔，此時回傳 (None, 0)。"""
    raw = page.evaluate("localStorage.getItem('gameState')")
    if not raw:
        return None, 0
    s = json.loads(raw)
    cells = s["grid"]["cells"]  # cells[x][y]，x 是欄、y 是列
    size = s["grid"]["size"]
    grid = [[(cells[x][y] or {}).get("value", 0) for x in range(size)] for y in range(size)]
    return grid, s.get("score", 0)


def is_over(page):
    return page.evaluate("document.querySelector('.game-message').classList.contains('game-over')")


def board_text(grid):
    lines = [f"row{i + 1}: " + " ".join(str(v) if v else "." for v in row) for i, row in enumerate(grid)]
    return "\n".join(lines)


# ---------- 2048 規則模擬（不含隨機出新方塊） ----------

def slide_row_left(row):
    tiles = [v for v in row if v]
    out, gained, i = [], 0, 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            out.append(tiles[i] * 2)
            gained += tiles[i] * 2
            i += 2
        else:
            out.append(tiles[i])
            i += 1
    return out + [0] * (len(row) - len(out)), gained


def simulate(grid, direction):
    """回傳 (移動後盤面, 這步合併得分)。"""
    g = [list(r) for r in grid]
    if direction in ("up", "down"):
        g = [list(c) for c in zip(*g)]  # 轉置：欄變列
    if direction in ("right", "down"):
        g = [r[::-1] for r in g]
    rows, gained = [], 0
    for r in g:
        nr, s = slide_row_left(r)
        rows.append(nr)
        gained += s
    if direction in ("right", "down"):
        rows = [r[::-1] for r in rows]
    if direction in ("up", "down"):
        rows = [list(c) for c in zip(*rows)]
    return rows, gained


def legal_moves(grid):
    return [d for d in KEYS if simulate(grid, d)[0] != grid]


# ---------- 模式 1：laya 直接看盤面選方向 ----------

def laya_state(grid):
    """盤面 → laya 的輸入。teach_laya.py 訓練時也用這個，兩邊格式必須一致。"""
    return {"game": "2048 puzzle, 4x4 board, '.' is empty", "board": board_text(grid)}


def policy_laya(agent, grid):
    probs = agent.predict(laya_state(grid), QUESTIONS)["answers"]["move"]["probabilities"]
    info = " ".join(f"{k}:{v:.2f}" for k, v in probs.items())
    return sorted(probs, key=probs.get, reverse=True), info


# ---------- 模式 2：先模擬每個方向，再讓 laya 比較結果 ----------

def describe_outcome(grid, direction):
    after, gained = simulate(grid, direction)
    empty = sum(v == 0 for r in after for v in r)
    top = max(max(r) for r in after)
    corners = (after[0][0], after[0][3], after[3][0], after[3][3])
    corner = "yes" if top in corners else "no"
    return (f"after {direction}: {empty} empty cells, merges gain {gained} points, "
            f"largest tile {top} in a corner: {corner}; board " +
            " / ".join(" ".join(str(v) if v else "." for v in r) for r in after))


def policy_lookahead(agent, grid):
    moves = legal_moves(grid)
    if len(moves) <= 1:
        return moves or list(KEYS), "只有一個合法方向"
    state = {"game": "2048 puzzle. Each candidate move below was simulated one step ahead.",
             "candidates": "\n".join(describe_outcome(grid, d) for d in moves)}
    q = {"move": {
        "type": "choice",
        "instructions": ("Pick the move whose resulting 2048 board is best: more empty cells, "
                         "more merged points, and the largest tile kept in a corner."),
        "criteria": {d: describe_outcome(grid, d).split(";")[0] for d in moves},
    }}
    probs = agent.predict(state, q)["answers"]["move"]["probabilities"]
    info = " ".join(f"{k}:{v:.2f}" for k, v in probs.items())
    return sorted(probs, key=probs.get, reverse=True), info


# ---------- 模式 3：expectimax 搜尋（不用 laya） ----------

# 蛇形權重：鼓勵大方塊沿著左上角排成一條蛇
SNAKE = [[4 ** 15, 4 ** 14, 4 ** 13, 4 ** 12],
         [4 ** 8, 4 ** 9, 4 ** 10, 4 ** 11],
         [4 ** 7, 4 ** 6, 4 ** 5, 4 ** 4],
         [4 ** 0, 4 ** 1, 4 ** 2, 4 ** 3]]


def heuristic(grid):
    return sum(SNAKE[y][x] * grid[y][x] for y in range(4) for x in range(4))


def _max_node(grid, depth):
    best = None
    for d in KEYS:
        after, _ = simulate(grid, d)
        if after != grid:
            v = _chance_node(after, depth)
            best = v if best is None else max(best, v)
    return best if best is not None else 0  # 無路可走 = 輸了


def _chance_node(grid, depth):
    if depth == 0:
        return heuristic(grid)
    empties = [(y, x) for y in range(4) for x in range(4) if grid[y][x] == 0]
    if not empties:
        return heuristic(grid)
    total = 0.0
    for y, x in empties:
        for tile, p in ((2, 0.9), (4, 0.1)):
            grid[y][x] = tile
            total += p * _max_node(grid, depth - 1)
        grid[y][x] = 0
    return total / len(empties)


def policy_expectimax(_agent, grid):
    empty = sum(v == 0 for r in grid for v in r)
    depth = 3 if empty <= 3 else 2  # 空格少時分支少，可以多看一層
    scores = {}
    for d in KEYS:
        after, _ = simulate(grid, d)
        if after != grid:
            scores[d] = _chance_node([list(r) for r in after], depth - 1)
    if not scores:
        return list(KEYS), "無路可走"
    order = sorted(scores, key=scores.get, reverse=True)
    return order + [d for d in KEYS if d not in scores], f"depth={depth}"


POLICIES = {"laya": policy_laya, "lookahead": policy_lookahead, "expectimax": policy_expectimax}


# ---------- 瀏覽器操作 ----------

def play_one_game(page, agent, policy, delay):
    page.evaluate("localStorage.removeItem('gameState')")
    page.reload(wait_until="domcontentloaded")
    page.wait_for_selector(".tile-container", state="attached")
    page.wait_for_function("localStorage.getItem('gameState') !== null")

    moves, fallbacks = 0, 0
    grid, score = read_grid(page)
    while not is_over(page):
        # 原版達到 2048 會跳出「You win」，按 Keep going 繼續玩
        if page.evaluate("document.querySelector('.game-message').classList.contains('game-won')"):
            page.click(".keep-playing-button")
        order, info = policy(agent, grid)
        for i, direction in enumerate(order):
            page.keyboard.press(KEYS[direction])
            if is_over(page):
                new_grid = None
                break
            new_grid, new_score = read_grid(page)
            if new_grid != grid:
                break
        else:
            break  # 四個方向都動不了（理論上此時遊戲已結束）

        moves += 1
        fallbacks += i > 0
        tag = "" if i == 0 else f"  (第 {i + 1} 選擇，前面的方向無效)"
        print(f"#{moves:4d} {direction:5s} [{info}] score={score}{tag}")
        if new_grid is None:
            break
        grid, score = new_grid, new_score
        if delay:
            time.sleep(delay)

    final_score = int(page.inner_text(".score-container").split()[0])
    best = max(max(r) for r in grid)
    print("\n最後盤面 (結束前一步):\n" + board_text(grid))
    print(f"遊戲結束：{moves} 步，分數 {final_score}，最大方塊 {best}，第一選擇無效 {fallbacks} 次\n")
    return final_score, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=POLICIES, default="laya",
                    help="laya=直接問模型; lookahead=先模擬再問模型; expectimax=搜尋演算法(不用模型)")
    ap.add_argument("--model-dir", default=MODEL_DIR,
                    help="laya 模型資料夾；訓練後的模型在 laya_2048_model")
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--delay", type=float, default=0.15, help="每步之間的秒數")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    agent = None if args.mode == "expectimax" else load_model(args.model_dir)
    policy = POLICIES[args.mode]
    url = start_game_server()
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)
        page = browser.new_page(viewport={"width": 600, "height": 800})
        page.goto(url, wait_until="domcontentloaded")
        results = [play_one_game(page, agent, policy, args.delay) for _ in range(args.games)]
        if len(results) > 1:
            print("各局結果 (分數, 最大方塊):", results)
        if not args.headless:
            try:
                input("按 Enter 關閉瀏覽器...")
            except EOFError:  # 沒有鍵盤輸入（例如在背景執行）時，保留視窗一陣子
                time.sleep(20)
        browser.close()


if __name__ == "__main__":
    main()
