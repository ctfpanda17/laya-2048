"""2048 速度賽：用教好的 laya 盡快在網頁上做出 2048 方塊，計時從第一次按鍵到 2048 出現。

加速重點:
  - 直接呼叫 laya 的模型本體，並用 CUDA Graph 錄下整個運算後重播（每步約 7ms，原本 predict() 約 50ms）
  - 自己判斷哪些方向能動，不浪費按鍵
  - 每步只跟瀏覽器來回一次讀盤面，沒有任何等待

用法:
    python speedrun.py                 # 開視窗跑，成功做出 2048 就停（最多試 10 次）
    python speedrun.py --tries 3       # 最多試 3 次
    python speedrun.py --headless
    python speedrun.py --no-record     # 不錄影

成功的那次會錄成 mp4 存在 videos/，畫面下方有計時器與步數。
"""
import argparse
import os
import shutil
import subprocess
import tempfile
import time

import torch

from play_2048 import KEYS, QUESTIONS, board_text, laya_state, legal_moves, start_game_server

DIRS = list(KEYS)  # 與 QUESTIONS 的 criteria 順序相同：up, down, left, right
MAX_TOKENS = 160   # 固定輸入長度，CUDA Graph 需要固定形狀（實際約 130 個 token）


class FastLaya:
    def __init__(self, model_dir):
        import laya
        from laya.common import collate_items
        self.collate = collate_items
        self.agent = laya.load(model_dir, device="cuda")
        self.model = self.agent.model.eval()
        self.internal = {"move": self.agent._to_internal(QUESTIONS["move"])}
        self.pad = self.agent.tok.pad_token_id

        start = [[2, 4, 8, 16], [0, 0, 2, 4], [0, 0, 0, 2], [0, 0, 0, 0]]
        self.static = [t.cuda() for t in self._encode(start)]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(3):
                self.model(*self.static)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self.graph):
            self.out = self.model(*self.static)[0]

    def _encode(self, grid):
        b = self.collate([self.agent._encode_state(laya_state(grid), ["move"], self.internal)], self.pad)
        n = b["input_ids"].shape[1]
        if n > MAX_TOKENS:
            raise ValueError(f"盤面編碼長度 {n} 超過 {MAX_TOKENS}")
        ids = torch.full((1, MAX_TOKENS), self.pad, dtype=torch.long)
        att = torch.zeros((1, MAX_TOKENS), dtype=torch.long)
        ids[:, :n] = b["input_ids"]
        att[:, :n] = b["attention_mask"]
        return [ids, att, b["marker_pos"], b["marker_mask"], b["qtype"]]

    @torch.inference_mode()
    def choose(self, grid, legal):
        for st, t in zip(self.static, self._encode(grid)):
            st.copy_(t, non_blocking=True)
        self.graph.replay()
        logits = self.out[0, :4].tolist()
        return max(legal, key=lambda d: logits[DIRS.index(d)])


READ_STATE = """() => {
  const raw = localStorage.getItem('gameState');
  if (!raw) return null;
  const s = JSON.parse(raw);
  const grid = [0, 1, 2, 3].map(y => [0, 1, 2, 3].map(x => {
    const c = s.grid.cells[x][y]; return c ? c.value : 0; }));
  return {grid, won: s.won, score: s.score};
}"""


# 疊在遊戲畫面下方的計時器：第一次按鍵開始跑，結束時由 Python 填入正式成績並停住
OVERLAY = """() => {
  const box = document.createElement('div');
  box.id = 'laya-timer';
  box.style.cssText = 'position:fixed;left:0;right:0;bottom:0;padding:10px 16px;z-index:9999;' +
    'background:#2b2622;color:#f9f6f2;font:600 18px/1.3 "Clear Sans",Arial,sans-serif;' +
    'display:flex;justify-content:space-between;align-items:center';
  box.innerHTML = '<span>laya AI · 2048 speedrun</span>' +
    '<span><span id="laya-moves">0</span> moves</span>' +
    '<span id="laya-time" style="font:700 30px monospace">0:00.000</span>';
  document.body.appendChild(box);
  const fmt = ms => { const m = Math.floor(ms / 60000), s = (ms % 60000) / 1000;
    return m + ':' + s.toFixed(3).padStart(6, '0'); };
  let t0 = null, moves = 0, done = false;
  document.addEventListener('keydown', () => {
    if (done) return;
    if (t0 === null) t0 = performance.now();
    document.getElementById('laya-moves').textContent = ++moves;
  }, true);
  (function tick() {
    if (!done && t0 !== null) document.getElementById('laya-time').textContent = fmt(performance.now() - t0);
    requestAnimationFrame(tick);
  })();
  window.layaFinish = (ms, n, won) => {
    done = true;
    document.getElementById('laya-time').textContent = fmt(ms);
    document.getElementById('laya-moves').textContent = n;
    box.style.background = won ? '#edc22e' : '#776e65';
    box.firstChild.textContent = won ? 'laya AI · 2048 reached!' : 'laya AI · game over';
  };
}"""


def one_run(page, brain):
    page.wait_for_function("localStorage.getItem('gameState') !== null")
    page.evaluate(OVERLAY)
    state = page.evaluate(READ_STATE)
    moves, t0 = 0, None
    while state and not state["won"]:
        legal = legal_moves(state["grid"])
        if not legal:
            break
        d = brain.choose(state["grid"], legal)
        if t0 is None:
            t0 = time.perf_counter()  # 第一次按鍵開始計時
        page.keyboard.press(KEYS[d])
        moves += 1
        state = page.evaluate(READ_STATE)
    elapsed = time.perf_counter() - t0
    won = bool(state and state["won"])
    page.evaluate("([ms, n, won]) => window.layaFinish(ms, n, won)", [elapsed * 1000, moves, won])
    return won, elapsed, moves


def fmt_time(sec):
    return f"{int(sec)}s{round((sec % 1) * 1000):03d}ms"


def to_mp4(webm, mp4):
    """Playwright 錄的是 webm；轉成 mp4，YouTube / Bilibili / 論壇都比較好用。沒有 ffmpeg 就保留 webm。"""
    if shutil.which("ffmpeg") is None:
        dst = os.path.splitext(mp4)[0] + ".webm"
        shutil.move(webm, dst)
        return dst
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", webm, "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "20", mp4], check=True)
    os.remove(webm)
    return mp4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="laya_2048_dagger")
    ap.add_argument("--tries", type=int, default=10, help="最多試幾次；第一次成功就停")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--no-record", action="store_true", help="不錄影")
    ap.add_argument("--video-dir", default="videos")
    args = ap.parse_args()

    print("載入 laya 並錄製 CUDA Graph ...")
    brain = FastLaya(args.model_dir)
    url = start_game_server()
    from playwright.sync_api import sync_playwright

    size = {"width": 600, "height": 880}  # 下方留空間給計時器，不擋到棋盤
    tmp_dir = tempfile.mkdtemp(prefix="laya_video_")
    best, video = None, None
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)
        for i in range(1, args.tries + 1):
            # 每次嘗試開新的分頁並各自錄影，成功的留下、失敗的丟掉
            rec = {} if args.no_record else {"record_video_dir": tmp_dir, "record_video_size": size}
            context = browser.new_context(viewport=size, **rec)
            page = context.new_page()
            page.goto(url, wait_until="domcontentloaded")
            ok, sec, moves = one_run(page, brain)
            if ok:
                print(f"第 {i} 次：{sec:.3f} 秒做出 2048（{moves} 步，每步 {sec / moves * 1000:.1f}ms）")
                page.wait_for_timeout(3000)  # 讓影片最後停在成績畫面幾秒
            else:
                print(f"第 {i} 次：失敗，{moves} 步後無路可走（{sec:.3f} 秒），重來")
            webm = page.video.path() if page.video else None
            context.close()  # 關掉分頁時 Playwright 才會把影片寫完
            if webm and ok:
                os.makedirs(args.video_dir, exist_ok=True)
                name = f"laya_2048_{fmt_time(sec)}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                video = to_mp4(webm, os.path.join(args.video_dir, name))
            elif webm:
                os.remove(webm)
            if ok:
                best = sec
                break  # 成功就停
        browser.close()
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print()
    if best is not None:
        print(f"成績：{fmt_time(best)}")
        if video:
            print(f"影片：{video}")
    else:
        print(f"{args.tries} 次都沒做出 2048")


if __name__ == "__main__":
    main()
