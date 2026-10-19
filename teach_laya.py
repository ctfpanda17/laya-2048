"""教 laya 玩 2048：讓 expectimax 當老師，用模仿學習微調 laya。

步驟:
    python teach_laya.py gen   --games 200    # 1. 老師自己玩，記錄 (盤面, 老師選的方向) → data_2048.jsonl
    python teach_laya.py train --epochs 2     # 2. 微調 laya → laya_2048_model/
    python teach_laya.py eval  --games 20     # 3. 在模擬器裡讓學生自己玩，看分數
    python teach_laya.py dagger --iters 3     # 4. DAgger：學生自己玩 → 強老師 (teacher.py) 標註 → 再訓練 → laya_2048_dagger/

之後用瀏覽器看它玩:
    python play_2048.py --mode laya --model-dir laya_2048_dagger
"""
import argparse
import json
import multiprocessing as mp
import os
import random
import shutil
import time

from play_2048 import (HERE, KEYS, MODEL_DIR, QUESTIONS, laya_state, legal_moves,
                       policy_expectimax, simulate)

DATA_FILE = os.path.join(HERE, "data_2048.jsonl")
OUT_DIR = os.path.join(HERE, "laya_2048_model")
DAGGER_FILE = os.path.join(HERE, "data_dagger.jsonl")
DAGGER_DIR = os.path.join(HERE, "laya_2048_dagger")
DIRS = list(KEYS)  # 與 QUESTIONS 的 criteria 順序相同：up, down, left, right


# ---------- 純 Python 的 2048 模擬器（不需要瀏覽器，快很多） ----------

def spawn(grid, rng):
    empties = [(y, x) for y in range(4) for x in range(4) if grid[y][x] == 0]
    y, x = rng.choice(empties)
    grid[y][x] = 2 if rng.random() < 0.9 else 4


def new_game(rng):
    g = [[0] * 4 for _ in range(4)]
    spawn(g, rng)
    spawn(g, rng)
    return g


def run_game(choose, rng, record=None, explore=0.0):
    """用 choose(grid) -> 方向 玩一局；record 不為 None 時記錄老師的答案。回傳 (分數, 最大方塊, 步數)。"""
    grid, score, steps = new_game(rng), 0, 0
    while True:
        moves = legal_moves(grid)
        if not moves:
            break
        if record is not None:
            teacher = policy_expectimax(None, grid)[0][0]
            record.append({"grid": grid, "move": teacher})
            # 偶爾故意走別步，讓資料涵蓋老師自己不會走到的盤面（學生犯錯後也知道怎麼救）
            d = rng.choice(moves) if rng.random() < explore else teacher
        else:
            d = choose(grid)
            if d not in moves:  # 學生選了不能動的方向 → 跟網頁版一樣改用下一個合法方向
                d = moves[0]
        grid, gained = simulate(grid, d)
        score += gained
        steps += 1
        spawn(grid, rng)
    return score, max(max(r) for r in grid), steps


def _gen_worker(args):
    seed, explore = args
    rng, rec = random.Random(seed), []
    score, best, steps = run_game(None, rng, record=rec, explore=explore)
    return rec, score, best


def cmd_gen(a):
    t = time.time()
    jobs = [(a.seed + i, a.explore) for i in range(a.games)]
    n = 0
    with mp.Pool(a.workers) as pool, open(DATA_FILE, "w", encoding="utf-8") as f:
        for i, (rec, score, best) in enumerate(pool.imap_unordered(_gen_worker, jobs), 1):
            for r in rec:
                f.write(json.dumps(r) + "\n")
            n += len(rec)
            print(f"老師第 {i}/{a.games} 局：分數 {score}，最大 {best}，累計 {n} 筆  ({time.time() - t:.0f}s)")
    print(f"已寫入 {DATA_FILE}")


# ---------- 微調 ----------

def encode(agent, internal, grid):
    return agent._encode_state(laya_state(grid), ["move"], internal)


def finetune(rows, init_dir, out_dir, epochs, lr, batch):
    """從 init_dir 的模型開始，用 rows [{grid, move}] 微調，存到 out_dir。回傳驗證準確率。"""
    import laya
    import torch
    from laya.common import collate_items

    rng = random.Random(0)
    rows = list(rows)
    rng.shuffle(rows)
    n_val = min(2000, len(rows) // 10)
    val, train = rows[:n_val], rows[n_val:]
    print(f"訓練 {len(train)} 筆，驗證 {len(val)} 筆（起點模型: {os.path.basename(init_dir)}）")

    agent = laya.load(init_dir, device="cuda")
    model, dev = agent.model, agent.device
    internal = {"move": agent._to_internal(QUESTIONS["move"])}
    pad = agent.tok.pad_token_id

    def batches(data, bs, shuffle):
        idx = list(range(len(data)))
        if shuffle:
            rng.shuffle(idx)
        for i in range(0, len(idx), bs):
            chunk = [data[j] for j in idx[i:i + bs]]
            b = collate_items([encode(agent, internal, r["grid"]) for r in chunk], pad)
            y = torch.tensor([DIRS.index(r["move"]) for r in chunk], device=dev)
            yield b, y

    def forward(b):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(b["input_ids"].to(dev), b["attention_mask"].to(dev),
                              b["marker_pos"].to(dev), b["marker_mask"].to(dev), b["qtype"].to(dev))
        return logits.float()

    @torch.no_grad()
    def evaluate():
        model.eval()
        ok = 0
        for b, y in batches(val, 128, False):
            ok += (forward(b).argmax(-1) == y).sum().item()
        return ok / len(val)

    print(f"微調前驗證準確率（和老師選一樣的比例）: {evaluate():.3f}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = epochs * ((len(train) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total, pct_start=0.06)
    step, t, acc = 0, time.time(), None
    for ep in range(epochs):
        model.train()
        for b, y in batches(train, batch, True):
            loss = torch.nn.functional.cross_entropy(forward(b), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 1000 == 0:
                print(f"epoch {ep + 1} step {step}/{total} loss {loss.item():.3f}  ({time.time() - t:.0f}s)")
        acc = evaluate()
        print(f"epoch {ep + 1} 結束，驗證準確率 {acc:.3f}")

    # 存成和原模型一樣的資料夾結構，laya.load 可以直接讀。
    # 先寫到暫存資料夾再換名，init_dir == out_dir 時也不會把正在讀的設定檔刪掉。
    from safetensors.torch import save_file
    tmp = out_dir + "_tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    shutil.copytree(init_dir, tmp, ignore=shutil.ignore_patterns("model.safetensors", ".cache"))
    model.eval()
    save_file({k: v.detach().contiguous().cpu() for k, v in model.state_dict().items()},
              os.path.join(tmp, "model.safetensors"))
    del agent, model, opt
    torch.cuda.empty_cache()
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.rename(tmp, out_dir)
    print(f"已存到 {out_dir}")
    return acc


def cmd_train(a):
    rows = [json.loads(l) for l in open(DATA_FILE, encoding="utf-8")]
    finetune(rows, MODEL_DIR, OUT_DIR, a.epochs, a.lr, a.batch)


# ---------- DAgger：學生自己玩、老師標註學生走過的盤面、再訓練 ----------

def student_play(agent, n_games, seed):
    """n_games 局同時進行，每一步把所有盤面一起丟給 GPU。回傳 (走過的盤面, 每局 (分數, 最大方塊, 步數))。"""
    active = []
    for i in range(n_games):
        rng = random.Random(seed + i)
        active.append({"grid": new_game(rng), "rng": rng, "score": 0, "steps": 0})
    visited, results = [], []
    while active:
        outs = agent.predict_batch([laya_state(g["grid"]) for g in active], QUESTIONS, batch_size=128)
        still = []
        for g, out in zip(active, outs):
            probs = out["answers"]["move"]["probabilities"]
            visited.append(g["grid"])
            d = max(legal_moves(g["grid"]), key=probs.get)  # 只在能動的方向裡選
            g["grid"], gained = simulate(g["grid"], d)
            g["score"] += gained
            g["steps"] += 1
            spawn(g["grid"], g["rng"])
            if legal_moves(g["grid"]):
                still.append(g)
            else:
                results.append((g["score"], max(max(r) for r in g["grid"]), g["steps"]))
        active = still
    return visited, results


def _label(grid):
    from teacher import teacher_move
    return {"grid": grid, "move": teacher_move(grid)}


def label_all(grids, workers):
    t = time.time()
    with mp.Pool(workers) as pool:
        rows = pool.map(_label, grids, chunksize=64)
    print(f"老師標註 {len(rows)} 個盤面，花了 {time.time() - t:.0f}s")
    return rows


def summarize(results, title):
    bests = [b for _, b, _ in results]
    avg = sum(s for s, _, _ in results) / len(results)
    tiles = "  ".join(f"{t}:{sum(b >= t for b in bests)}/{len(bests)}" for t in (512, 1024, 2048, 4096))
    print(f"[{title}] 平均分數 {avg:.0f}  達成 {tiles}")
    return avg


def cmd_dagger(a):
    import laya
    import torch

    # 資料集：先用強老師重新標註原本的示範盤面，之後每輪加上學生自己走過的盤面
    if os.path.exists(DAGGER_FILE) and not a.fresh:
        dataset = [json.loads(l) for l in open(DAGGER_FILE, encoding="utf-8")]
        print(f"接續既有 DAgger 資料 {len(dataset)} 筆")
    else:
        expert = [json.loads(l)["grid"] for l in open(DATA_FILE, encoding="utf-8")]
        dataset = label_all(expert, a.workers)
        with open(DAGGER_FILE, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in dataset)

    model_dir = DAGGER_DIR if os.path.exists(DAGGER_DIR) and not a.fresh else OUT_DIR
    for it in range(1, a.iters + 1):
        print(f"\n===== DAgger 第 {it}/{a.iters} 輪 =====")
        agent = laya.load(model_dir, device="cuda")
        t = time.time()
        visited, results = student_play(agent, a.games, seed=100_000 * it + a.seed)
        del agent
        torch.cuda.empty_cache()
        print(f"學生自己玩 {a.games} 局、{len(visited)} 步，花了 {time.time() - t:.0f}s")
        summarize(results, f"第 {it} 輪訓練前的學生")

        new_rows = label_all(visited, a.workers)
        dataset += new_rows
        with open(DAGGER_FILE, "a", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in new_rows)

        finetune(dataset, model_dir, DAGGER_DIR, a.epochs, a.lr, a.batch)
        model_dir = DAGGER_DIR

    agent = laya.load(model_dir, device="cuda")
    _, results = student_play(agent, a.games, seed=999_999)
    summarize(results, "DAgger 完成後的學生")


# ---------- 評估：學生自己玩 ----------

def cmd_eval(a):
    import laya
    import torch
    agent = laya.load(a.model_dir, device="cuda" if torch.cuda.is_available() else "cpu")
    _, results = student_play(agent, a.games, seed=10_000)
    for i, (score, best, steps) in enumerate(results, 1):
        print(f"第 {i} 局：{steps} 步，分數 {score}，最大方塊 {best}")
    summarize(results, os.path.basename(a.model_dir))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--games", type=int, default=60)
    g.add_argument("--explore", type=float, default=0.1)
    g.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    g.add_argument("--seed", type=int, default=0)
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=2)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--lr", type=float, default=2e-5)
    e = sub.add_parser("eval")
    e.add_argument("--games", type=int, default=20)
    e.add_argument("--model-dir", default=OUT_DIR)
    d = sub.add_parser("dagger")
    d.add_argument("--iters", type=int, default=3)
    d.add_argument("--games", type=int, default=64, help="每輪學生自己玩幾局")
    d.add_argument("--epochs", type=int, default=1)
    d.add_argument("--batch", type=int, default=32)
    d.add_argument("--lr", type=float, default=1e-5)
    d.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--fresh", action="store_true", help="忽略之前的 DAgger 資料與模型，重新開始")
    a = ap.parse_args()
    {"gen": cmd_gen, "train": cmd_train, "eval": cmd_eval, "dagger": cmd_dagger}[a.cmd](a)


if __name__ == "__main__":
    main()
