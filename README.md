# laya 玩 2048

教文字分類模型 [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) 玩網頁版 2048：
先用 expectimax 搜尋當老師做模仿學習，再用 DAgger 讓它越玩越強，最後用 CUDA Graph 加速，
**11 秒內做出 2048 方塊**。

![做出 2048 的最後畫面](docs/demo_end.png)

示範影片：[docs/demo.mp4](docs/demo.mp4)（11.279 秒、923 步）

## 成果

在模擬器裡自己玩 64 局（同一組隨機種子）：

| 階段 | 平均分數 | 做到 2048 | 做到 4096 |
|---|---|---|---|
| 原始 laya（沒教過） | ~1,000 | 0 | 0 |
| 模仿 expectimax 之後 | 15,000 | 17/64 | 1/64 |
| DAgger 3 輪 | 42,720 | 48/64 | 27/64 |
| DAgger 5 輪 | 45,379 | 51/64 | 33/64 |

速度賽（[speedrun.py](speedrun.py)）：每步約 12ms，約 950 步做出 2048，**約 11 秒**。

## 怎麼做的

1. **把 2048 變成四選一的問題**：盤面寫成文字，問 laya「上、下、左、右哪一步最好」。
   laya 是一次前向傳播就給出各選項機率的分類模型（ModernBERT-large，4.2 億參數），原本完全不會玩，表現跟亂按差不多。
2. **模仿學習**（[teach_laya.py](teach_laya.py) `gen` / `train`）：讓 expectimax 搜尋自己玩，記錄每個盤面它選的方向，拿來微調 laya。
3. **DAgger**（`teach_laya.py dagger`）：讓 laya 自己玩，再請更強的老師（[teacher.py](teacher.py)，平均 10.8 萬分）標註 laya 走過的每個盤面，
   把資料累加後繼續訓練。這樣 laya 也學到「自己犯錯後該怎麼救」，比單純模仿有效得多。
4. **速度賽加速**（[speedrun.py](speedrun.py)）：跳過 laya 套件的包裝直接呼叫模型，並用 CUDA Graph 錄下整個運算後重播，
   每步推論從約 50ms 降到約 7ms；再加上只按得動的方向、每步只跟瀏覽器來回一次。

遊戲本身是 Gabriele Cirulli 開源的原版 2048（[game2048/](game2048/)，MIT 授權），在本機架起來用 Playwright 操作。

## 使用方式

需要 Python 3.10+、NVIDIA 顯卡（訓練與速度賽）。

```bash
pip install -r requirements.txt
python -m playwright install chromium
```

### 1. 訓練出模型

模型權重太大沒有放在 repo 裡，依序執行即可重新訓練（RTX 4090 上全部約 7–8 小時）：

```bash
python teach_laya.py gen --games 200     # 老師示範，產生 data_2048.jsonl
python teach_laya.py train --epochs 2    # 模仿學習 → laya_2048_model/
python teach_laya.py dagger --iters 5    # DAgger → laya_2048_dagger/
python teach_laya.py eval --games 64 --model-dir laya_2048_dagger
```

第一次執行會自動從 Hugging Face 下載原始 laya 到 `laya_model/`。

### 2. 看它玩

```bash
python play_2048.py --mode laya --model-dir laya_2048_dagger   # 教過的 laya
python play_2048.py --mode laya                                # 原始 laya（對照）
python play_2048.py --mode expectimax                          # 老師（不用模型）
```

### 3. 速度賽 + 錄影

```bash
python speedrun.py              # 做出 2048 就停（最多試 10 次），成功那次錄成 mp4 到 videos/
python speedrun.py --no-record  # 不錄影
```

計時從第一次按鍵開始，到 2048 方塊出現為止；畫面下方會顯示計時器與步數。

## 關於 speedrun.com

這是程式在玩，**不符合** [speedrun.com/2048](https://www.speedrun.com/2048) 的規則（禁止巨集、只收真人輸入，
且必須在指定的遊戲版本上玩），請不要把這裡的成績上傳到真人排行榜。

## 檔案

| 檔案 | 用途 |
|---|---|
| [play_2048.py](play_2048.py) | 開瀏覽器玩 2048；`--mode laya / lookahead / expectimax` |
| [teach_laya.py](teach_laya.py) | 產生資料、微調、DAgger、評估 |
| [teacher.py](teacher.py) | 較強的 expectimax 老師（DAgger 標註用） |
| [speedrun.py](speedrun.py) | 速度賽：CUDA Graph 加速 + 錄影 |
| [bench_teacher.py](bench_teacher.py) | 比較新舊老師的強度與速度 |
| [game2048/](game2048/) | 原版 2048 網頁（MIT，© Gabriele Cirulli） |

## 致謝與授權

本專案程式碼以 [MIT](LICENSE) 授權發布。

- 模型：[convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)（Apache-2.0）
- 遊戲：[gabrielecirulli/2048](https://github.com/gabrielecirulli/2048)（MIT），授權條款見 [game2048/LICENSE.txt](game2048/LICENSE.txt)
