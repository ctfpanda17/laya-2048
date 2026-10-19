import multiprocessing as mp, random, sys, time
from teach_laya import run_game
from play_2048 import policy_expectimax
from teacher import teacher_move

def old(g): return policy_expectimax(None, g)[0][0]
def job(a):
    name, seed = a
    t = time.time()
    s, b, n = run_game(teacher_move if name == "new" else old, random.Random(seed))
    return name, s, b, n, time.time() - t

if __name__ == "__main__":
    N = int(sys.argv[1])
    with mp.Pool(18) as p:
        res = p.map(job, [(k, 5000 + i) for k in ("old", "new") for i in range(N)])
    for k in ("old", "new"):
        r = [x for x in res if x[0] == k]
        bests = [x[2] for x in r]
        print(k, "平均分數", sum(x[1] for x in r) // N, "| 2048:", sum(b >= 2048 for b in bests), "/", N,
              "| 4096:", sum(b >= 4096 for b in bests), "| 每步 ms", round(1000 * sum(x[4] for x in r) / sum(x[3] for x in r), 1))
