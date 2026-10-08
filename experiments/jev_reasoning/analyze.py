"""
Summarise finished arms: the hops x loops accuracy grid for each, and the two
numbers that answer the question.

    loop gain   -- for each hops, best accuracy over loops minus accuracy at the
                   training depth (8): does spending more loops help?
    depth used  -- the fewest loops reaching 95% of that hops' best accuracy:
                   do harder problems use more loops?

    python analyze.py results/*.json
"""

import json
import sys

from train import EVAL_LOOPS


def main(paths):
    for path in paths:
        r = json.load(open(path))
        g, H = r["grid"], r["args"]["eval_hops"]
        train_h = r["args"]["train_hops"]
        print(f"\n== {r['name']}  ({r['params']:,} params, {r['train_minutes']:.0f} min)")
        print("acc       hops→ " + " ".join(f"{h:>4d}" for h in range(1, H + 1)))
        for n in EVAL_LOOPS:
            row = [g[f"{h},{n}"]["acc"] for h in range(1, H + 1)]
            print(f"loops {n:>3d}       " + " ".join(f"{a:4.2f}" for a in row))
        gain, used = [], []
        for h in range(1, H + 1):
            accs = {n: g[f"{h},{n}"]["acc"] for n in EVAL_LOOPS}
            best = max(accs.values())
            gain.append(best - accs[8])
            used.append(min(n for n, a in accs.items() if a >= 0.95 * best))
        print("gain over 8     " + " ".join(f"{x:+4.2f}" for x in gain))
        print("loops for 95%   " + " ".join(f"{x:>4d}" for x in used))
        ece8 = [g[f"{h},8"]["ece"] for h in range(1, train_h + 1)]
        noul8 = [g[f"{h},8"]["noul_acc"] for h in range(1, train_h + 1)]
        print(
            f"in-range (hops<={train_h}) at 8 loops: "
            f"ECE {sum(ece8) / len(ece8):.3f}, noul acc {sum(noul8) / len(noul8):.2f}"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
