#!/usr/bin/env python3
"""Align DCGM dmon samples (epoch-prefixed) with measured query windows; report utilization per arm."""
import json, sys, bisect, collections, statistics
dcgm, arms = sys.argv[1], sys.argv[2:]
S = collections.defaultdict(list)  # gpu -> [(t, gputl, smact, smocc, drama, pcitx, pcirx, power)]
for l in open(dcgm):
    p = l.split()
    if len(p) >= 10 and p[1] == 'GPU':
        try: S[int(p[2])].append((float(p[0]), float(p[3]), float(p[4]), float(p[5]), float(p[6]), float(p[7]), float(p[8]), float(p[9])))
        except ValueError: pass
T = {g: [x[0] for x in v] for g, v in S.items()}
out = {}
for ev in arms:
    recs = [json.loads(l) for l in open(ev + '/query/timings.jsonl')]
    meas = [r for r in recs if not r['warmup'] and r['status'] == 'ok']
    acc = collections.defaultdict(list); perq = collections.defaultdict(list)
    for r in meas:
        for g in S:
            i, j = bisect.bisect_left(T[g], r['start_epoch']), bisect.bisect_right(T[g], r['end_epoch'])
            for x in S[g][i:j]:
                acc[g].append(x); perq[r['query']].append(x)
    if not any(acc.values()): continue
    allx = [x for g in acc for x in acc[g]]
    m = lambda xs, k: statistics.mean(x[k] for x in xs)
    res = dict(samples=len(allx), gpu_util_pct=m(allx, 1), sm_active=m(allx, 2), sm_occupancy=m(allx, 3),
               dram_active=m(allx, 4), pcie_tx_MBps=m(allx, 5) / 1e6, pcie_rx_MBps=m(allx, 6) / 1e6, power_W=m(allx, 7),
               frac_samples_gpu_idle=sum(x[1] == 0 for x in allx) / len(allx),
               frac_samples_smact_ge_0_5=sum(x[2] >= 0.5 for x in allx) / len(allx),
               per_gpu_sm_active={g: m(v, 2) for g, v in sorted(acc.items())})
    q = sorted(((n, len(v), m(v, 2), m(v, 3), m(v, 1)) for n, v in perq.items()), key=lambda z: -z[1])
    res['longest_queries'] = [dict(query=n, samples=c, sm_active=round(a, 3), sm_occ=round(o, 3), gpu_util=round(u, 1)) for n, c, a, o, u in q[:12]]
    out[ev] = res
print(json.dumps(out, indent=1))
