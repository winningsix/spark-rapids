#!/usr/bin/env python3
"""Split task-free wall time of each measured median iteration into driver-side phases.

For every query's median iteration (by elapsed_s), inside [start_epoch, end_epoch]:
  - union of task [launch, finish] intervals (all jobs, including async subqueries)
  - each task-free gap is attributed by what the scheduler was doing:
      pre_submit   : no stage submitted-and-unfinished exists (driver planning/AQE/collect)
      submitted    : a stage is submitted but none of its tasks launched yet (DAG/task scheduling,
                     task-binary serialization, resource offers)
  - also report per stage: submission -> first task launch latency.
Usage: analyze-gap-split-v1.py <evidence-dir> [--out file.json]
"""
import argparse, collections, json, statistics, subprocess
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('evidence', type=Path)
ap.add_argument('--out', type=Path)
a = ap.parse_args()
q = a.evidence / 'query'
records = [json.loads(l) for l in (q / 'timings.jsonl').read_text().splitlines()]
groups = collections.defaultdict(list)
for r in records:
    if not r['warmup']:
        groups[r['query']].append(r)
median_iter = {}
for name, items in groups.items():
    items = sorted(items, key=lambda r: r['elapsed_s'])
    median_iter[name] = items[len(items) // 2]

tasks = []          # (launch, finish, stage)
stage_sub = {}      # stage key -> submission time
stage_done = {}
first_launch = {}
for f in list(a.evidence.glob('events/**/events_*')) + list(a.evidence.glob('eventlogs/**/events_*')):
    cmd = ['zstd', '-dc', str(f)] if f.name.endswith('.zstd') else ['cat', str(f)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    for line in proc.stdout:
        e = json.loads(line); k = e['Event']
        if k == 'SparkListenerStageSubmitted':
            si = e['Stage Info']; key = (si['Stage ID'], si['Stage Attempt ID'])
            if si.get('Submission Time'):
                stage_sub[key] = si['Submission Time'] / 1000
        elif k == 'SparkListenerStageCompleted':
            si = e['Stage Info']; key = (si['Stage ID'], si['Stage Attempt ID'])
            if si.get('Submission Time'):
                stage_sub[key] = si['Submission Time'] / 1000
            if si.get('Completion Time'):
                stage_done[key] = si['Completion Time'] / 1000
        elif k == 'SparkListenerTaskEnd':
            ti = e['Task Info']; key = (e['Stage ID'], e['Stage Attempt ID'])
            l, fi = ti['Launch Time'] / 1000, ti['Finish Time'] / 1000
            tasks.append((l, fi, key))
            first_launch[key] = min(first_launch.get(key, l), l)
    proc.wait()

def union(iv):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out

rows = []
tot = collections.Counter()
for name, r in sorted(median_iter.items()):
    s0, e0 = r['start_epoch'], r['end_epoch']
    busy = union([(max(l, s0), min(f, e0)) for l, f, _ in tasks if f > s0 and l < e0])
    # waiting windows: stage submitted but no task launched yet
    waiting = union([(max(stage_sub[k], s0), min(first_launch[k], e0)) for k in first_launch
                     if k in stage_sub and first_launch[k] > s0 and stage_sub[k] < e0 and first_launch[k] > stage_sub[k]])
    gaps, cur = [], s0
    for s, e in busy:
        if s > cur: gaps.append((cur, s))
        cur = max(cur, e)
    if cur < e0: gaps.append((cur, e0))
    free = sum(e - s for s, e in gaps)
    def overlap(gs, ws):
        t = 0.0
        for gs_, ge in gs:
            for ws_, we in ws:
                t += max(0.0, min(ge, we) - max(gs_, ws_))
        return t
    submitted = overlap(gaps, waiting)
    first_t = busy[0][0] if busy else e0; last_t = busy[-1][1] if busy else e0
    before = sum(min(e, first_t) - s for s, e in gaps if s < first_t)
    after = sum(e - max(s, last_t) for s, e in gaps if e > last_t)
    lat = [first_launch[k] - stage_sub[k] for k in first_launch if k in stage_sub and s0 <= stage_sub[k] <= e0]
    row = dict(query=name, elapsed_s=r['elapsed_s'], task_free_s=free, submitted_no_task_s=submitted,
               pre_submit_s=free - submitted, before_first_task_s=before, after_last_task_s=after,
               between_s=free - before - after, stages=len(lat),
               submit_to_first_launch_median_ms=1000 * statistics.median(lat) if lat else None,
               submit_to_first_launch_sum_s=sum(lat))
    rows.append(row)
    for k in ('elapsed_s', 'task_free_s', 'submitted_no_task_s', 'pre_submit_s', 'before_first_task_s', 'between_s', 'after_last_task_s', 'submit_to_first_launch_sum_s'):
        tot[k] += row[k]
    tot['stages'] += len(lat)
summary = dict(evidence=str(a.evidence), queries=len(rows), totals=dict(tot),
               note='Median iteration per query. task_free = no task running from any job. submitted_no_task = '
                    'part of task_free during which some stage was submitted but had not launched its first task. '
                    'pre_submit = remainder (planning, AQE re-optimization, broadcast/collect on driver, result delivery).',
               top_task_free=sorted(rows, key=lambda x: -x['task_free_s'])[:15], rows=rows)
text = json.dumps(summary, indent=1)
if a.out:
    a.out.write_text(text + '\n')
print(json.dumps(dict(totals=summary['totals'], top=[(x['query'], round(x['task_free_s'], 3), round(x['pre_submit_s'], 3),
      round(x['submitted_no_task_s'], 3), x['stages']) for x in summary['top_task_free']]), indent=1))
