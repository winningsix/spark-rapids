#!/usr/bin/env python3
"""Validate the frozen workload, result rows, runtime health and aggregate timing."""
from pathlib import Path
import argparse
import collections
import csv
import datetime
import hashlib
import json
import math
import statistics
import subprocess
from urllib.parse import urlparse, unquote

BUNDLE=Path(__file__).resolve().parents[1]
read=lambda p:json.loads(p.read_text())

def equal_rows(name, actual, expected):
    if actual==expected:
        return True
    if name not in ['query39_part1','query39_part2']:
        return False
    def keyed(rows):
        result={tuple(row[i] for i in (0,1,2,5,6,7)):row for row in rows}
        assert len(result)==len(rows)
        return result
    a=keyed(actual);b=keyed(expected)
    if a.keys()!=b.keys():return False
    return all(x==y or isinstance(x,float) and isinstance(y,float) and
        math.isfinite(x) and math.isfinite(y) and math.isclose(x,y,rel_tol=1e-12,abs_tol=1e-12)
        for k in a for x,y in zip(a[k],b[k]))

def validate(root, output):
    q=root/'query';manifest=read(q/'manifest.json');reference=read(BUNDLE/'evidence/reference-queries.json')
    expected=read(BUNDLE/'manifest.json')
    assert manifest['spark_version']=='4.0.2'
    assert manifest['query_sha256']==expected['sql_sha256']
    assert manifest['args']['warmup']==1 and manifest['args']['iterations']==3
    assert not manifest['args']['cold'] and manifest['args']['table_statistics']=='all_columns'
    assert manifest['args']['direct_file_index']
    assert read(q/'schemas.json')==read(BUNDLE/'evidence/expected-schemas.json'), 'Table schema drift'
    conf=manifest['spark_conf']
    artifact=Path(conf['spark.driver.extraClassPath'])
    assert conf['spark.executor.extraClassPath']==str(artifact)
    jar_uri=urlparse(conf['spark.jars'])
    assert jar_uri.scheme in ['', 'file'] and ',' not in conf['spark.jars']
    submitted_jar=Path(unquote(jar_uri.path))
    if artifact.is_file():
        assert submitted_jar.resolve()==artifact.resolve()
        with artifact.open('rb') as stream:
            assert hashlib.file_digest(stream,'sha256').hexdigest()==expected['artifact_sha256']
    else:
        # Allows offline validation of an archived AWS run using its original audit.
        audit=read(root/'executor-artifact-audit.json')
        assert audit['all_copies_match'] and len(audit['copies'])==5
        assert audit['expected_sha256']==expected['artifact_sha256']
        assert submitted_jar.name==artifact.name
    gpus=int(read(root/'preflight.json').get('gpu_count',4)) if (root/'preflight.json').exists() else 4
    assert gpus in (2,4)
    template=dict(line.split(None,1) for line in (BUNDLE/'config/spark-202s.properties.in').read_text().splitlines())
    if gpus!=4:template.update({'spark.cores.max':str(10*gpus),'spark.executor.instances':str(gpus)})
    for key,value in template.items():
        if '@@' not in value:
            assert conf.get(key)==value,(key,conf.get(key),value)
    assert conf.get('spark.sql.hive.manageFilesourcePartitions')=='false'
    assert str(conf.get('spark.sql.adaptive.coalescePartitions.parallelismFirst','true')).lower()=='true'
    groups=collections.defaultdict(list)
    records=[json.loads(line) for line in (q/'timings.jsonl').read_text().splitlines()]
    for record in records:
        assert record['status']=='ok' and record['executor_count']==gpus
        groups[record['query']].append(record)
    assert set(groups)==set(manifest['query_names'])<=set(reference)
    if set(groups)==set(reference):
        assert manifest['query_names']==list(reference),'Full-run query order changed'
    medians={};answer_checks=[]
    for name,items in groups.items():
        assert sorted(r['iteration'] for r in items)==[0,1,2,3]
        assert [r['iteration'] for r in items if r['warmup']]==[0]
        medians[name]=statistics.median(r['elapsed_s'] for r in items if not r['warmup'])
        ref=[json.loads(line) for line in (BUNDLE/'evidence/reference-rows'/f'{name}-i1.jsonl').read_text().splitlines() if line]
        for r in items:
            text=(q/'rows'/f"{name}-i{r['iteration']}.jsonl").read_text()
            actual=[json.loads(line) for line in text.splitlines() if line]
            # The harness hashes its payload before adding the trailing newline.
            assert hashlib.sha256(text[:-1].encode()).hexdigest()==r['result_sha256']
            assert len(actual)==r['row_count']
            assert equal_rows(name,actual,ref),(name,r['iteration'],'Answer differs from AWS reference')
        answer_checks.append(name)
    executors=set();removals=[];last_end=0;ended=False;tasks=0
    events=list((root/'events').glob('**/events_*.zstd'))+list((root/'eventlogs').glob('**/events_*.zstd'))
    assert events,'Missing Spark event logs (install the zstd CLI to read them)'
    for file in events:
        process=subprocess.Popen(['zstd','-dc',str(file)],stdout=subprocess.PIPE,text=True)
        for line in process.stdout:
            event=json.loads(line);kind=event['Event']
            if kind=='SparkListenerExecutorAdded':executors.add(event['Executor ID'])
            elif kind=='SparkListenerExecutorRemoved':removals.append(event)
            elif kind=='SparkListenerTaskEnd':
                tasks+=1
                assert event['Task End Reason']['Reason']=='Success',event['Task End Reason']
                assert event['Task Info']['Attempt']==0,'Task retry'
                assert event.get('Task Metrics',{}).get('Disk Bytes Spilled',0)==0,'Spark disk spill'
            elif kind=='SparkListenerStageCompleted':
                assert not event['Stage Info'].get('Failure Reason'),event['Stage Info']
            elif kind=='SparkListenerJobEnd':
                assert event['Job Result']['Result']=='JobSucceeded',event['Job Result']
                last_end=max(last_end,event['Completion Time'])
            elif kind=='SparkListenerApplicationEnd':ended=True
        assert process.wait()==0
    assert len(executors)==gpus and ended and tasks>0
    assert not [e for e in removals if e['Timestamp']<=last_end]
    samples=[]
    with (q/'gpu.csv').open() as stream:
        for row in csv.reader(stream):
            try:
                epoch=datetime.datetime.strptime(row[0].strip(),'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=datetime.timezone.utc).timestamp()
                samples.append((epoch,int(row[1]),float(row[3])))
            except (ValueError,IndexError):pass
    # Executors may be bound to any physical GPU indices; report the busiest gpus devices.
    overall={g:statistics.mean(u for t,i,u in samples if i==g) for g in {i for t,i,u in samples}}
    used=sorted(sorted(overall,key=lambda g:-overall[g])[:gpus])
    busy=[]
    for r in records:
        if r['warmup']:continue
        per_gpu=[[u for t,g,u in samples if g==gpu and r['start_epoch']<=t<=r['end_epoch']] for gpu in used]
        assert all(per_gpu),'GPU telemetry does not cover all measured queries; check TZ=UTC'
        busy.append((r['elapsed_s'],statistics.mean(statistics.mean(v) for v in per_gpu)))
    total=sum(medians.values());full=set(groups)==set(reference)
    result=dict(validated=True,gpu_count=gpus,gpu_indices_reported=used,full103=full,queries=len(groups),records=len(records),
        sum_of_per_query_medians_s=total,aws_reference_s=sum(reference[n]['median_s'] for n in groups),
        gpu_query_window_mean_pct=sum(t*u for t,u in busy)/sum(t for t,u in busy),
        comparison='Original SQL and schemas, every result compared against the frozen AWS GPU reference; Q39a/b tolerance 1e-12',
        no_task_failures_retries_midrun_executor_loss_or_spark_disk_spill=True,
        note='Fresh CPU equivalence and RAPIDS allocation/retry counters require separate review; see driver/executor logs.',
        per_query_medians=medians)
    output.mkdir(parents=True,exist_ok=True)
    (output/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    with (output/'comparison.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['query','tencent_median_s','aws_median_s','aws_divided_by_tencent'])
        for name,value in medians.items():writer.writerow([name,value,reference[name]['median_s'],reference[name]['median_s']/value])
    print(json.dumps({k:v for k,v in result.items() if k!='per_query_medians'},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();validate(args.run,args.output or args.run/'validation')
