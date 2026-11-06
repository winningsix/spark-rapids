"""TPC-DS runner using upstream SQL parsing/table registration, plus result and time evidence."""
import argparse
import datetime
import decimal
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

BENCH_ROOT = Path(os.environ.get('BENCH_ROOT', '/mnt/bench')).resolve()
sys.path.insert(0, str(BENCH_ROOT / 'benchmarks' / 'nds'))
from nds_power import gen_sql_from_stream, setup_tables
from benchmark_statistics import analyze_table
from pyspark.sql import SparkSession

def canonical(value):
    if isinstance(value, decimal.Decimal):
        text = format(value, 'f')
        if '.' in text:
            text = text.rstrip('0').rstrip('.')
        return {'decimal': '0' if text == '-0' else text}
    if isinstance(value, (datetime.date, datetime.datetime)):
        return {'date': value.isoformat()}
    if isinstance(value, (list, tuple)):
        return [canonical(x) for x in value]
    if isinstance(value, dict):
        return {k: canonical(v) for k,v in value.items()}
    return value

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',default=(BENCH_ROOT / 'data').as_uri())
    p.add_argument('--queries',default='all')
    p.add_argument('--output',required=True)
    p.add_argument('--warmup',type=int,default=1)
    p.add_argument('--iterations',type=int,default=3)
    p.add_argument('--cold',action='store_true')
    p.add_argument('--timeout',type=int,default=600)
    p.add_argument('--continue-on-failure',action='store_true')
    p.add_argument('--table-statistics', choices=['none', 'table', 'all_columns'], default='none')
    p.add_argument('--direct-file-index', action='store_true')
    p.add_argument('--expected-executors', type=int, default=4)
    args=p.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
    (out/'plans').mkdir(); (out/'rows').mkdir()
    sqlpath=BENCH_ROOT / 'queries.sql'
    queries=gen_sql_from_stream(str(sqlpath))
    if args.queries!='all':
        wanted=args.queries.split(',')
        assert all(x in queries for x in wanted), (wanted,list(queries))
        queries={x:queries[x] for x in wanted}
    assert not any('_setup' in x or '_cleanup' in x for x in queries), 'This runner expects pure SELECT stream'
    spark=SparkSession.builder.appName('g5-cuDF-TPCDS-SF1000-'+out.name).getOrCreate()
    spark.sparkContext.setLogLevel('WARN')
    deadline=time.monotonic()+180
    while spark.sparkContext._jsc.sc().getExecutorMemoryStatus().size()!=args.expected_executors+1:
        if time.monotonic()>deadline: raise RuntimeError(f'Expected {args.expected_executors} executors plus driver')
        time.sleep(1)
    identity={'application_id':spark.sparkContext.applicationId,'spark_version':spark.version,
              'args':vars(args),'query_names':list(queries),'query_sha256':hashlib.sha256(sqlpath.read_bytes()).hexdigest(),
              'spark_conf':dict(spark.sparkContext.getConf().getAll()),'start_epoch':time.time(),
              'metric':'nvidia-smi utilization.gpu; time average across all four GPUs within query execution windows'}
    (out/'manifest.json').write_text(json.dumps(identity,indent=2))
    # This harness runs four standalone workers on one host. Validate actual
    # executor block-manager paths, not just the submitted spark.local.dir.
    scratch_root=(BENCH_ROOT / 'spark-local').resolve()
    scratch_checks=[]
    for log in BENCH_ROOT.glob(f'worker*/{spark.sparkContext.applicationId}/*/stderr'):
        directories=re.findall(r'(?:Shim)?DiskBlockManager: Created local directory at (\S+)',log.read_text())
        for directory in directories:
            resolved=Path(directory).resolve()
            scratch_checks.append({'log':str(log),'directory':directory,
                'on_expected_scratch_path':resolved.is_relative_to(scratch_root),
                'on_non_root_filesystem':resolved.exists() and resolved.stat().st_dev!=Path('/').stat().st_dev})
    (out/'executor-scratch-validation.json').write_text(json.dumps(scratch_checks,indent=2))
    if len({c['log'] for c in scratch_checks})!=args.expected_executors or not all(c['on_expected_scratch_path'] and c['on_non_root_filesystem'] for c in scratch_checks):
        spark.stop()
        raise RuntimeError('Executor scratch directories are not verified on local NVMe; refusing benchmark')
    statistics_start = time.perf_counter()
    setup_steps = []
    if args.table_statistics == 'none':
        setup_tables(spark, args.dataset, 'parquet', True, setup_steps)
    else:
        setup_tables(spark, args.dataset, 'parquet', True, setup_steps,
                     analyze=args.table_statistics)
        for table in spark.catalog.listTables():
            assert not table.isTemporary, table.name
            before = time.perf_counter()
            analyze_table(spark, args.table_statistics, table.name)
            setup_steps.append((spark.sparkContext.applicationId,
                                'Analyze ' + table.name, (time.perf_counter() - before) * 1000))
    statistics_audit = {
        'mode': args.table_statistics,
        'setup_seconds': time.perf_counter() - statistics_start,
        'setup_in_query_execution_time': False,
        'steps': setup_steps,
        'tables': {table.name: [row.asDict(recursive=True) for row in spark.sql(
            'DESCRIBE TABLE EXTENDED `' + table.name.replace('`', '``') + '`').collect()]
            for table in spark.catalog.listTables()}}
    if args.table_statistics != 'none':
        catalog = spark._jsparkSession.sessionState().catalog()
        parser = spark._jsparkSession.sessionState().sqlParser()
        statistics_audit['catalog_statistics'] = {
            table.name: str(catalog.getTableMetadata(
                parser.parseTableIdentifier(table.name)).stats())
            for table in spark.catalog.listTables()}
    (out / 'statistics-setup.json').write_text(json.dumps(statistics_audit, indent=2, default=str))
    if args.direct_file_index:
        assert args.table_statistics == 'all_columns', 'Retain complete catalog statistics'
        index_setup_start = time.perf_counter()
        before_stats = statistics_audit['catalog_statistics']
        spark.conf.set('spark.sql.hive.manageFilesourcePartitions', 'false')
        index_classes = {}
        after_stats = {}
        for table in spark.catalog.listTables():
            spark.catalog.refreshTable(table.name)
            after_stats[table.name] = str(catalog.getTableMetadata(
                parser.parseTableIdentifier(table.name)).stats())
            location = spark.table(table.name)._jdf.queryExecution().analyzed().child().relation().location()
            index_classes[table.name] = location.getClass().getName()
            assert index_classes[table.name].endswith('InMemoryFileIndex'), index_classes
        assert after_stats == before_stats, 'Switching file index changed catalog statistics'
        effective_key = 'spark.sql.hive.manageFilesourcePartitions'
        identity['spark_conf'][effective_key] = spark.conf.get(effective_key)
        identity['effective_sql_overrides'] = {effective_key: spark.conf.get(effective_key)}
        (out/'manifest.json').write_text(json.dumps(identity, indent=2))
        (out/'file-index-setup.json').write_text(json.dumps({
            'seconds': time.perf_counter() - index_setup_start,
            'included_in_query_time': False, 'statistics_unchanged': True,
            'catalog_statistics_before': before_stats, 'catalog_statistics_after': after_stats,
            'index_classes': index_classes}, indent=2))
    (out/'schemas.json').write_text(json.dumps({t.name:spark.table(t.name).schema.jsonValue() for t in spark.catalog.listTables()},indent=2))
    gpu_log=(out/'gpu.csv').open('w')
    gpu=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,power.draw','--format=csv,noheader,nounits','-lms','200'],stdout=gpu_log,stderr=(out/'gpu.stderr').open('w'))
    records=[]
    try:
        for name,sql in queries.items():
            for iteration in range(args.warmup+args.iterations):
                if args.cold:
                    subprocess.run(['sudo','sh','-c','sync; echo 3 > /proc/sys/vm/drop_caches'],check=True)
                job=f'{name}-i{iteration}'
                spark.sparkContext.setJobGroup(job,job,interruptOnCancel=True)
                spark.sparkContext.setLocalProperty('spark.job.description',job)
                timed_out=threading.Event()
                def cancel():
                    timed_out.set(); spark.sparkContext.cancelJobGroup(job)
                timer=threading.Timer(args.timeout,cancel)
                rec={'query':name,'iteration':iteration,'warmup':iteration<args.warmup,'start_epoch':time.time(),'status':'running'}
                print('QUERY_START '+json.dumps(rec),flush=True)
                timer.start()
                try:
                    t=time.perf_counter()
                    df=spark.sql(sql)
                    rows=df.collect()
                    elapsed=time.perf_counter()-t
                    end=time.time()
                    if timed_out.is_set(): raise TimeoutError(job)
                    rec.update(status='ok',elapsed_s=elapsed,end_epoch=end,row_count=len(rows))
                    encoded=sorted(json.dumps(canonical(tuple(row)),sort_keys=True,ensure_ascii=False) for row in rows)
                    payload='\n'.join(encoded)
                    rec['result_sha256']=hashlib.sha256(payload.encode()).hexdigest()
                    (out/'rows'/f'{job}.jsonl').write_text(payload+'\n')
                    (out/'plans'/f'{job}.txt').write_text(spark.sparkContext._jvm.PythonSQLUtils.explainString(df._jdf.queryExecution(),'extended'))
                    (out/'plans'/f'{job}-cost.txt').write_text(spark.sparkContext._jvm.PythonSQLUtils.explainString(df._jdf.queryExecution(),'cost'))
                except Exception:
                    rec.update(status='timeout' if timed_out.is_set() else 'failed',end_epoch=time.time(),error=traceback.format_exc())
                finally:
                    timer.cancel()
                    rec['executor_count']=spark.sparkContext._jsc.sc().getExecutorMemoryStatus().size()-1
                    records.append(rec)
                    with (out/'timings.jsonl').open('a') as f: f.write(json.dumps(rec)+'\n')
                    print('QUERY_END '+json.dumps(rec),flush=True)
                if rec['status']!='ok':
                    if args.continue_on_failure:
                        break
                    raise RuntimeError('Query failed; retained evidence: '+job)
        failed=sorted({r['query'] for r in records if r['status']!='ok'})
        (out/'complete.json').write_text(json.dumps({'status':'complete_with_failures' if failed else 'complete','queries':len(queries),'records':len(records),'failed_queries':failed,'end_epoch':time.time()},indent=2))
    finally:
        gpu.terminate(); gpu.wait(timeout=10); gpu_log.close()
        spark.stop()

if __name__=='__main__': main()
