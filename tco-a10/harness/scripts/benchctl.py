#!/usr/bin/env python3
"""Reproduce the frozen 202.9279 s workload on one host with four GPUs."""
import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import statistics
import subprocess
import sys
import tarfile
import time
import urllib.request

BUNDLE = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((BUNDLE/'manifest.json').read_text())

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def master():
    with urllib.request.urlopen('http://127.0.0.1:8080/json/', timeout=5) as response:
        return json.load(response)

def environment(root):
    env = os.environ.copy()
    env.update(BENCH_ROOT=str(root), SPARK_HOME=str(root/'spark-4.0.2-bin-hadoop3'),
        SPARK_LOCAL_IP='127.0.0.1', SPARK_LOG_DIR=str(root/'daemon-logs'),
        SPARK_PID_DIR=str(root/'pids'), SPARK_IDENT_STRING='tpcds202s-master',
        PYSPARK_PYTHON=sys.executable, PYSPARK_DRIVER_PYTHON=sys.executable,
        SPARK_CONF_DIR=str(root/'spark-4.0.2-bin-hadoop3/conf'), TZ='UTC')
    assert env.get('JAVA_HOME'), 'Set JAVA_HOME to a Java 17 installation.'
    return env

def verify_binary():
    jar = BUNDLE/MANIFEST['artifact']
    assert sha(jar) == MANIFEST['artifact_sha256'], 'Plugin JAR changed'
    assert sha(BUNDLE/'benchmark/queries.sql') == MANIFEST['sql_sha256'], 'SQL changed'
    return jar

def gpu_count(root):
    # Recorded by prepare; the frozen 202.93-second package uses four GPUs.
    return int(json.loads((root/'preflight.json').read_text()).get('gpu_count',4))

def prepare(root, data, gpus=4):
    assert gpus in (2,4), 'Supported GPU counts: 4 (frozen reproduction) or 2 (TCO right-sizing)'
    verify_binary()
    assert data and data.is_dir(), '--data must name the existing SF1000 Parquet directory'
    root.mkdir(parents=True, exist_ok=True)
    assert root.stat().st_dev != Path('/').stat().st_dev, 'Use a mounted data disk for BENCH_ROOT'
    env = environment(root)
    version = subprocess.check_output([str(Path(env['JAVA_HOME'])/'bin/java'), '-version'], stderr=subprocess.STDOUT, text=True)
    assert 'version "17.' in version, version
    spark = Path(env['SPARK_HOME'])
    archive = BUNDLE/'runtime/spark-4.0.2-bin-hadoop3.tgz'
    assert sha(archive) == MANIFEST['spark_archive_sha256']
    if not spark.exists():
        with tarfile.open(archive) as source:
            source.extractall(root)
    assert (spark/'jars/spark-sql_2.13-4.0.2.jar').is_file()
    expected_jars=json.loads((BUNDLE/'runtime/spark-jars-sha256.json').read_text())
    assert {f.relative_to(spark).as_posix() for f in spark.rglob('*.jar')}==set(expected_jars), 'Spark distribution contains different JARs'
    for name,digest in expected_jars.items():
        assert sha(spark/name)==digest, f'Spark runtime differs: {name}'
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.total,driver_version', '--format=csv,noheader,nounits'], text=True)
    assert len(gpu.strip().splitlines()) == gpus, gpu
    for line in gpu.strip().splitlines():
        assert int(line.split(',')[2].strip()) >= 22000, gpu
    for name in ['daemon-logs','pids','spark-local','results','artifacts']:
        (root/name).mkdir(exist_ok=True)
    for src, dst in [(data, root/'data'), (BUNDLE/'benchmark/benchmarks',root/'benchmarks')]:
        if dst.exists():
            assert dst.resolve() == src.resolve(), f'Existing path differs: {dst}'
        else:
            dst.symlink_to(src.resolve(), target_is_directory=True)
    shutil.copyfile(BUNDLE/'benchmark/queries.sql',root/'queries.sql')
    discovery = root/'gpu-discovery.sh'
    discovery.write_text('#!/bin/bash\nexec python3 -c \'import json,subprocess; print(json.dumps({"name":"gpu","addresses":subprocess.check_output(["nvidia-smi","--query-gpu=index","--format=csv,noheader"]).decode().split()}))\'\n')
    discovery.chmod(0o755)
    inventory = sorted((f.relative_to(data).as_posix(),f.stat().st_size) for f in data.rglob('*.parquet'))
    assert inventory, 'No Parquet files found'
    (root/'dataset-files.json').write_text(json.dumps(inventory)+'\n')
    layout = dict(files=len(inventory), bytes=sum(size for _,size in inventory),
        matches_aws_file_count_and_bytes=len(inventory)==12056 and sum(s for _,s in inventory)==337444612977,
        note='Count/size agreement is not a content hash. Keep the same dataset and schema; record all layout differences.')
    (root/'preflight.json').write_text(json.dumps(dict(gpu_count=gpus,gpus=gpu,java=version,dataset=layout,
        plugin_sha256=MANIFEST['artifact_sha256'],bundle=str(BUNDLE)),indent=2)+'\n')
    print(json.dumps(layout,indent=2))

def start(root):
    env = environment(root)
    assert (root/'preflight.json').is_file(), 'Run prepare first'
    for port in [7077,8080,8081,8082,8083,8084]:
        with socket.socket() as sock:
            assert sock.connect_ex(('127.0.0.1',port)) != 0, f'Port {port} is already in use'
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True).strip(), 'GPU processes are already running'
    spark = Path(env['SPARK_HOME'])
    count=gpu_count(root)
    subprocess.run([str(spark/'sbin/start-master.sh'),'--host','127.0.0.1','--port','7077','--webui-port','8080'],env=env,check=True)
    for index in range(count):
        (root/f'worker{index}').mkdir(exist_ok=True)
        (root/f'spark-local/worker{index}').mkdir(exist_ok=True)
        resource = root/f'gpu-{index}.json'
        resource.write_text(json.dumps([dict(id=dict(componentName='spark.worker',resourceName='gpu'),addresses=[str(index)])]))
        worker = dict(env, SPARK_IDENT_STRING=f'tpcds202s-g{index}', SPARK_WORKER_DIR=str(root/f'worker{index}'),
            SPARK_LOCAL_DIRS=str(root/f'spark-local/worker{index}'),
            SPARK_WORKER_OPTS=f'-Dspark.worker.resource.gpu.amount=1 -Dspark.worker.resourcesFile={resource}')
        subprocess.run([str(spark/'sbin/start-worker.sh'),'--host','127.0.0.1','--cores','10','--memory','40g',
            '--webui-port',str(8081+index),'spark://127.0.0.1:7077'],env=worker,check=True)
    deadline=time.time()+90
    while time.time()<deadline:
        status=master()
        if len([w for w in status['workers'] if w['state']=='ALIVE'])==count:
            (root/'cluster-started.json').write_text(json.dumps(status,indent=2)+'\n')
            print(f'Ready: {count} workers, 10 CPU cores and 1 GPU per worker.')
            return
        time.sleep(2)
    raise RuntimeError(f'{count} workers did not register')

def submission_properties(props):
    # OptimizerDriverPlugin appends its SQL extension unconditionally at startup.
    # Keep the template as the expected effective configuration; omit this one
    # extension from the submission so that the runtime contains it exactly once.
    conf=dict(line.split(None,1) for line in props.splitlines())
    optimizer='com.nvidia.spark.rapids.optimizer.SQLOptimizerPlugin'
    conf['spark.sql.extensions']=','.join(
        x for x in conf['spark.sql.extensions'].split(',') if x!=optimizer)
    return ''.join(key+' '+value+'\n' for key,value in conf.items())

def run(root, name, queries):
    assert name and name.replace('-','').replace('_','').isalnum(), 'Use a simple run name'
    assert not master()['activeapps'], 'Another Spark application is active'
    jar=verify_binary()
    out=root/'results'/name
    out.mkdir()
    (out/'events').mkdir()
    env=environment(root)
    props=(BUNDLE/'config/spark-202s.properties.in').read_text()
    for key,value in [('@@ROOT@@',root),('@@RUN@@',out),('@@JAR@@',jar)]:props=props.replace(key,str(value))
    count=gpu_count(root)
    if count!=4:
        # TCO right-sizing: same per-executor resources, fewer executors/GPUs.
        props=props.replace('spark.cores.max 40\n',f'spark.cores.max {10*count}\n').replace('spark.executor.instances 4\n',f'spark.executor.instances {count}\n')
        assert f'spark.cores.max {10*count}\n' in props and f'spark.executor.instances {count}\n' in props
    (out/'submitted.properties').write_text(submission_properties(props))
    shutil.copyfile(root/'preflight.json',out/'preflight.json')
    command=[str(Path(env['SPARK_HOME'])/'bin/spark-submit'),'--master','spark://127.0.0.1:7077',
        '--properties-file',str(out/'submitted.properties'),str(BUNDLE/'scripts/run_benchmark.py'),
        '--dataset',(root/'data').as_uri(),'--queries',queries,'--output',str(out/'query'),
        '--warmup','1','--iterations','3','--timeout','180','--table-statistics','all_columns','--direct-file-index',
        '--expected-executors',str(count)]
    (out/'command.json').write_text(json.dumps(command,indent=2)+'\n')
    with (out/'driver.log').open('x') as log:
        result=subprocess.run(command,env=env,cwd=out,stdout=log,stderr=subprocess.STDOUT)
    (out/'exit-code.txt').write_text(str(result.returncode)+'\n')
    assert result.returncode==0, f'Run failed. Inspect {out}/driver.log'
    print(f'Completed. Validate with: python3 {BUNDLE}/scripts/validate_results.py {out}')

def stop(root):
    assert (root/'cluster-started.json').exists(), 'This utility did not start this cluster'
    assert not master()['activeapps'], 'Wait for the active application before stopping'
    env=environment(root);spark=Path(env['SPARK_HOME'])
    for index in range(gpu_count(root)):
        subprocess.run([str(spark/'sbin/stop-worker.sh')],env=dict(env,SPARK_IDENT_STRING=f'tpcds202s-g{index}'),check=True)
    subprocess.run([str(spark/'sbin/stop-master.sh')],env=env,check=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','start','run','stop'])
    parser.add_argument('--root',type=Path,default=Path('/mnt/bench'))
    parser.add_argument('--data',type=Path)
    parser.add_argument('--name')
    parser.add_argument('--queries',default='all')
    parser.add_argument('--gpus',type=int,choices=[2,4],default=4,help='prepare only: number of GPUs/executors')
    args=parser.parse_args();root=args.root.resolve()
    assert not any(c.isspace() for c in str(root)+str(BUNDLE)), 'Use paths without whitespace'
    if args.action=='prepare':prepare(root,args.data.resolve() if args.data else None,args.gpus)
    elif args.action=='start':start(root)
    elif args.action=='run':run(root,args.name,args.queries)
    else:stop(root)
