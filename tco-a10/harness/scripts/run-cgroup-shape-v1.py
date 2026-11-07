"""Benchmark runs inside a cgroup v2 group (cpuset + memory cap) to emulate smaller g5 shapes, or on a real host.

Usage: run-g512-v1.py <protocol.json>
protocol: {"name", "queries": "all"|"q,q", "memory_max": "185G",
           "arms": [{"name", "jar": "<local jar name>", "closure_overlay": bool, "conf": {k: v}}]}
Each arm runs 1 warmup + 3 measured per query with the rebaseline harness, DCGM 1 Hz and a
2-second cgroup memory sampler, then archives evidence to <name>-<arm>-evidence.
"""
from pathlib import Path
import datetime, hashlib, json, os, shlex, subprocess, sys, time

p = Path(__file__).resolve().parent
proto_path = Path(sys.argv[1]).resolve()
proto = json.loads(proto_path.read_text())
NAME = proto['name']
c = json.loads((p / proto.get('connection', 'connection.json')).read_text())
ssh = ['ssh', '-i', c['key'], '-o', 'ConnectTimeout=15', c['host']]
MEM = proto.get('memory_max', '185G')
CPUS = proto.get('cpus', '0-23,48-71')
def setup(cpus, mem, workers=5):
    return f'''CG=/sys/fs/cgroup/tcog512; set -e; mkdir -p $CG
echo {cpus} > $CG/cpuset.cpus; echo 0 > $CG/cpuset.mems; echo {mem} > $CG/memory.max; echo 0 > $CG/memory.swap.max
n=0; for p in $(pgrep -u ubuntu -x java); do if grep -qaE "deploy[.](master[.]Master|worker[.]Worker)" /proc/$p/cmdline; then echo $p > $CG/cgroup.procs; n=$((n+1)); fi; done
test $n -eq {workers}; echo 1 > $CG/memory.peak 2>/dev/null || true
sync; echo 3 > /proc/sys/vm/drop_caches
echo cpus=$(cat $CG/cpuset.cpus.effective) memory_max=$(cat $CG/memory.max)'''


TEARDOWN = '''CG=/sys/fs/cgroup/tcog512; D=$1; for f in memory.peak memory.events memory.stat cpu.stat cpuset.cpus.effective memory.max; do cp $CG/$f $D/cgroup-$f.txt; done
cat /sys/fs/cgroup/cpuset.cpus.effective > $CG/cpuset.cpus; echo max > $CG/memory.max; echo max > $CG/memory.swap.max; chown -R ubuntu $D'''
SAMPLER = 'setsid nohup /mnt/bench/tools/memsampler.sh > {dest}/memsample.txt 2>&1 < /dev/null &'
DCGM = ("setsid nohup bash -c 'dcgmi dmon -e 203,1002,1003,1005,1009,1010,155 -d 1000 | while IFS= read -r l; "
        "do printf \"%s %s\\n\" $(date +%s.%N) \"$l\"; done' > {dest}/dcgm.txt 2>&1 < /dev/null &")
TOOLS = ['guarded-operator-metrics-v1.jar', 'exchange-reuse-audit-v3.jar', 'sum-eval-mode-gpu-tools-v3.jar']
state = dict(controller_pid=os.getpid(), protocol=str(proto_path))


def save(phase, **extra):
    state.update(phase=phase, updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra)
    (p / f'{NAME}-state.json').write_text(json.dumps(state, indent=2) + '\n')


def remote(args):
    return subprocess.check_output(ssh + [shlex.join(args)], text=True)


def idle():
    remote(['python3', '-c', "import json,urllib.request,subprocess; assert not json.load(urllib.request.urlopen('http://127.0.0.1:8080/json/'))['activeapps']; assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()"])


def sha(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


try:
    root = f'/mnt/bench/{NAME}'
    remote(['mkdir', '-p', root])
    for arm in proto['arms']:
        a = arm['name']
        idle()
        jar_local = p / arm['jar']
        jar = f"{root}/{arm['jar']}"
        if remote(['bash', '-c', f'test -f {jar} && sha256sum {jar} || true']).split()[:1] != [sha(jar_local)]:
            subprocess.run(['scp', '-q', '-i', c['key'], str(jar_local), c['host'] + ':' + jar], check=True)
        assert remote(['sha256sum', jar]).split()[0] == sha(jar_local)
        dest = f'{root}/{a}'
        remote(['mkdir', dest]); remote(['mkdir', dest + '/events'])
        jars = [jar] + [f'/mnt/bench/tools/{t}' for t in TOOLS]
        props = Path(proto.get('properties_template', p.parent / 'config/spark-202s.properties.in')).read_text() \
            .replace('@@ROOT@@', '/mnt/bench').replace('@@RUN@@', dest).replace('@@JAR@@', jar)
        conf = dict(line.split(None, 1) for line in props.splitlines())
        conf.update({'spark.jars': ','.join(jars), 'spark.driver.extraClassPath': ':'.join(jars),
                     'spark.executor.extraClassPath': ':'.join(jars)})
        conf['spark.sql.extensions'] = ','.join(x for x in conf['spark.sql.extensions'].split(',')
                                                if x != 'com.nvidia.spark.rapids.optimizer.SQLOptimizerPlugin')
        if arm.get('closure_overlay'):
            conf['spark.driver.extraClassPath'] = '/mnt/bench/tools/closure-check-overlay-v1.jar:' + conf['spark.driver.extraClassPath']
        conf.update(arm.get('conf', {}))
        propfile = p / f'{NAME}-{a}.properties'
        propfile.write_text(''.join(k + ' ' + v + '\n' for k, v in conf.items()))
        subprocess.run(['scp', '-q', '-i', c['key'], str(propfile), c['host'] + ':' + dest + '/submitted.properties'], check=True)
        scheduled = subprocess.check_output(ssh + ['sudo shutdown -c && sudo shutdown -h +100 && cat /run/systemd/shutdown/scheduled'], text=True)
        stop = int(next(x.split('=', 1)[1] for x in scheduled.splitlines() if x.startswith('USEC='))) / 1e6
        cmd = ['env', 'JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64', 'SPARK_HOME=/mnt/bench/spark-4.0.2-bin-hadoop3',
               'SPARK_LOCAL_IP=127.0.0.1', 'timeout', '--kill-after=20s', '5400',
               '/mnt/bench/spark-4.0.2-bin-hadoop3/bin/spark-submit', '--master', 'spark://127.0.0.1:7077',
               '--properties-file', dest + '/submitted.properties', '/mnt/bench/tools/' + proto.get('harness', 'run_benchmark_executors_v1.py'),
               '--queries', proto.get('queries', 'all'), '--output', dest + '/query', '--warmup', '1', '--iterations', '3',
               '--timeout', '180', '--table-statistics', 'all_columns', '--direct-file-index', '--expected-executors', str(proto.get('executors', 4))] + proto.get('harness_args', [])
        state[f'{a}_cgroup'] = subprocess.check_output(ssh + ['sudo bash -c ' + shlex.quote(setup(arm.get('cpus', CPUS), arm.get('memory_max', MEM), proto.get('daemons', 5)))], text=True)
        cmd = ['sh', '-c', 'echo $$ | sudo tee /sys/fs/cgroup/tcog512/cgroup.procs >/dev/null && exec "$@"', 'sh'] + cmd
        (p / f'{NAME}-{a}-command.json').write_text(json.dumps(cmd, indent=2) + '\n')
        remote(['bash', '-c', DCGM.format(dest=dest)])
        remote(['bash', '-c', SAMPLER.format(dest=dest)])
        save(a, shutdown_epoch=stop)
        with (p / f'{NAME}-{a}.log').open('x') as log:
            result = subprocess.run(ssh + [shlex.join(cmd)], stdout=log, stderr=subprocess.STDOUT, timeout=5500)
        subprocess.run(ssh + ['pkill -f "dcgmi dmon -e 203"; pkill -f memsampler.sh; true'])
        subprocess.check_output(ssh + ['sudo bash -c ' + shlex.quote(TEARDOWN) + ' teardown ' + shlex.quote(dest)], text=True)
        remote_archive = dest + '.tgz'
        remote(['tar', '-czf', remote_archive, '-C', dest, '.'])
        expected = remote(['sha256sum', remote_archive]).split()[0]
        store = Path(os.environ.get('EVIDENCE_STORE', p / 'evidence')); store.mkdir(parents=True, exist_ok=True)
        archive = store / f'evidence-{NAME}-{a}.tgz'
        subprocess.run(['scp', '-q', '-i', c['key'], c['host'] + ':' + remote_archive, str(archive)], check=True)
        assert sha(archive) == expected
        folder = store / f'{NAME}-{a}-evidence'
        folder.mkdir()
        (p / f'{NAME}-{a}-evidence').symlink_to(folder)
        subprocess.run(['tar', '-xzf', str(archive), '-C', str(folder)], check=True)
        save(a + '_archived', **{f'{a}_archive_sha256': expected, f'{a}_returncode': result.returncode})
        assert result.returncode == 0, a
    save('complete_requires_review')
except Exception as e:
    save('failed_requires_inspection', failed_phase=state.get('phase'), error=repr(e))
    try:
        subprocess.run(ssh + ['pkill -f "dcgmi dmon -e 203"; pkill -f memsampler.sh; sudo bash -c ' +
                              shlex.quote('test -e /sys/fs/cgroup/tcog512 && { ' + TEARDOWN + '; }') + ' teardown /tmp'], timeout=120)
    except Exception:
        pass
    raise
