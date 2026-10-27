import sys, os, json, time, platform, resource
from pathlib import Path
import numpy as np, scipy
import evallib as ev, designs
TD, E2D = sys.argv[1], sys.argv[2]; D = designs.all_designs()
def machine():
    cpu = next((l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')), platform.processor())
    mem = next((int(l.split()[1]) for l in open('/proc/meminfo') if l.startswith('MemTotal')), 0) / 1024**2
    return dict(cpu=cpu, mem_gib=round(mem, 2), cores_visible=os.cpu_count(), affinity=sorted(os.sched_getaffinity(0)), python=platform.python_version(),
                numpy=np.__version__, scipy=scipy.__version__, load_avg=os.getloadavg(), host=platform.node(),
                slurm={k: os.environ.get(k) for k in ('SLURM_JOB_ID', 'SLURMD_NODENAME', 'SLURM_JOB_PARTITION', 'SLURM_JOB_CPUS_PER_NODE', 'SLURM_CPUS_ON_NODE')}, threads={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})
tags = sorted(p.name[:-len('.s4786.npz')] for p in Path(TD).glob('*.s4786.npz')); assert len(tags) == 96
out = dict(machine_start=machine(), units={})
for t in tags:
    cell = ev.load_cell(TD, t); seq = ev.load_seq(TD, t, 's4786', cell); meta = cell['meta']; st = json.loads((Path(E2D) / f'{t}.json').read_text())['seq']['s4786']['C1']
    t0 = time.perf_counter(); a = ev.eval_new(cell, seq, 'C1'); t1 = time.perf_counter()
    b = ev.eval_frozen(cell, seq, 'C1', D['s4786'], meta['seed'] + 50_000 + meta['cell']); t2 = time.perf_counter()
    out['units'][t] = dict(n=len(seq['i']), cge_sec=t1 - t0, conv_sec=t2 - t1, cge_rmse=a['rmse_mv'], conv_rmse=b['rmse_mv'],
                           cge_d=abs(a['rmse_mv'] - st['v3']['rmse_mv']), conv_d=abs(b['rmse_mv'] - st['v2']['rmse_mv']))
    print(t, '%.3f s  %.1f s' % (t1 - t0, t2 - t1), flush=True)
    json.dump(out, open('timing_fits_idle.json', 'w'), indent=1)
U = out['units'].values()
out['summary'] = dict(n=len(out['units']), samples=sorted({u['n'] for u in U}), cge_median_sec=float(np.median([u['cge_sec'] for u in U])), cge_max_sec=max(u['cge_sec'] for u in U),
                      conv_median_sec=float(np.median([u['conv_sec'] for u in U])), conv_min_sec=min(u['conv_sec'] for u in U), conv_max_sec=max(u['conv_sec'] for u in U),
                      max_cge_rmse_diff_mv=max(u['cge_d'] for u in U), max_conv_rmse_diff_mv=max(u['conv_d'] for u in U),
                      peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2)
out['machine_end'] = machine(); json.dump(out, open('timing_fits_idle.json', 'w'), indent=1); print(json.dumps(out['summary'], indent=1))
