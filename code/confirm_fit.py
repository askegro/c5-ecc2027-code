import sys, json, time
from pathlib import Path
import numpy as np
import evallib as ev, designs

TD = Path(sys.argv[1] if len(sys.argv) > 1 else 'traces_confirm'); OUT = Path(sys.argv[2] if len(sys.argv) > 2 else 'results_confirm')
OUT.mkdir(exist_ok=True)
LOOP = '--loop' in sys.argv
REVERSE = '--reverse' in sys.argv
ONLY = sys.argv[sys.argv.index('--tag') + 1] if '--tag' in sys.argv else None
SEQS = ('s4786', 's9588', 'short74', 'passive'); FROZEN = ('s4786',)
D = designs.all_designs()
N_EXPECTED = 96


def process(tag):
    cell = ev.load_cell(TD, tag); meta = cell['meta']
    res = dict(meta=meta, zref=json.loads((TD / f'{tag}.zref.json').read_text()), seq={})
    for name in SEQS:
        seq = ev.load_seq(TD, tag, name, cell); r = {}
        for arm in ('C1', 'C2'):
            t0 = time.time(); a = ev.eval_new(cell, seq, arm); a['sec'] = time.time() - t0
            b = ev.eval_new(cell, seq, arm)
            a['repeat_identical'] = bool(a['rmse_mv'] == b['rmse_mv'])
            r[arm] = dict(v3=a)
            if name in FROZEN:
                r[arm]['v2'] = ev.eval_frozen(cell, seq, arm, D[name], meta['seed'] + 50_000 + meta['cell'])
        res['seq'][name] = r
    (OUT / f'{tag}.json').write_text(json.dumps(res, indent=1))
    print(tag, {n: round(res['seq'][n]['C1']['v3']['rmse_mv'], 3) for n in SEQS}, 'v2:', {n: round(res['seq'][n]['C1']['v2']['rmse_mv'], 3) for n in FROZEN}, flush=True)


if ONLY:
    if not (OUT / f'{ONLY}.json').exists():
        process(ONLY)
    sys.exit(0)

while True:
    tags = sorted(f.name.split('.')[0] for f in TD.glob('*.zref.json'))
    todo = [t for t in tags if not (OUT / f'{t}.json').exists() and all((TD / f'{t}.{s}.npz').exists() for s in SEQS + ('val',))]
    for t in (todo[::-1] if REVERSE else todo):
        if not (OUT / f'{t}.json').exists():
            process(t)
    if not LOOP or len(list(OUT.glob('*.json'))) >= N_EXPECTED:
        break
    if not todo:
        time.sleep(30)
print('FIT_DONE', len(list(OUT.glob('*.json'))))
