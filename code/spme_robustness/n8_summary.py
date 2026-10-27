import sys, json, numpy as np
from pathlib import Path
S = {p.stem: json.loads(p.read_text()) for p in sorted((Path(sys.argv[1]) / 'split').glob('*.json'))}
print(f'{len(S)} units')
for c in ('cold', 'nominal', 'hot', 'aged'):
    u = [s for s in S.values() if s['cond'] == c]
    if not u: continue
    print(c, ' '.join(f"{e} all {np.median([s[e]['all'] for s in u]):.2f} drive {np.median([s[e]['drive'] for s in u]):.2f} rest {np.median([s[e]['rest'] for s in u]):.2f} |" for e in ('cge', 'lin')),
          f"SOC change {np.median([s['soc_change_pct'] for s in u]):.2f} %")
