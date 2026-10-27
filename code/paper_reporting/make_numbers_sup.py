import json, sys
d = json.load(open('sup_contrast.json')); out = sys.argv[1] if len(sys.argv) > 1 else 'numbers_sup.tex'
C = ('cold', 'nominal', 'hot', 'aged'); L = []
def sgn(x, p): return ('\\ensuremath{-}' if x < 0 else '\\ensuremath{+}') + f'{abs(x):.{p}f}'
def put(k, v): L.append('\\expandafter\\def\\csname num@%s\\endcsname{%s}' % (k, v))
def h(x): return f'{x:.3f}' if x < 0.1 else f'{x:.2f}'
def rng(xs, p=2): return f'{sgn(min(xs), p)} to {sgn(max(xs), p)}'
s = d['soc1']; cs = s['contrasts']
for c in C:
    for k, name in (('full', 'cge'), ('v2', 'v2'), ('nooffset', 'noff')): put(f'sup.soc1.{name}.{c}', f"{s['unit_medians'][c][k]:.1f}")
    put(f'sup.soc1.diff.{c}', sgn(cs['cge_vs_v2'][c]['median'], 2)); put(f'sup.soc1.nlow.{c}', cs['cge_vs_v2'][c]['n_lower'])
put('sup.soc1.holm.min', h(min(cs['cge_vs_v2'][c]['p_holm'] for c in C)))
put('sup.soc1.lin.cold.diff', f"{abs(cs['cge_vs_linear']['cold']['median']):.2f}"); put('sup.soc1.lin.cold.nlow', cs['cge_vs_linear']['cold']['n_lower']); put('sup.soc1.lin.cold.holm', h(cs['cge_vs_linear']['cold']['p_holm']))
put('sup.soc1.lin.other.holm.min', h(min(cs['cge_vs_linear'][c]['p_holm'] for c in C if c != 'cold')))
put('sup.soc1.n', s['n_units'])
for arm in ('C1', 'C2'):
    r = d[f'raw_{arm}']; k = r['contrasts']
    put(f'sup.raw.{arm}.diffrange', rng([k['raw_vs_v2'][c]['median'] for c in C])); put(f'sup.raw.{arm}.holm.min', h(min(k['raw_vs_v2'][c]['p_holm'] for c in C)))
    for c in C: put(f'sup.raw.cge.{arm}.{c}.nlow', k['raw_vs_cge'][c]['n_higher']); put(f'sup.raw.cge.{arm}.{c}.holm', h(k['raw_vs_cge'][c]['p_holm']))
    put(f'sup.raw.cge.{arm}.nlowmin', min(k['raw_vs_cge'][c]['n_higher'] for c in C)); put(f'sup.raw.cge.{arm}.holm.max', h(max(k['raw_vs_cge'][c]['p_holm'] for c in C)))
    put(f'sup.raw.lin.{arm}.nlowmin', min(k['raw_vs_linear'][c]['n_higher'] for c in C)); put(f'sup.raw.lin.{arm}.holm.max', h(max(k['raw_vs_linear'][c]['p_holm'] for c in C)))
    put(f'sup.raw.{arm}.n', r['n_units']); put(f'sup.raw.{arm}.fail', r['fail'])
put('sup.raw.holm.min', h(min(d[f'raw_{a}']['contrasts']['raw_vs_v2'][c]['p_holm'] for a in ('C1','C2') for c in C)))
put('sup.raw.maxabs', '%.2f' % (int(max(abs(d[f'raw_{a}']['contrasts']['raw_vs_v2'][c]['median']) for a in ('C1','C2') for c in C)*100+0.999999)/100))
put('sup.raw.C1.v2range', '%.2f--%.2f' % (min(d['raw_C1']['unit_medians'][c]['v2'] for c in C), max(d['raw_C1']['unit_medians'][c]['v2'] for c in C)))
put('sup.raw.C1.rawrange', '%.2f--%.2f' % (min(d['raw_C1']['unit_medians'][c]['raw'] for c in C), max(d['raw_C1']['unit_medians'][c]['raw'] for c in C)))
open(out, 'w').write('\n'.join(L) + '\n'); print(len(L), 'macros ->', out)
