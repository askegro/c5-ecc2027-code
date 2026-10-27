import sys, json
A, B = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])); S = A['summary']; CONDS = ('cold', 'nominal', 'hot', 'aged')
assert len(A['units']) == 96 and len(B['units']) == 96
M = {'nz.med': '%.3f' % max(abs(S['cge.B'][c] - S['cge.A'][c]) for c in CONDS), 'nz.unit': '%.2f' % S['cge.maxunit.B-A'],
     'nz.resample': '%.3f' % max(max(v) for v in B['spread'].values())}
open('numbers_noise.tex', 'w').write(''.join(f'\\expandafter\\def\\csname num@{k}\\endcsname{{{v}}}\n' for k, v in M.items())); print(M)
