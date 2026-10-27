import sys, json, re
def load(p):
    T = json.load(open(p)); S = T['summary']; m0, m1 = T['machine_start'], T['machine_end']
    assert S['n'] == 96 and S['samples'] == [26523] and S['max_cge_rmse_diff_mv'] < 1e-9 and S['max_conv_rmse_diff_mv'] < 1e-2
    assert m0['cpu'] == m1['cpu'] and len(m0['affinity']) == 1 and all(v == '1' for v in m0['threads'].values())
    return S, m0
def cpu_name(s):
    m = re.match(r'AMD EPYC (\w+)', s)
    if m: return 'AMD EPYC ' + m.group(1)
    g = re.search(r'@\s*([0-9.]+)\s*GHz', s)
    return ('Intel Xeon' if 'Xeon' in s else s) + (' at %.1f~GHz' % float(g.group(1)) if g else '')
S, m = load(sys.argv[1]); Sc, mc = load(sys.argv[2])
assert m['slurm']['SLURM_JOB_CPUS_PER_NODE'] == m['slurm']['SLURM_CPUS_ON_NODE']
out = {'tm.cge': '%.3f' % S['cge_median_sec'], 'tm.conv': '%.0f' % S['conv_median_sec'], 'tm.conv.min': '%.0f' % S['conv_min_sec'],
       'tm.conv.max': '%.0f' % S['conv_max_sec'], 'tm.n': str(S['n']), 'tm.ratio': '%.0f' % (S['conv_median_sec'] / S['cge_median_sec']),
       'tm.cpu': cpu_name(m['cpu']), 'tm.cores': str(m['cores_visible']), 'tm.mem': '%.0f' % m['mem_gib'], 'tm.rss': '%.2f' % S['peak_rss_gib'],
       'tmc.cge': '%.2f' % Sc['cge_median_sec'], 'tmc.conv': '%.0f' % Sc['conv_median_sec'], 'tmc.cpu': cpu_name(mc['cpu'])}
with open('numbers_timing.tex', 'w') as f:
    for k, v in out.items(): f.write('\\expandafter\\def\\csname num@%s\\endcsname{%s}\n' % (k, v))
print(out)
