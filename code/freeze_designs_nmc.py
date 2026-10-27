import os
for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='1'
os.environ['PYBAMM_DISABLE_TELEMETRY']='true'
import json,hashlib,time
from pathlib import Path
import numpy as np
import pandas as pd
from pseudoeis.pipeline import candidate_segment_pool,parameters_from_prior_config
from pseudoeis.models import ECMTruthModel
from pseudoeis.fim import impedance_jacobian,fisher_information
from pseudoeis.signals import spectrum_energy,spectral_coverage_score
ROOT=Path(__file__).resolve().parent
cfg=json.loads((ROOT/'config.json').read_text());ex=cfg['excitation']['proposed'];dt=.01
freq=np.array(ex['target_frequencies_hz']);freq=freq[(freq>0)&(freq<.45/dt)]
seed=20260724
lo=candidate_segment_pool('low_band',ex['low_band'],dt,100,seed)
hi=candidate_segment_pool('high_band',ex['high_band'],dt,100,seed+100000)
models=[ECMTruthModel(parameters_from_prior_config(ch,cfg)).effective_parameters(25,.90,.50) for ch in ['NMC']]
weights=(freq/freq.min())**-.35*10**2/(.001**2+1e-30)
jacs=[impedance_jacobian(freq,m) for m in models]
outer=np.array([np.real(np.einsum('fi,fj->fij',j.conj(),j)) for j in jacs])
rows=[];start=time.time()
for i,a in enumerate(lo):
 for j,b in enumerate(hi):
  q=np.r_[a.q,b.q];energy=spectrum_energy(q,dt,freq)
  f=np.einsum('f,cfij->cij',energy*weights,outer)+1e-12*np.eye(7)[None,:,:]
  sign,logdet=np.linalg.slogdet(f);eig=np.linalg.eigvalsh(f);cond=eig[:,-1]/np.maximum(eig[:,0],1e-12)
  scores=logdet-.02*np.log1p(cond);score=float(scores.min())
  if i*100+j in [0,173,999,2357,5781,9999]:
   expected=min(fisher_information(q,dt,freq,m,.001,10,.35,1e-12,'D').score for m in models)
   assert np.isclose(score,expected,rtol=1e-9,atol=1e-8),(score,expected)
  rows.append(dict(candidate_id=i*100+j,low_index=i,high_index=j,information_score=score,coverage_score=spectral_coverage_score(q,dt,freq)))
 if i%20==0:print('Scored',len(rows),'elapsed',round(time.time()-start,1),flush=True)
d=pd.DataFrame(rows);d.to_csv(ROOT/'candidate_scores.csv',index=False)
info=int(d.information_score.idxmax());coverage=int(d.coverage_score.idxmax());ordered=d.sort_values(['information_score','candidate_id']).candidate_id.to_numpy()
ids=list(dict.fromkeys([info,coverage]+[int(ordered[round(q*(len(d)-1))]) for q in [0,.17,.33,.5,.67,.83]]))
for cid in ordered:
 if len(ids)>=8:break
 if int(cid) not in ids:ids.append(int(cid))
frozen={}
for cid in ids:
 r=rows[cid];a,b=lo[r['low_index']],hi[r['high_index']];q=np.r_[a.q,b.q]
 frozen[str(cid)]=dict(**r,q_sha256=hashlib.sha256(q.astype('<f8').tobytes()).hexdigest(),q=q.tolist(),segments=[dict(name=s.name,q=s.q.tolist(),dt_s=s.dt_s,bit_period_s=s.bit_period_s,lfsr_order=s.lfsr_order,seed_index=s.seed_index) for s in [a,b]],dt_s=dt,target_frequencies_hz=freq.tolist())
(ROOT/'frozen_designs.json').write_text(json.dumps(frozen))
plan=dict(status='frozen_before_evaluation',candidate_pool_seed=seed,candidate_count=10000,information_winner=info,coverage_winner=coverage,panel_candidates=ids,duration_s=round(time.time()-start,2),
 strata=[
  dict(name='cold_NMC',chemistry='NMC',operating_point_id=0,soc=.2,temperature_c=10,soh=1),
  dict(name='nominal_NMC',chemistry='NMC',operating_point_id=1,soc=.5,temperature_c=25,soh=1),
  dict(name='hot_NMC',chemistry='NMC',operating_point_id=2,soc=.8,temperature_c=40,soh=1),
  dict(name='aged_NMC',chemistry='NMC',operating_point_id=3,soc=.5,temperature_c=25,soh=.85),
 ],
 scenarios_per_stratum=10,panel_scenarios_per_stratum=4,scenario_seed_base=3109202600,
 controls='One 265.23 s record per cell; same four-cell truth pack, initial states, scheduling priors, pairing, orientation, current envelope, measurement-noise arrays, SOC bias and optimizer random starts across sequences. Common design name fixes current seed. Jitter zero avoids sequence-dependent edge RNG consumption; old 200 us setting was effectively inactive. No retries.',
 primary='Per-scenario median of all four cell independent-profile voltage RMSEs. Paired information minus coverage, 10 fresh scenarios per stratum, four NMC strata; two-sided exact sign tests with Holm across four strata; pointwise 95% paired bootstrap median intervals, 20000 draws.',
 secondary='Impedance NRMSE and real-impedance absolute relative error at 0.008 Hz: secondary contrasts adjusted together across the four strata. Acceptance counts descriptive.',
 candidate_analysis='Eight candidates chosen using design scores only, before performance. First four fresh packs per stratum. Spearman score-versus-error correlations computed within each pack, summarized across four packs; exploratory, no confirmatory correlation p values.',
 inference_scope='Four prespecified NMC strata (cold/nominal/hot/aged), not a claim about any other chemistry.',
 frozen_designs_sha256=hashlib.sha256((ROOT/'frozen_designs.json').read_bytes()).hexdigest(),candidate_scores_sha256=hashlib.sha256((ROOT/'candidate_scores.csv').read_bytes()).hexdigest())
(ROOT/'protocol.json').write_text(json.dumps(plan,indent=2)+'\n')
print(json.dumps({k:v for k,v in plan.items() if k not in ['controls','primary','secondary','candidate_analysis']},indent=2),flush=True)
