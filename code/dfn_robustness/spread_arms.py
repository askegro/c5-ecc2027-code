import math
R_REF = 2.751e-3
LIT_CAP, LIT_IMP = 0.0028, 0.0072
RC_MEAN = 1.0e-4
RC_SD = math.sqrt((LIT_IMP * (R_REF + RC_MEAN))**2 - (LIT_CAP * R_REF)**2)
ARMS = dict(
    orig=dict(cap_batch=0.02, cap_unit=0.02, res_batch=0.02, res_unit=0.03, contact='legacy', rc_mean_ohm=0.0, rc_sd_ohm=0.0, soh_sd=0.012),
    lit=dict(cap_batch=0.0, cap_unit=LIT_CAP, res_batch=0.0, res_unit=0.0, contact='series', rc_mean_ohm=RC_MEAN, rc_sd_ohm=RC_SD, soh_sd=0.0),
    dl=dict(cap_batch=0.02, cap_unit=0.02, res_batch=0.02, res_unit=0.03, contact='legacy', rc_mean_ohm=0.0, rc_sd_ohm=0.0, soh_sd=0.012, surface='differential'),
    zero=dict(cap_batch=0.0, cap_unit=0.0, res_batch=0.0, res_unit=0.0, contact='legacy', rc_mean_ohm=0.0, rc_sd_ohm=0.0, soh_sd=0.0),
)
def arm(name):
    if name not in ARMS: raise KeyError(f'unknown N6_SPREAD arm {name!r}; choose one of {sorted(ARMS)}')
    return dict(ARMS[name], name=name)
