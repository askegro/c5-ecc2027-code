import numpy as np
import cvx_estimator as ce

rng = np.random.default_rng(0)
N, DT = 26523, 0.01


def switched_current(seed=1):
    r = np.random.default_rng(seed)
    q = np.repeat(r.choice([-1.0, 1.0], N // 50 + 1), 50)[:N]
    ip = 20.0 * (1 + 0.2 * np.sin(2 * np.pi * np.arange(N) * DT / 53.0) + 0.1 * np.sin(2 * np.pi * np.arange(N) * DT / 7.3))
    return 0.5 * (1 + q) * ip


def in_class_model():
    taus, scales = ce.tau_grid(), ce.bv_scales(60.0)
    r = np.zeros(len(taus)); r[[10, 25, 33, 40]] = [4e-4, 3e-4, 9e-4, 6e-4]
    rho = np.zeros(len(scales)); rho[2] = 1.5e-3
    return ce.LocalModel(taus=taus, scales=scales, b=0.0, r0=6e-4, rho=rho, r=r, offset=False, n_samples=0, train_rmse_mv=0.0, kkt_violation=0.0)


def test_forward_consistency_and_exact_recovery():
    i = switched_current(); m0 = in_class_model(); y = m0.predict_drop(i, DT)
    m = ce.fit(i, y, DT, offset=False)
    assert m.certified and m.train_rmse_mv < 2e-3, (m.kkt_violation, m.train_rmse_mv)
    v = switched_current(7)[:3600]
    assert np.max(np.abs(m.predict_drop(v, 0.05) - m0.predict_drop(v, 0.05))) < 2e-5
    return m.train_rmse_mv


def test_deterministic():
    i = switched_current(); y = in_class_model().predict_drop(i, DT) + rng.normal(0, 5e-4, N)
    a = ce.fit(i, y, DT, offset=True); b = ce.fit(i, y, DT, offset=True)
    assert np.array_equal(a.r, b.r) and a.b == b.b and a.r0 == b.r0 and np.array_equal(a.rho, b.rho)


def test_noise_and_offset():
    i = switched_current(); m0 = in_class_model(); v = switched_current(7)[:3600]
    errs = []
    for k in range(5):
        y = m0.predict_drop(i, DT) + 0.0123 + np.random.default_rng(100 + k).normal(0, 5e-4, N)
        m = ce.fit(i, y, DT, offset=True)
        assert m.certified and abs(m.b - 0.0123) < 2e-4, m.b
        assert np.all(m.r >= 0) and np.all(m.rho >= 0) and m.r0 >= 0
        errs.append(1e3 * np.sqrt(np.mean((m.predict_drop(v, 0.05) - 0.0123 - m0.predict_drop(v, 0.05)) ** 2)))
    assert max(errs) < 0.1, errs
    return max(errs)


def test_scale_invariance():
    i = switched_current(); y = in_class_model().predict_drop(i, DT) + rng.normal(0, 5e-4, N)
    a = ce.fit(i, y, DT, offset=True); b = ce.fit(i, 1e3 * y, DT, offset=True)
    v = switched_current(7)[:3600]
    assert np.max(np.abs(1e3 * a.predict_drop(v, .05) - b.predict_drop(v, .05))) < 1e-6


def test_degenerate_inputs():
    for cur in (np.full(N, 20.0), np.zeros(N), switched_current()[:500]):
        y = 2e-3 * cur + rng.normal(0, 5e-4, len(cur))
        for off in (True, False):
            m = ce.fit(cur, y, DT, offset=off)
            assert m.certified and np.all(np.isfinite(m.predict_drop(cur, DT)))
            assert abs(m.b) < 1.0 and m.r_inst_small_signal < 1.0 and m.r.sum() < 1.0, (m.b, m.r_inst_small_signal)
    for bad in (np.array([np.nan] * N), ):
        try:
            ce.fit(bad, bad, DT, offset=True); raise AssertionError("NaN input accepted")
        except ValueError:
            pass


if __name__ == "__main__":
    print("exact recovery, train rmse (mV):", test_forward_consistency_and_exact_recovery())
    test_deterministic(); print("deterministic: ok")
    print("noise + offset, worst prediction error (mV):", test_noise_and_offset())
    test_scale_invariance(); print("scale invariance: ok")
    test_degenerate_inputs(); print("degenerate inputs: ok")
    print("ALL TESTS PASSED")
