import torch

from akh import contract, hexagon_timer as ht, reference, stats


def test_reference_matches_torch_rms_norm():
    for case in contract.perf_cases() + contract.validation_cases():
        x, w = contract.make_inputs(case)
        ref = reference.rmsnorm_ref(x, w)
        tt = torch.nn.functional.rms_norm(x.float(), (x.shape[1],), weight=w.float(), eps=contract.EPS).half()
        assert contract.check(case, ref, tt).ok, case.name
        assert torch.isfinite(ref.float()).all(), case.name


def test_check_reports_max_error_and_location():
    case = contract.perf_cases()[0]
    x, w = contract.make_inputs(case)
    ref = reference.rmsnorm_ref(x, w)
    bad = ref.clone()
    bad[0, 7] += 0.5
    r = contract.check(case, bad, ref)
    assert not r.ok and r.n_bad == 1 and abs(r.max_abs_err - 0.5) < 0.01 and r.worst_index == (0, 7)


def test_fp16_accumulation_fails_large_case():
    case = next(c for c in contract.validation_cases() if c.kind == "large")
    x, w = contract.make_inputs(case)
    ms16 = (x * x).float().mean(-1, keepdim=True)  # x*x overflows in fp16 -> inf
    y = (x.float() * torch.rsqrt(ms16 + contract.EPS) * w.float()).half()
    assert not contract.check(case, y, reference.rmsnorm_ref(x, w)).ok


STOCK = """Test_Info: {
\tName:rmsnorm_row_kernel
\tResult:Pass
\tPerf:12.000000
\tUnits:us
}
"""


def test_parse_stock_and_ak_perf():
    r = ht.parse_perf_text(STOCK)
    assert (r.name, r.result, r.perf, r.timer) == ("rmsnorm_row_kernel", "Pass", 12.0, "stock")
    ak = STOCK.replace("12.000000", "11.430000") + (
        "AK_Timer:ak-timer-v1\nAK_Iterations:100\nAK_FirstCall_us:85\nAK_FirstCall_pcycles:100\n"
        "AK_Steady_total_us:1143\nAK_Steady_avg_pcycles:16002.5\n")
    r = ht.parse_perf_text(ak)
    assert r.timer == "ak-timer-v1" and r.steady_avg_us == 11.43
    assert ht.ak_float(r, "FirstCall_us") == 85 and ht.ak_float(r, "Steady_avg_pcycles") == 16002.5
    assert ht.parse_perf_text("garbage").timer == "missing"


def test_ak_template_renders_balanced_cpp():
    s = ht.render_ak_block("kern(1, dt0);", 50, "kern", "/data/local/tmp/x/perf.txt")
    assert s.count("kern(1, dt0);") == 2 and s.count("{") == s.count("}")
    assert "ak_i < 50" in s and '"/data/local/tmp/x/perf.txt"' in s and "AK_Timer:ak-timer-v1\\n" in s


def test_parse_executor_log():
    log = """==> Running the kernel on device ...
adb -H h -s S push /tmp/a.so /data/local/tmp/k/lib
Command took 0.2000 seconds.
adb -H h -s S shell 'cd /data/local/tmp/k; ...; ./run_main_on_hexagon 3 /data/local/tmp/k/lib.so'
Command took 0.9000 seconds.
adb -H h -s S pull /data/local/tmp/k/o0.raw /tmp/k
Command took 0.1000 seconds.
==> Ran successfully, collecting results
"""
    e = ht.parse_executor_log(log)
    assert e["run_s"] == 0.9 and e["push_s"] == 0.2 and e["pull_s"] == 0.1 and e["ran_on_device"]
    assert not ht.parse_executor_log("==> Running the kernel on device ...\nError executing the command")["ran_on_device"]


def test_decide():
    base = [10.0, 10.1, 9.9, 10.05, 9.95, 10.0]
    assert stats.decide(base, [8.0] * 6).verdict == "improved"
    assert stats.decide(base, [10.3] * 6).verdict == "keep_previous"
    # 0.5% gain, baseline spread ~2% -> more rounds, then keep previous
    assert stats.decide(base, [9.95] * 6, rounds_done=1, max_rounds=3).verdict == "need_more"
    assert stats.decide(base, [9.95] * 6, rounds_done=3, max_rounds=3).verdict == "keep_previous"
    # noisy baseline: 5% gain inside a 20% spread is not enough
    noisy = [9, 11, 10, 12, 8, 10]
    assert stats.decide(noisy, [9.5] * 6, rounds_done=3).verdict == "keep_previous"
    # DSP better but host latency reproducibly 30% worse -> hold
    d = stats.decide(base, [8.0] * 6, [100.0] * 6, [130.0] * 6)
    assert d.verdict == "hold_host_regression"
    assert stats.decide(base, [None, None]).verdict == "no_timing"


def test_systematic_scale_error_is_caught():
    # regression for exp hos0016: mean over padded block (4096) instead of H=4000
    case = next(c for c in contract.validation_cases() if c.name == "val_tail_3x4000")
    x, w = contract.make_inputs(case)
    xf = x.float()
    y = (xf * torch.rsqrt(xf.pow(2).sum(-1, keepdim=True) / 4096 + contract.EPS) * w.float()).half()
    r = contract.check(case, y, reference.rmsnorm_ref(x, w))
    assert not r.ok and "rel_rms_err" in r.note
