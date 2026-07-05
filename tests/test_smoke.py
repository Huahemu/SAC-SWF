from sac_swf.sac_swf import SACSWFConfig, sac_swf
from sac_swf.synthetic import SyntheticConfig, generate_sample
from sac_swf.metrics import evaluate_structure


def test_sac_swf_smoke():
    sample = generate_sample(SyntheticConfig(size=64, seed=123))
    image = sample["I"]
    target = sample["S"]
    out, inter = sac_swf(image, config=SACSWFConfig(radius=3), return_intermediates=True)
    assert out.shape == image.shape
    assert "beta" in inter
    metrics = evaluate_structure(out, target, inp=image)
    assert "psnr" in metrics
