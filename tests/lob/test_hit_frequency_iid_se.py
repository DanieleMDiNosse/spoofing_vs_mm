from pathlib import Path
import importlib.util
import math
import re
import numpy as np
import polars as pl
import pytest

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('hit_se', REPO/'scripts/generate_hit_frequency_iid_se.py')
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('sample', [[0,1], [0,0,1,1,1], [0]*15+[1]*3, [0]*12, [1]*10])
def test_matches_sample_sd_over_sqrt_n(sample):
    expected = np.std(sample, ddof=1)/np.sqrt(len(sample))
    assert module.iid_hit_se(sum(sample), len(sample)) == pytest.approx(expected)


@pytest.mark.parametrize('h,n', [(0,0),(0,1),(1,1)])
def test_insufficient_support_is_missing(h,n):
    assert module.iid_hit_se(h,n) is None


@pytest.mark.parametrize('h,n', [(-1,10),(11,10),(0,-1),(1.5,10)])
def test_invalid_counts_are_rejected(h,n):
    with pytest.raises(ValueError): module.iid_hit_se(h,n)


def test_empirical_render_preserves_centers_and_other_figures(tmp_path):
    out = tmp_path/'render'
    module.generate(out)
    frame = pl.read_csv(out/'hit_frequency_standard_errors.csv')
    assert frame.height==60
    assert frame.select('instrument_id','side','rank').unique().height==60
    assert frame['standard_error'].is_finite().all()
    np.testing.assert_allclose(frame['sample_sd'].to_numpy()/np.sqrt(frame['exposure_count'].to_numpy()),frame['standard_error'].to_numpy(),rtol=1e-14)
    old=module.SOURCE.read_text();new=(out/'kernel_figures.tex').read_text()
    assert old.split('% data column:')[0]==new.split('% data column:')[0]
    assert re.findall(r'plot coordinates \{([^}]*)\}',old)==re.findall(r'plot coordinates \{([^}]*)\}',new)
    assert new.count('line width=0.5pt')==180
    assert 'overlapping windows or serial dependence' in new
    assert new.count(r'\label{fig:empirical_topn_execution_risk_profiles}')==1
    with pytest.raises(FileExistsError): module.generate(out)
