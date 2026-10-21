import numpy as np
import polars as pl
import pytest
from scripts.wmsci_j_manuscript_additions import ecdf, identity_counts


def test_ecdf_ties_zero_and_endpoint():
    x, y = ecdf([2., 0., 2., 1.])
    np.testing.assert_array_equal(x, [0., 1., 2.])
    np.testing.assert_allclose(y, [.25, .5, 1.])
    assert np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0)


@pytest.mark.parametrize('values', [[], [np.nan], [np.inf]])
def test_ecdf_rejects_missing_or_empty(values):
    with pytest.raises(ValueError):
        ecdf(values)


def test_identity_counts_preserve_literal_ids_and_cross_instrument_recurrence():
    selected = pl.DataFrame({'instrument': ['X','X','Y','Y'],
        'actor_key': ['firm:001_2','firm:001_2','firm:001_2','client_original:0003'],
        'actor_id': ['001_2','001_2','001_2','0003'],
        'identity_level': ['firm','firm','firm','client_original'],
        'execution_anchor_mode': ['passive','aggressive','passive','passive']})
    aliases = selected.select('instrument','actor_key').with_columns(
        pl.Series('actor_alias', ['A1','A1','A1','A2']))
    counts = identity_counts(selected, aliases)
    assert counts.height == 3
    assert counts['executions'].sum() == 4
    assert set(counts['actor_id']) == {'001_2','0003'}
    row = counts.filter((pl.col('instrument')=='X')).row(0,named=True)
    assert row['executions']==2 and row['passive']==1 and row['aggressive']==1


def test_render_conditions_and_exports(tmp_path):
    from scripts.wmsci_j_manuscript_additions import render_additions
    from scripts.generate_wmsci_j_top10_manuscript import table
    population = pl.DataFrame({'instrument':['X']*5,
        'WMSCI_event':[0., 1., 2., float('nan'), float('inf')]})
    joined = pl.DataFrame({'instrument':['X']*4, 'complete_grid':[True,True,False,True],
                          'J':[0., 2., 999., float('inf')]})
    selected = pl.DataFrame({'instrument':['X'], 'actor_key':['firm:001_2'],
        'actor_id':['001_2'], 'identity_level':['firm'], 'execution_anchor_mode':['passive']})
    aliases = pl.DataFrame({'instrument':['X'], 'actor_key':['firm:001_2'], 'actor_alias':['A1']})
    result = render_additions(population, joined, selected, aliases, tmp_path, table)
    assert result['population_counts']=={'X':{'positive_W':2,'conditional_J':2}}
    cdf = pl.read_csv(tmp_path/'wmsci_j_ecdf.csv')
    assert cdf.filter(pl.col('metric')=='J')['value'].to_list()==[0.,2.]
    assert (tmp_path/'wmsci_j_ecdf.pdf').stat().st_size > 1000
    assert r'001\_2' in (tmp_path/'identified_participants.tex').read_text()
