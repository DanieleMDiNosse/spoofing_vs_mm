"""Presentation-only contracts; scientific rank/price inputs remain frozen."""
from pathlib import Path
import importlib.util
import re
import polars as pl
import pytest

ROOT=Path(__file__).resolve().parents[2]
REVIEW=ROOT/'paper/reviews/20261015_152614_participant_results'


def module():
    path=ROOT/'scripts/generate_participant_manuscript.py'
    spec=importlib.util.spec_from_file_location('participant_manuscript',path)
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_single_axis_three_exact_horizon_curves_and_legend():
    m=module()
    counts=pl.read_csv(ROOT/'paper/generated/withdrawal_rank_20261015_v2/cell_counts.csv')
    fig,data=m.candidate_count_figure(counts)
    try:
        assert len(fig.axes)==1
        ax=fig.axes[0]
        assert len(ax.lines)==3
        assert [text.get_text() for text in ax.get_legend().get_texts()]==['Ferrari','Nexi','Risanamento']
        assert data.height==12
        for line,instrument in zip(ax.lines,['FERRARI','NEXI','RISANAMENTO']):
            expected=counts.filter((pl.col('instrument')==instrument)&(pl.col('age_policy')=='exp_h30')).sort('horizon_ns')
            assert list(line.get_xdata())==[1.,2.,5.,10.]
            assert list(line.get_ydata())==expected['candidate_executions'].to_list()
        assert ax.get_ylim()[0]==0
        fig.canvas.draw()
        renderer=fig.canvas.get_renderer()
        assert ax.get_legend().get_window_extent(renderer).y0>=ax.get_window_extent(renderer).y1
        for artist in [ax.xaxis.label,ax.yaxis.label,ax.get_legend(),*ax.get_xticklabels(),*ax.get_yticklabels()]:
            if artist.get_visible() and (not hasattr(artist,'get_text') or artist.get_text()):
                box=artist.get_window_extent(renderer)
                assert box.x0>=0 and box.y0>=0
                assert box.x1<=fig.bbox.width and box.y1<=fig.bbox.height
    finally:
        m.plt.close(fig)


def test_chart_rejects_missing_or_duplicate_setting():
    m=module();counts=pl.read_csv(ROOT/'paper/generated/withdrawal_rank_20261015_v2/cell_counts.csv')
    keep=counts.filter(pl.col('age_policy')=='exp_h30')
    with pytest.raises(ValueError,match='twelve'):
        m.candidate_count_figure(keep.head(11))
    with pytest.raises(ValueError,match='twelve'):
        m.candidate_count_figure(pl.concat([keep,keep.head(1)]))




def test_existing_output_refused_before_any_inputs(tmp_path,monkeypatch):
    m=module()
    def unexpected(*args,**kwargs):
        raise AssertionError('input read before refusal')
    monkeypatch.setattr(m.base,'build',unexpected)
    with pytest.raises(FileExistsError):
        m.build(tmp_path)


def test_instrument_groups_keep_labels_and_cells_together(tmp_path):
    m=module()
    package=ROOT/'paper/generated/withdrawal_rank_20261015_participants_v2'
    if not package.exists():
        pytest.skip('real participant package absent')
    tables={name:pl.read_csv(package/f'participant_{name}.csv',infer_schema_length=None)
            for name in ['summary','entities']}
    m.render_participants(tmp_path,tables)
    for name,n in [('client_groups.tex',33),('intermediary_groups.tex',50)]:
        actual=(tmp_path/name).read_text()
        assert actual.replace(r'\\*',r'\\')==(package/name).read_text()
        assert len(re.findall(r'^[CF]\d+ &',actual,re.M))==n
        # Each heading plus all but the last row of its instrument cannot break.
        assert actual.count(r'\\*')==n
        assert 'resizebox' not in actual and 'scriptsize' not in actual
