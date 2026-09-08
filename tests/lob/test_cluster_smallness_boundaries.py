from datetime import datetime, timedelta

import pytest

from spoofing_detection.lob.execution_clusters import cluster_execution_fills


def fill(index, anchor='passive', timestamp=None, denominator=10.):
    return dict(partition_id='P', sort_index=index, event_class='fill', ORDERID='O',
        client_original_id='C', side_label='ask', event_price=100., fill_qty=6.,
        LASTSHARES=6., LASTTRADEDPX=100., LEAVESQTY=10.,
        event_ts=timestamp or datetime(2024, 1, 1, 12, 0, 0, index * 1000),
        PASSIVEORDER='Y' if anchor == 'passive' else 'N',
        AGGRESSIVEORDER='Y' if anchor == 'aggressive' else 'N',
        metric_row=dict(same_level_market_visible_qty_pre=denominator,
            same_level_actor_visible_qty_pre=denominator,
            smallness_fraction_market_level=.6, smallness_fraction_actor_level=.6))


@pytest.mark.parametrize('anchor', ['passive', 'aggressive'])
@pytest.mark.parametrize('denominator', [10., 0., None, float('nan')])
def test_smallness_uses_all_children_and_keeps_aliases(anchor, denominator):
    clusters, members = cluster_execution_fills([fill(1, anchor, denominator=denominator), fill(2, anchor)], max_gap_ms=100)
    assert len(clusters) == 1 and len(members) == 2
    row = clusters[0]
    for field in ('smallness_fraction_market_level', 'smallness_fraction_actor_level',
                  'passive_smallness_fraction_market_level', 'passive_smallness_fraction_actor_level'):
        if anchor == 'passive' and denominator == 10.:
            assert row[field] == pytest.approx(1.2)
        else:
            assert row[field] is None


@pytest.mark.parametrize('anchor', ['passive', 'aggressive'])
def test_cluster_never_crosses_day_even_with_same_partition(anchor):
    t = datetime(2024, 1, 1, 23, 59, 59, 990000)
    rows, _ = cluster_execution_fills([fill(1, anchor, t), fill(2, anchor, t + timedelta(milliseconds=20))], max_gap_ms=100)
    assert len(rows) == 2
