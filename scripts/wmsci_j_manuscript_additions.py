"""Descriptive identity table and paired ECDFs from retained observations."""
import numpy as np
import polars as pl


def ecdf(values):
    """Right-continuous empirical CDF, aggregating ties without smoothing."""
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError('ECDF requires nonempty finite observations')
    x, counts = np.unique(values, return_counts=True)
    return x, np.cumsum(counts) / values.size


def identity_counts(selected, aliases):
    """One row per instrument and recorded actor, retaining literal source IDs."""
    mapping = aliases.select('instrument', 'actor_key', 'actor_alias').unique()
    counts = selected.group_by('instrument', 'actor_key', 'actor_id', 'identity_level').agg(
        pl.len().alias('executions'),
        (pl.col('execution_anchor_mode') == 'passive').sum().alias('passive'),
        (pl.col('execution_anchor_mode') == 'aggressive').sum().alias('aggressive'),
    ).join(mapping, on=['instrument', 'actor_key'], validate='m:1')
    assert counts['executions'].sum() == selected.height
    assert (counts['passive'] + counts['aggressive']).equals(counts['executions'])
    assert counts['actor_alias'].null_count() == 0
    assert set(counts['identity_level']) <= {'firm', 'client_original'}
    return counts.sort(['instrument', 'identity_level', 'actor_id'])


def render_additions(population, joined, selected, aliases, output, table_writer):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    actors = identity_counts(selected, aliases)
    actors.write_csv(output / 'identified_participants.csv')
    rows = []
    for r in actors.to_dicts():
        source_id = r['actor_id'].replace('_', r'\_')
        firm = r'\texttt{' + source_id + '}' if r['identity_level'] == 'firm' else '--'
        client = r'\texttt{' + source_id + '}' if r['identity_level'] == 'client_original' else '--'
        rows.append([r['instrument'].title(), r['actor_alias'], firm, client,
                     r['executions'], r['passive'], r['aggressive']])
    table_writer(output / 'identified_participants.tex',
        'Recorded participants in the final conditional top-J selection, by instrument. '
        'The alias links each participant to the trajectory labels. Firm ID is the original firm-fallback identifier; '
        'client ID is the original client short code. A dash means that the identifier is not the attribution key '
        'for that row, not that the associated firm or client is absent from the source data. '
        'N, P, and A count selected executions, passive executions, and aggressive executions. '
        'Repeated firm IDs across instruments retain their original identity; firm attribution can pool several clients.',
        'tab:wmsci-j-identities',
        ['Instrument', 'Alias', 'Firm ID', 'Client ID', '$N$', '$P$', '$A$'], rows, 'llllrrr')

    plot_rows = []
    counts = {}
    with plt.rc_context({'font.size': 13, 'axes.labelsize': 14, 'axes.titlesize': 14,
                         'legend.fontsize': 11}):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
        for instrument, color in zip(sorted(population['instrument'].unique()),
                                     ['#2563eb', '#d97706', '#dc2626']):
            positive = population.filter((pl.col('instrument') == instrument) &
                pl.col('WMSCI_event').is_finite() & (pl.col('WMSCI_event') > 0))
            tail = joined.filter((pl.col('instrument') == instrument) &
                pl.col('complete_grid') & pl.col('J').is_finite())
            counts[instrument] = {'positive_W': positive.height, 'conditional_J': tail.height}
            for ax, metric, frame, column in [(axes[0], 'WMSCI', positive, 'WMSCI_event'),
                                              (axes[1], 'J', tail, 'J')]:
                raw_x, y = ecdf(frame[column].to_numpy())
                x = np.log1p(raw_x) if metric == 'WMSCI' else raw_x
                # Begin at zero probability immediately before the first observation.
                start = min(0., float(np.nextafter(x[0], -np.inf)))
                ax.step(np.r_[start, x], np.r_[0., y], where='post', color=color,
                        lw=2, label=f'{instrument.title()} (n={frame.height:,})')
                plot_rows.extend({'instrument': instrument, 'metric': metric,
                    'value': float(v), 'display_value': float(d), 'cumulative_probability': float(p),
                    'n': frame.height} for v, d, p in zip(raw_x, x, y))
        axes[0].set_title('(a) Positive WMSCI', loc='left')
        axes[1].set_title('(b) J within top 10% WMSCI', loc='left')
        axes[0].set_xlabel('log(1 + WMSCI) (dimensionless)')
        axes[1].set_xlabel('J (dimensionless)')
        axes[0].set_ylabel('Cumulative probability')
        for ax in axes:
            ax.set_ylim(0, 1.02)
            ax.set_xlim(left=-.02)
            ax.grid(alpha=.2)
            ax.legend(loc='lower right', frameon=False)
        fig.tight_layout(w_pad=2)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax in axes:
            for artist in [ax._left_title, ax.xaxis.label, ax.yaxis.label, ax.get_legend(),
                           *ax.get_xticklabels(), *ax.get_yticklabels()]:
                if artist.get_visible() and (not hasattr(artist, 'get_text') or artist.get_text()):
                    b = artist.get_window_extent(renderer)
                    # Only check ticks inside the displayed axis range.
                    if artist in ax.get_xticklabels() and not ax.get_xlim()[0] <= artist.get_position()[0] <= ax.get_xlim()[1]:
                        continue
                    if artist in ax.get_yticklabels() and not ax.get_ylim()[0] <= artist.get_position()[1] <= ax.get_ylim()[1]:
                        continue
                    assert b.x0 >= -1 and b.y0 >= -1 and b.x1 <= fig.bbox.x1 + 1 and b.y1 <= fig.bbox.y1 + 1, (artist, b)
        for ext in ['pdf', 'png']:
            fig.savefig(output / f'wmsci_j_ecdf.{ext}', dpi=180)
        plt.close(fig)
    pl.DataFrame(plot_rows).write_csv(output / 'wmsci_j_ecdf.csv')
    return {'population_counts': counts, 'identity_rows': actors.height,
            'identity_policy': 'original attribution identifiers, explicitly requested by author',
            'cdf_policy': 'right-continuous steps; W positive finite; J complete finite within exact W tail; zeros in J retained',
            'geometry_bounds_checked': True}
