"""Add s/sqrt(N) uncertainty to retained H1 hit frequencies; no replay."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import re
import polars as pl

REPO = Path(__file__).resolve().parents[1]
CAL = REPO / 'outputs/depth_kernel_calibration/20261012_100415_top10_h1'
SOURCE = REPO / 'paper/generated/new_kernel_h1_kernel/kernel_figures.tex'


def iid_hit_se(hits: int, exposures: int) -> float | None:
    """Sample SD (ddof=1) / sqrt(N) for binary snapshot-hit indicators."""
    if not isinstance(hits, int) or not isinstance(exposures, int):
        raise ValueError('Counts must be integers')
    if exposures < 0 or hits < 0 or hits > exposures:
        raise ValueError('Require 0 <= hits <= exposures')
    if exposures < 2:
        return None
    p = hits / exposures
    return math.sqrt(p * (1.0 - p) / (exposures - 1))


def generate(output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    text = SOURCE.read_text()
    prefix, hit = text.split('% data column: hit_probability;', 1)
    hit = '% data column: hit_probability;' + hit
    parts = re.split(r'(\\begin\{scope\}\[xshift=[^\]]+\])', hit)
    assert len(parts) == 7
    rows = []
    bindings = {str(SOURCE.relative_to(REPO)): hashlib.sha256(SOURCE.read_bytes()).hexdigest()}
    for index, instrument in enumerate(['RISANAMENTO', 'NEXI', 'FERRARI']):
        path = CAL / f'{instrument}_top10_h1/empirical_depth_kernel.csv'
        frame = pl.read_csv(path)
        assert frame.height == 20
        assert frame.select('side', 'rank').unique().height == 20
        bindings[str(path.relative_to(REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
        bars = []
        for side, color in [('ask', 'blue!70!black'), ('bid', 'orange!85!black')]:
            group = frame.filter(pl.col('side') == side).sort('rank')
            assert group['rank'].to_list() == list(range(1, 11))
            for row in group.iter_rows(named=True):
                n, h = row['exposure_count'], row['hit_count']
                p = h / n if n else None
                if p is None or not math.isclose(p, row['hit_probability'], rel_tol=1e-14, abs_tol=1e-16):
                    raise ValueError('Retained hit center disagrees with counts')
                se = iid_hit_se(h, n)
                rows.append({'instrument_id': instrument, 'side': side, 'rank': row['rank'],
                             'exposure_count': n, 'hit_count': h, 'center': row['hit_probability'],
                             'sample_sd': se * math.sqrt(n) if se is not None else None,
                             'standard_error': se, 'method': 'sample_sd_ddof1_over_sqrt_N_iid'})
                if se is not None:
                    k = row['rank']
                    # Preserve symmetric SE bars; do not truncate them at probability bounds.
                    low, high = p - se, p + se
                    bars.append(f'  \\draw[{color},line width=0.5pt] ({k},{low:.17g}) -- ({k},{high:.17g});')
                    for y in [low, high]:
                        bars.append(f'  \\draw[{color},line width=0.5pt] ({k-0.13:.2f},{y:.17g}) -- ({k+0.13:.2f},{y:.17g});')
        body_index = 2 + 2 * index
        parts[body_index] = parts[body_index].replace('\\end{scope}', '\n'.join(bars) + '\n\\end{scope}', 1)
    hit = ''.join(parts)
    caption = (r' Bars show $\pm s/\sqrt{N}$, where $N$ is the number of exposed snapshots for each instrument, side and rank and $s$ is the sample standard deviation of their binary hit indicators (divisor $N-1$). These analytical standard errors assume independent observations and do not correct for overlapping windows or serial dependence. A zero bar for zero observed hits is an empirical plug-in value, not evidence of zero uncertainty.')
    needle = r'\label{fig:empirical_topn_execution_risk_profiles}'
    assert hit.count(needle) == 1
    end = hit.index(needle)
    pos = hit.rfind('}', 0, end)
    hit = hit[:pos] + caption + hit[pos:]
    output.mkdir(parents=True)
    pl.DataFrame(rows).write_csv(output / 'hit_frequency_standard_errors.csv')
    (output / 'hit_frequency.tex').write_text(hit)
    (output / 'kernel_figures.tex').write_text(prefix + hit)
    artifacts = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}
    metadata = {'method': 'sample_sd_ddof1_over_sqrt_N_iid', 'formula': 'sqrt(p*(1-p)/(N-1))',
                'H_seconds': 1, 'rows': len(rows), 'source_bindings': bindings,
                'other_kernel_figures_preserved_exactly': True,
                'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'serial_dependence_adjusted': False, 'artifacts': artifacts}
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Rendered {len(rows)} hit-frequency centers and analytical SEs; no detection or raw replay.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    generate(parser.parse_args().output_root)
