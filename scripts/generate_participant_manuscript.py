#!/usr/bin/env python3
"""Compact horizon figure and participant summaries from frozen rank-only inputs."""
from __future__ import annotations
import argparse
import importlib.util
import math
import json
import shutil
from pathlib import Path
import polars as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
REVIEW=Path('paper/reviews/20261015_152614_participant_results')
OUTPUT=Path('paper/generated/withdrawal_rank_20261015_participants_v3')
SOURCE_PACKAGE=Path('paper/generated/withdrawal_rank_20261015_v2')
INSTRUMENTS=['FERRARI','NEXI','RISANAMENTO']
_spec=importlib.util.spec_from_file_location('participant_frozen_base',Path(__file__).with_name('generate_rank_only_manuscript.py'))
base=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)


def candidate_count_figure(counts):
    """One axis, three literal raw-count curves; no interpolation or normalization."""
    data=counts.filter(pl.col('age_policy')=='exp_h30').select('instrument','horizon_ns','candidate_executions').sort(['instrument','horizon_ns'])
    required={(i,h*10**9) for i in INSTRUMENTS for h in [1,2,5,10]}
    if data.height!=12 or set(data.select('instrument','horizon_ns').iter_rows())!=required:
        raise ValueError('expected twelve distinct primary-kernel horizon settings')
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':15,'axes.labelsize':15,
                         'xtick.labelsize':14,'ytick.labelsize':14,'legend.fontsize':14,'pdf.fonttype':42}):
        fig,ax=plt.subplots(figsize=(6.8,4.3),layout='constrained')
        for instrument,color,marker in zip(INSTRUMENTS,['#0072B2','#D55E00','#009E73'],['o','s','^']):
            sub=data.filter(pl.col('instrument')==instrument)
            ax.plot(sub['horizon_ns']/10**9,sub['candidate_executions'],label=instrument.title(),
                    color=color,marker=marker,linewidth=2,markersize=7)
        ax.set_xlabel('Withdrawal horizon H (seconds)')
        ax.set_ylabel('Candidate executions (count)')
        ax.set_xticks([1,2,5,10])
        upper=max(1000,math.ceil(data['candidate_executions'].max()/1000)*1000)
        ax.set_ylim(0,upper);ax.set_yticks(range(0,upper+1,1000))
        ax.grid(alpha=.2)
        ax.legend(loc='lower center',bbox_to_anchor=(.5,1.02),ncol=3,
                  columnspacing=1.0,handlelength=1.8,frameon=False)
    return fig,data


def numeric(value):
    if value is None or (isinstance(value,float) and not math.isfinite(value)):
        return '--'
    if isinstance(value,int):
        return f'{value:,}'
    return r'\num{'+format(value,'.4g')+'}'


def render_participants(out,tables):
    summary=tables['summary'].sort(['instrument','perspective'])
    labels={'identified_client':'Clients','intermediary':'Intermediaries'}
    rows=[]
    for row in summary.to_dicts():
        rows.append([row['instrument'].title(),labels[row['perspective']],
            f"{row['candidate_entity_count']}/{row['entity_count']}",row['execution_count'],
            row['candidate_count'],f"{row['largest_candidate_group_count']}/{row['candidate_count']}"])
    base.table(out,'participant_summary.tex','tab:participant-summary',
        'Primary results grouped by identified client and by intermediary, separately within instrument. '
        'Entities gives groups with candidates / all observed groups. Executions is all activity in that view; '
        'candidates counts the raw physical sequences. Largest gives the largest group\'s candidate count / '
        'all candidates in that view, not its candidate rate. Intermediaries include both client-identified and '
        'firm-only activity. The two views overlap and must not be added.',
        ['Instrument','View',r'\shortstack{Entities\\with/all}',r'\shortstack{All\\executions}',
         'Candidates',r'\shortstack{Largest\\group / total}'],rows)
    for perspective,filename,title,label in [
        ('identified_client','client_groups.tex','Identified-client','tab:client-groups'),
        ('intermediary','intermediary_groups.tex','Intermediary','tab:intermediary-groups')]:
        entities=tables['entities'].filter((pl.col('perspective')==perspective)&(pl.col('candidate_count')>0)).sort(['instrument','entity_alias'])
        caption=(title+' groups with at least one primary candidate. All is the group\'s observed executions; '
            r'$N$ is its candidate executions and the percentage is $100N/\mathrm{All}$. '
            '$W$ and $J$ are medians over supported candidate values, not participant scores. '
            'Both counts positive FPM and complete positive two-second REV with the full $N$ denominator. '
            'Zero, adverse and missing prices are not exclusions. Aliases are stable across instruments; '
            'rows are in alias order, not a new priority ranking.')
        header=r'Group & All & $N$ & $N$/All (\%) & Median $W$ & Median $J$ & Both \\'
        text='\\begingroup\n\\small\n\\setlength{\\LTcapwidth}{\\textwidth}\n'
        text+='\\begin{longtable}{lrrrrrr}\n\\caption{'+caption+'}\\label{'+label+'}\\\\\n'
        text+='\\toprule\n'+header+'\n\\midrule\n\\endfirsthead\n'
        text+='\\multicolumn{7}{l}{'+title+' groups (continued)}\\\\\n\\toprule\n'+header+'\n\\midrule\n\\endhead\n'
        text+='\\midrule\\multicolumn{7}{r}{Continued on next page}\\\\\n\\endfoot\n\\bottomrule\n\\endlastfoot\n'
        last=None
        records=entities.to_dicts()
        for index,row in enumerate(records):
            if row['instrument']!=last:
                text+='\\multicolumn{7}{l}{\\textbf{'+row['instrument'].title()+'}}\\\\*\n'
                last=row['instrument']
            cells=[row['entity_alias'],row['execution_count'],row['candidate_count'],
                100*row['candidate_fraction'],row['W_median'],row['J_median'],row['FPM_REV_positive_count']]
            same_group=index+1<len(records) and records[index+1]['instrument']==row['instrument']
            text+=' & '.join(numeric(v) if not isinstance(v,str) else v for v in cells)+(r' \\*' if same_group else r' \\')+'\n'
        text+='\\end{longtable}\n\\endgroup\n'
        (out/filename).write_text(text)


def build(out):
    out=Path(out)
    if out.exists():
        raise FileExistsError(f'publication output exists: {out}; use a fresh directory')
    if out.resolve().parent!=(ROOT/'paper/generated').resolve() or not out.name.startswith('withdrawal_rank_'):
        raise ValueError('use a fresh withdrawal_rank_* under paper/generated')
    from scripts.summarize_ranked_participants import load_native_participants
    baseline=base.load_json(ROOT/REVIEW/'baseline_identity.json')
    for path,expected in baseline['protected'].items():
        if base.digest(ROOT/path)!=expected:
            raise ValueError('protected input changed: '+path)
    base.validate_stage(ROOT/SOURCE_PACKAGE,'publication_package_verified')
    result=load_native_participants()
    # Start from authenticated published bytes: the new analysis is a downstream
    # grouping, not a rerun or a hand-edit of historical scientific tables.
    prior=base.load_json(ROOT/SOURCE_PACKAGE/'artifact_manifest.json')
    out.mkdir(parents=True,exist_ok=False)
    retired={'candidate_counts.pdf','withdrawal_and_profile_summary.pdf','facts.json','style_metadata.json'}
    for name in prior['artifacts']:
        if name not in retired:
            shutil.copyfile(ROOT/SOURCE_PACKAGE/name,out/name)
            assert base.digest(out/name)==prior['artifacts'][name]
    for name,frame in result['tables'].items():
        frame.write_csv(out/('participant_'+name+'.csv'))
    entities=result['tables']['entities']
    for perspective,name in [('identified_client','client_groups.csv'),('intermediary','intermediary_groups.csv')]:
        entities.filter(pl.col('perspective')==perspective).write_csv(out/name)
    render_participants(out,result['tables'])
    counts=pl.read_csv(out/'cell_counts.csv')
    fig,data=candidate_count_figure(counts)
    data.write_csv(out/'candidate_count_curves.csv')
    figure=base._old.save_figure(fig,out,'candidate_counts.pdf',
        'Primary-kernel candidate counts at four horizons; three instruments on one common count scale. '
        'Lines connect settings, not an event-time trajectory. Samples have different coverage; no prevalence interpretation.',
        'candidate_count_curves.csv')
    private=ROOT/REVIEW/'private_identity_mapping.parquet'
    provenance=ROOT/REVIEW/'participant_input_provenance.json'
    if private.exists():
        assert pl.read_parquet(private).equals(result['identity_mapping'],null_equal=True),'private identity map drift'
    else:
        result['identity_mapping'].write_parquet(private)
    if provenance.exists():
        assert base.load_json(provenance)==result['provenance'],'participant input provenance drift'
    else:
        base.write_json(provenance,result['provenance'])
    bindings={str(SOURCE_PACKAGE/'artifact_manifest.json'):base.digest(ROOT/SOURCE_PACKAGE/'artifact_manifest.json'),
        str(REVIEW/'baseline_identity.json'):base.digest(ROOT/REVIEW/'baseline_identity.json'),
        str(REVIEW/'participant_input_provenance.json'):base.digest(provenance),
        str(Path(__file__).relative_to(ROOT)):base.digest(__file__),
        'scripts/summarize_ranked_participants.py':base.digest(ROOT/'scripts/summarize_ranked_participants.py')}
    facts=base.load_json(ROOT/SOURCE_PACKAGE/'facts.json')
    facts.update(version='withdrawal_rank_participants_v1',
        presentation='one three-line horizon plot; diagnostic-case figure omitted; client and intermediary summaries added',
        reused_assets=[],figures=[figure],participant_summary=result['tables']['summary'].to_dicts(),
        participant_accounting=result['tables']['accounting'].to_dicts(),
        participant_entity_rows=entities.height,
        participant_strata_rows=result['tables']['entity_strata'].height,
        participant_group_rule='identified canonical clients; original execution firm across both client and fallback activity; views overlap',
        participant_identity_scope='stable report-local C/F aliases across loaded instruments; not formal anonymization',
        member_firm_validation=result['provenance']['member_firm_validation'],
        presentation_input_bindings=bindings)
    base.write_json(out/'facts.json',facts)
    base.write_json(out/'style_metadata.json',{'figures':[figure],'minimum_font_pt':13,
        'publication_width':r'\textwidth','asset_policy':'new common-axis plot from unchanged counts; no case-profile figure'})
    base.write_json(out/'artifact_manifest.json',{'version':facts['version'],
        'artifacts':{p.name:base.digest(p) for p in sorted(out.iterdir()) if p.is_file()},
        'presentation_input_bindings':bindings})
    base.write_json(out/'COMPLETE.json',{'status':'publication_package_verified','scope':'all_candidate_market_rank_only_with_participant_groups',
        'artifact_manifest_sha256':base.digest(out/'artifact_manifest.json'),
        'primary_candidate_executions':6914,'participant_entity_rows':entities.height})
    base.validate_stage(out,'publication_package_verified')
    print(json.dumps({'output':str(out),'participant_summary':facts['participant_summary'],
        'member_firm_validation':facts['member_firm_validation']},indent=2))
    return facts


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args()
    build(args.output if args.output.is_absolute() else ROOT/args.output)
