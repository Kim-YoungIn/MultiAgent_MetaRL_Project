"""Semantic-game catalogs with audits computed from the ACTUAL payoff tensor."""

from dataclasses import replace
import itertools
import random

from .common import validate_n, validate_seed, finite_reward
from .rule_metrics import (stable_digest, rounded_values, canonical_values, common_values,
                           tensor_distance, expression_tokens, depends_on_axis)
from .relational_rules import (generate_relational_rule, social_rule, relational_to_dict,
                               relational_from_dict, describe_relational_rule)
from .rules import compile_rule


BASE_FAMILIES=('social','contest','reciprocity','hidden')


def audit_relational_rule(rule):
    """No PD label inferred from syntax/name. Exhaust all realized stage payoffs."""
    n=rule.n
    tensor=compile_rule(rule)
    values=tensor.values[:n**4]
    quadrant_values={k:[] for k in ('R','S','T','P')}
    all_quadrants=True
    dominant=True
    best_responses=[]
    for p,q in itertools.product(range(n),repeat=2):
        quadrants=set()
        for a,b in itertools.product(range(n),repeat=2):
            c0,c1=a==p,b==q
            key='R' if c0 and c1 else ('S' if c0 else ('T' if c1 else 'P'))
            quadrants.add(key)
            quadrant_values[key].append(tensor[0,p,q,a,b])
        all_quadrants &= len(quadrants)==4
        choices=[]
        for b in range(n):
            rewards=[tensor[0,p,q,a,b] for a in range(n)]
            maximum=max(rewards)
            choices.append([a for a,r in enumerate(rewards) if abs(r-maximum)<1e-10])
            dominant &= all(rewards[a]>rewards[p]+1e-10 for a in range(n) if a!=p)
        common=set(choices[0])
        for acts in choices[1:]: common &= set(acts)
        best_responses.append({'previous':[p,q],'by_opponent_action':choices,
                               'common_best_response':sorted(common),
                               'response_sets_differ':any(a!=choices[0] for a in choices[1:]),
                               'no_universal_pure_best_response':not common})
    exact=all_quadrants and all(max(v)-min(v)<1e-10 for v in quadrant_values.values())
    payoffs={k:v[0] for k,v in quadrant_values.items()} if exact else None
    pd=False
    efficiency=None
    game_class='not_exact_repeat_change_game'
    if exact:
        R,S,T,P=(payoffs[k] for k in ('R','S','T','P'))
        pd=T>R+1e-10 and R>P+1e-10 and P>S+1e-10
        efficiency=2*R>T+S+1e-10
        if T>R and P>S: game_class='defection_dominant'
        elif R>T and S>P: game_class='cooperation_dominant'
        elif R>T and P>S: game_class='coordination'
        elif T>R and S>P: game_class='anti_coordination'
        else: game_class='ties_or_boundary'
    return {'pd':{'exact_embedding':bool(exact),'is_prisoners_dilemma':bool(pd),
                  'quadrants_reachable':bool(all_quadrants),'payoffs':payoffs,
                  'quadrant_ranges':{k:[min(v),max(v)] for k,v in quadrant_values.items()},
                  'defection_strictly_dominates':bool(dominant),
                  'two_R_gt_T_plus_S':efficiency,'game_class':game_class},
            'axis_dependence':[depends_on_axis(values,n,i) for i in range(4)],
            'best_responses':best_responses,
            'states_without_universal_pure_best_response':sum(x['no_universal_pure_best_response'] for x in best_responses),
            'interpretation':'stage action sets only; not repeated-game strategy equivalence or exploration necessity'}


def _family_for(index):
    return BASE_FAMILIES[index%4] if index<20 else 'exception'


def _split_for(index):
    return 'train' if index<12 else ('fresh_rules' if index<20 else 'structural')


SOCIAL_STRATA={0:'prisoners_dilemma',4:'coordination',8:'anti_coordination',
               12:'cooperation_dominant',16:'prisoners_dilemma'}


def _matches_social_stratum(audit, stratum):
    pd=audit['pd']
    return pd['is_prisoners_dilemma'] if stratum=='prisoners_dilemma' else pd['exact_embedding'] and pd['game_class']==stratum


def build_relational_manifest(*,n=3,seed=20261114):
    validate_n(n)
    if n>5: raise ValueError('permutation audit is bounded at n<=5')
    if validate_seed(seed) is None: raise TypeError('explicit catalog seed required')
    rng=random.Random(seed)
    entries=[];seen=set()
    rejections={'attempts':0,'constant':0,'duplicate':0,'incentive_stratum':0}
    while len(entries)<28:
        if rejections['attempts']>=20000:
            raise RuntimeError('could not generate the declared distinct catalog')
        index=len(entries)
        source_seed=rng.getrandbits(63)
        family=_family_for(index)
        rejections['attempts']+=1
        source=social_rule(n=n,T=5,R=3,P=1,S=0) if index==0 else generate_relational_rule(n=n,seed=source_seed,family=family)
        raw=common_values(source)
        low,high=min(raw),max(raw)
        if high-low<1e-10:
            rejections['constant']+=1;continue
        source_audit=audit_relational_rule(source)
        stratum=SOCIAL_STRATA.get(index)
        if stratum and not _matches_social_stratum(source_audit,stratum):
            rejections['incentive_stratum']+=1;continue
        factor,shift=2/(high-low),-1-2*low/(high-low)
        played=replace(source,scale=source.scale*factor,shift=source.shift*factor+shift)
        values=common_values(played)
        canonical=stable_digest(canonical_values(values,n))
        if canonical in seen:
            rejections['duplicate']+=1;continue
        seen.add(canonical)
        entries.append({'id':f'r{index:03d}','split':_split_for(index),'family':family,'seed':source_seed,
                        'source_kind':'declared_pd_control' if index==0 else 'sampled',
                        'incentive_stratum':stratum,
                        'source_rule':relational_to_dict(source),'rule':relational_to_dict(played),
                        'normalization':{'factor':factor,'shift':shift},
                        'common_payoffs':list(values),'fingerprint':stable_digest(rounded_values(values)),
                        'canonical_fingerprint':canonical,'description':describe_relational_rule(source),
                        'source_audit':source_audit,'audit':audit_relational_rule(played)})
    result={'version':2,'catalog':'relational','n':n,'seed':seed,'rules':entries,
            'catalog_audit':rejections,'base_families':list(BASE_FAMILIES),
            'normalization_note':'positive affine transform defines actual task; online learning path not preserved',
            'semantic_note':'repeat=C/change=D; uniform raw actions are not uniform C/D',
            'sampling_note':'no mandatory current-action-match component; no filtering by neural performance'}
    validate_relational_manifest(result)
    return result


def validate_relational_manifest(manifest):
    if manifest.get('version')!=2 or manifest.get('catalog')!='relational':
        raise ValueError('unsupported relational catalog')
    n=validate_n(manifest['n'])
    entries=manifest['rules']
    if [r['id'] for r in entries]!=[f'r{i:03d}' for i in range(28)]:
        raise ValueError('expected predefined 28 catalog slots')
    seen=set()
    for i,entry in enumerate(entries):
        if entry['family']!=_family_for(i) or entry['split']!=_split_for(i):
            raise ValueError('catalog family/split was changed')
        source=relational_from_dict(entry['source_rule'])
        played=relational_from_dict(entry['rule'])
        if source.n!=n or played.n!=n: raise ValueError('action count mismatch')
        if validate_seed(entry['seed']) is None: raise ValueError('source seed must be explicit')
        expected_kind='declared_pd_control' if i==0 else 'sampled'
        if entry['source_kind']!=expected_kind: raise ValueError('source kind mismatch')
        if entry.get('incentive_stratum')!=SOCIAL_STRATA.get(i): raise ValueError('incentive stratum mismatch')
        regenerated=social_rule(n=n,T=5,R=3,P=1,S=0) if i==0 else generate_relational_rule(n=n,seed=entry['seed'],family=entry['family'])
        if relational_to_dict(regenerated)!=entry['source_rule']:
            raise ValueError('source seed does not reproduce the declared logical rule')
        norm=entry['normalization']
        if finite_reward(norm['factor'])<=0: raise ValueError('normalization factor must be positive')
        finite_reward(norm['shift'])
        expected=replace(source,scale=source.scale*norm['factor'],shift=source.shift*norm['factor']+norm['shift'])
        if relational_to_dict(expected)!=relational_to_dict(played):
            raise ValueError('incorrect normalized rule provenance')
        values=common_values(played)
        if rounded_values(values)!=rounded_values(entry['common_payoffs']) or entry['fingerprint']!=stable_digest(rounded_values(values)):
            raise ValueError('payoff fingerprint mismatch')
        if abs(min(values)+1)>1e-9 or abs(max(values)-1)>1e-9: raise ValueError('normalization range mismatch')
        canonical=stable_digest(canonical_values(values,n))
        if entry['canonical_fingerprint']!=canonical or canonical in seen:
            raise ValueError('duplicate or corrupt canonical fingerprint')
        seen.add(canonical)
        if entry['source_audit']!=audit_relational_rule(source) or entry['audit']!=audit_relational_rule(played):
            raise ValueError('compiled semantic audit mismatch')
        if i in SOCIAL_STRATA and not _matches_social_stratum(entry['audit'],SOCIAL_STRATA[i]):
            raise ValueError('social incentive stratum not realized by compiled payoff')
        if entry['description']!=describe_relational_rule(source): raise ValueError('description mismatch')
    first=entries[0]
    if not first['audit']['pd']['is_prisoners_dilemma'] or first['source_kind']!='declared_pd_control':
        raise ValueError('missing declared PD control')
    return manifest


def relational_distances(manifest):
    sources={r['id']:relational_from_dict(r['source_rule']) for r in manifest['rules']}
    played={r['id']:relational_from_dict(r['rule']) for r in manifest['rules']}
    train=[r['id'] for r in manifest['rules'] if r['split']=='train']
    def structure_difference(left,right):
        a,b=expression_tokens(relational_to_dict(left)),expression_tokens(relational_to_dict(right))
        keys=set(a)|set(b)
        union=sum(max(a[k],b[k]) for k in keys)
        return 1-sum(min(a[k],b[k]) for k in keys)/union if union else 0.0
    result={}
    for key,rule in played.items():
        rows=[{'train_rule_id':other,'raw_payoff_l1':tensor_distance(sources[key],sources[other]),
               'normalized_payoff_l1':tensor_distance(rule,played[other]),
               'permutation_payoff_l1':tensor_distance(rule,played[other],canonical=True),
               'structural_token_dissimilarity':structure_difference(rule,played[other])} for other in train]
        result[key]={'nearest_'+metric:min(r[metric] for r in rows) for metric in
                     ('raw_payoff_l1','normalized_payoff_l1','permutation_payoff_l1','structural_token_dissimilarity')}
    return result
