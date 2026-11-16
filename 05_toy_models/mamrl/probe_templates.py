"""Symmetric probe games and an exact fixed-opponent information witness."""
from dataclasses import replace
from functools import lru_cache
import random

from .common import finite_reward, validate_seed
from .relational_rules import Relation, Reward, RelationalRule, cyclic_outcomes, relational_to_dict, relational_from_dict, describe_relational_rule
from .rule_metrics import common_values, canonical_values, rounded_values, stable_digest
from .relational_catalog import audit_relational_rule


def build_probe_rule(*, safe_reward=1, risk_scale=3, marker_cost=.2, theta=1, roles=(0,1,2), exception_bonus=0, repeat_bonus=0, copy_bonus=0):
    safe_reward, risk_scale, marker_cost, exception_bonus, repeat_bonus, copy_bonus = map(finite_reward, (safe_reward,risk_scale,marker_cost,exception_bonus,repeat_bonus,copy_bonus))
    roles=tuple(roles)
    if sorted(roles)!=[0,1,2] or any(type(x) is not int for x in roles):raise ValueError('roles must permute 0,1,2')
    if type(theta) is not int or theta not in (-1,1):raise ValueError('theta must be -1 or 1')
    if risk_scale<=0 or marker_cost<=0:raise ValueError('risk scale and marker cost must be positive')
    base=cyclic_outcomes(3);inverse={a:i for i,a in enumerate(roles)}
    outcomes=tuple(tuple(base[inverse[i]][inverse[j]] for j in range(3)) for i in range(3))
    safe=Relation('eq',left='own',right=roles[0]);cost=Relation('eq',left='own',right=roles[1])
    risky=Reward('add',children=(Reward('scale',value=theta*risk_scale,children=(Reward('outcome',left='own',right='other'),)),
                                Reward('if',condition=cost,children=(Reward('const',value=-marker_cost),Reward('const',value=0)))))
    root=Reward('if',condition=safe,children=(Reward('const',value=safe_reward),risky))
    for variable, bonus in (('prev_own',repeat_bonus), ('prev_other',copy_bonus)):
        if bonus:
            relation=Relation('eq',left='own',right=variable)
            term=Reward('if',condition=relation,children=(Reward('const',value=bonus),Reward('const',value=0)))
            root=Reward('add',children=(root,term))
    if exception_bonus:
        both=Relation('and',children=(Relation('eq',left='own',right='prev_own'),Relation('eq',left='other',right='prev_other')))
        root=Reward('add',children=(root,Reward('if',condition=both,children=(Reward('const',value=exception_bonus),Reward('const',value=0)))))
    return RelationalRule(3,outcomes,root)


def _pair(seed, split, pair_index):
    rng=random.Random(seed); roles=list(range(3));rng.shuffle(roles)
    return {'roles':roles,'safe_reward':round(rng.uniform(.5,1.4),8), 'risk_scale':3.0,
            'marker_cost':round(rng.uniform(.15,.5),8),
            'exception_bonus':0.0, 'repeat_bonus':.15 if split=='structural' or pair_index%2==0 else 0.0,
            'copy_bonus':.15 if split=='structural' or pair_index%2==1 else 0.0}


def _entry(index, split, seed, theta):
    params={**_pair(seed,split,index//2),'theta':theta}
    source=build_probe_rule(**params); raw=common_values(source);lo,hi=min(raw),max(raw)
    factor,shift=2/(hi-lo),-1-2*lo/(hi-lo)
    played=replace(source,scale=factor,shift=shift);values=common_values(played)
    return {'id':f'r{index:03d}','split':split,'family':'temporal_recombination' if split=='structural' else 'probe_template',
            'seed':seed,'parameters':params,'source_rule':relational_to_dict(source),'rule':relational_to_dict(played),
            'normalization':{'factor':factor,'shift':shift},'common_payoffs':list(values),
            'fingerprint':stable_digest(rounded_values(values)),
            'canonical_fingerprint':stable_digest(canonical_values(values,3)),
            'description':describe_relational_rule(source),'source_audit':audit_relational_rule(source),
            'audit':audit_relational_rule(played)}


def build_probe_manifest(*,n=3,seed=20261116,train_rules=8,fresh_rules=8,structural_rules=8):
    if type(n) is not int or n!=3:raise ValueError('probe catalog currently requires n=3')
    if validate_seed(seed) is None:raise TypeError('explicit seed required')
    counts=(train_rules,fresh_rules,structural_rules)
    if any(type(x) is not int or x<=0 or x%2 for x in counts):raise ValueError('counts must be positive even integers')
    rng=random.Random(seed);entries=[];seen=set();rejected=0
    for split,count in zip(('train','fresh_rules','structural'),counts):
        for _ in range(count//2):
            while True:
                seedpair=rng.getrandbits(63);i=len(entries)
                pair=[_entry(i,split,seedpair,-1),_entry(i+1,split,seedpair,1)]
                fps=[x['canonical_fingerprint'] for x in pair]
                if len(set(fps))==2 and not any(x in seen for x in fps):break
                rejected+=1
                if rejected>20000:raise RuntimeError('could not generate distinct probe games')
            seen.update(fps);entries.extend(pair)
    result={'version':3,'catalog':'probe_template','n':3,'seed':seed,'rules':entries,
            'counts':dict(zip(('train','fresh_rules','structural'),counts)),
            'catalog_audit':{'duplicate_pairs_rejected':rejected},
            'structural_note':'heldout combines individually trained repeat/copy bonuses; no new relation',
            'sampling_note':'theta pairs share parameters; role permutations are not independent structural diversity'}
    validate_probe_manifest(result);return result


def validate_probe_manifest(manifest):
    if manifest.get('version')!=3 or manifest.get('catalog')!='probe_template' or manifest.get('n')!=3:raise ValueError('invalid probe catalog')
    if validate_seed(manifest['seed']) is None:raise ValueError('explicit seed required')
    entries=manifest['rules'];counts=manifest['counts'];seen=set();i=0
    sampler=random.Random(manifest['seed']);rejected=0
    for split in ('train','fresh_rules','structural'):
        count=counts[split]
        if type(count) is not int or count<=0 or count%2:raise ValueError('invalid split count')
        for _ in range(count//2):
            while True:
                expected_seed=sampler.getrandbits(63)
                candidate=[_entry(i,split,expected_seed,-1),_entry(i+1,split,expected_seed,1)]
                fps=[item['canonical_fingerprint'] for item in candidate]
                if len(set(fps))==2 and not any(fp in seen for fp in fps):break
                rejected+=1
                if rejected>20000:raise ValueError('too many rejected candidates')
            for theta in (-1,1):
                if i>=len(entries):raise ValueError('missing entry')
                e=entries[i]
                if e['seed']!=expected_seed or e['split']!=split or e['id']!=f'r{i:03d}' or e['parameters']['theta']!=theta:raise ValueError('invalid split/id/theta')
                if theta==1 and e['seed']!=entries[i-1]['seed']:raise ValueError('pair seed mismatch')
                if e!=_entry(i,split,e['seed'],theta):raise ValueError('rule/source/seed/fingerprint mismatch')
                fp=e['canonical_fingerprint']
                if fp in seen:raise ValueError('duplicate canonical game')
                seen.add(fp);i+=1
    if i!=len(entries):raise ValueError('unexpected entries')
    if manifest['catalog_audit']!={'duplicate_pairs_rejected':rejected}:raise ValueError('rejection audit mismatch')
    return manifest


def exact_information_audit(*,safe_reward=1,risk_scale=3,marker_cost=.2,horizon=2):
    q,r,c=map(finite_reward,(safe_reward,risk_scale,marker_cost))
    if r<=0 or c<=0 or type(horizon) is not int or horizon<=0:raise ValueError('invalid audit parameters')
    # Fixed opponent plays role 0. Payoff order follows cyclic_outcomes(3).
    payoff={-1:(q,-r-c,r),1:(q,r-c,-r)}
    @lru_cache(None)
    def value(left,possible):
        if not left:return 0.0
        return max(sum((reward+value(left-1,tuple(t for t in possible if payoff[t][a]==reward)))
                       for reward in {payoff[t][a] for t in possible}
                       for _ in range(sum(payoff[t][a]==reward for t in possible)))/len(possible)
                   for a in range(3))
    @lru_cache(None)
    def myopic(left,possible):
        if not left:return 0.0
        means=[sum(payoff[t][a] for t in possible)/len(possible) for a in range(3)]
        a=max(range(3),key=lambda x:means[x])
        return sum(payoff[t][a]+myopic(left-1,tuple(x for x in possible if payoff[x][a]==payoff[t][a])) for t in possible)/len(possible)
    prior=(-1,1);immediate=[sum(payoff[t][a] for t in prior)/2 for a in range(3)]
    action_values=[]
    for a in range(3):
        action_values.append(sum((payoff[t][a]+value(horizon-1,tuple(x for x in prior if payoff[x][a]==payoff[t][a]))) for t in prior)/2)
    optimal=max(action_values);no_info=horizon*max(immediate)
    oracle=sum(horizon*max(payoff[t]) for t in prior)/2
    return {'prior':{'-1':.5,'1':.5},'fixed_opponent_action':0,'horizon':horizon,
            'payoffs':{str(k):list(v) for k,v in payoff.items()},'immediate_expected':immediate,
            'optimal_initial_actions':[a for a,v in enumerate(action_values) if abs(v-optimal)<1e-10],
            'optimal_total':optimal,'no_information_total':no_info,'myopic_total':myopic(horizon,prior),'oracle_total':oracle,
            'information_value':optimal-no_info,'exploration_cost':max(immediate)-immediate[2],
            'reachable_reward_histories':{'action_0':[q],'action_1':[-r-c,r-c],'action_2':[r,-r]},
            'scope':'fixed opponent witness only; not a guarantee against learning opponents'}
