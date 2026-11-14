"""Independent contracts for learned exploration and adaptive opponents."""

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from mamrl.environment import Observation
from mamrl.rules import RuleSpec, RuleTerm

try:
    import torch
except ImportError:
    torch = None


class OnlineQContract(unittest.TestCase):
    def test_hand_calculated_q_learning_and_epsilon_greedy_distribution(self):
        from mamrl.adaptive_agents import OnlineQ

        state = Observation(0, 1, 0.0, False)
        following = Observation(2, 0, 1.0, False)
        q = OnlineQ(n=3, seed=17, alpha=.5, gamma=.95, epsilon=.2)
        for value in q.distribution(state):
            self.assertAlmostEqual(value, 1/3)
        # Q(following,1) = 0 + .5 * (2 - 0) = 1.
        q.update(following, 1, 2.0, following, terminal=True)
        # Q(state,0) = .5 * (1 + .95 * 1) = .975.
        q.update(state, 0, 1.0, following, terminal=False)
        distribution = q.distribution(state)
        for actual, expected in zip(distribution, (13/15, 1/15, 1/15)):
            self.assertAlmostEqual(actual, expected)
        # Terminal target must not bootstrap: .975 + .5 * (-2 - .975) = -.5125.
        q.update(state, 0, -2.0, following, terminal=True)
        distribution = q.distribution(state)
        for actual, expected in zip(distribution, (1/15, 7/15, 7/15)):
            self.assertAlmostEqual(actual, expected)

    def test_clone_preserves_rng_and_isolated_values_without_global_rng_effect(self):
        from mamrl.adaptive_agents import OnlineQ

        state = Observation(0, 1, 0.0, True)
        local = random.getstate()
        q = OnlineQ(n=3, seed=932, alpha=.5, gamma=.95, epsilon=.2)
        q.update(state, 1, 1.0, state, terminal=True)
        snapshot = q.snapshot()
        saved = json.dumps(snapshot, sort_keys=True, allow_nan=False)
        copy = q.clone()
        self.assertEqual([q.act(state) for _ in range(30)],
                         [copy.act(state) for _ in range(30)])
        copy.update(state, 0, 10.0, state, terminal=True)
        self.assertNotEqual(q.distribution(state), copy.distribution(state))
        self.assertEqual(json.dumps(snapshot, sort_keys=True, allow_nan=False), saved)
        self.assertEqual(random.getstate(), local)

    def test_nonfinite_invalid_actions_and_invalid_probabilities_are_rejected(self):
        from mamrl.adaptive_agents import OnlineQ

        state = Observation(0, 1, 0.0, True)
        for kwargs in ({'epsilon': -0.1}, {'epsilon': 1.01}, {'gamma': 1.1},
                       {'alpha': -1}, {'alpha': float('nan')}):
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                OnlineQ(n=3, seed=1, **kwargs)
        q = OnlineQ(n=3, seed=1)
        for action in (-1, 3, True):
            with self.subTest(action=action), self.assertRaises((TypeError, ValueError)):
                q.update(state, action, 1.0, state, terminal=False)
        with self.assertRaises((TypeError, ValueError)):
            q.update(state, 0, float('inf'), state, terminal=False)
        with self.assertRaises((TypeError, ValueError)):
            q.distribution(state, epsilon=1.5)


class MetaTrialContract(unittest.TestCase):
    @staticmethod
    def rule():
        # Hand calculation: both players receive 1 iff current actions match.
        return RuleSpec(3, (RuleTerm('coordination', 1),))

    @staticmethod
    def factory(*, n, seed, student):
        class Teacher:
            def act(self, observation):
                return observation.previous_opponent_action
        return Teacher()

    def test_recorded_decision_distributions_use_preupdate_learned_opponent_q(self):
        from mamrl.meta_exploration_rollout import run_meta_trial
        trial=run_meta_trial(self.rule(),self.factory,seed=51,steps=12,opponent_profile='rapid')
        q={}
        def key(obs):
            return tuple(obs[k] for k in ('previous_own_action','previous_opponent_action',
                                         'previous_own_reward','episode_start_flag'))
        nonuniform=0
        for event in trial['exploration']+trial['assessment']:
            state=key(event['observations'][1])
            values=q.get(state,[0.0]*3)
            best=max(values);ties=sum(v==best for v in values)
            expected=[.25/3+(.75/ties if v==best else 0) for v in values]
            for actual,value in zip(event['opponent_action_probabilities'],expected):
                self.assertAlmostEqual(actual,value)
            nonuniform+=sum(abs(p-1/3) for p in expected)>1e-12
            next_values=q.get(key(event['learning_next_observations'][1]),[0.0]*3)
            target=event['rewards'][1]+(.95*max(next_values) if not event['terminal'] else 0)
            action=event['actions'][1]
            changed=list(values)
            changed[action]+=.7*(target-values[action])
            q[state]=changed
        self.assertGreater(nonuniform,0)
        self.assertEqual(nonuniform,trial['opponent_nonuniform_action_events'])
        self.assertEqual(sum(trial['phase_steps'].values()),trial['environment_steps_with_control'])

    def test_simultaneous_actions_rewards_observations_and_boundaries(self):
        from mamrl.meta_exploration_rollout import run_meta_trial

        inputs=[]
        def teacher_factory(**kwargs):
            self.assertEqual(set(kwargs), {'n', 'seed', 'student'})
            teacher = self.factory(**kwargs)
            class Spy:
                def act(self, observation):
                    self.assert_fields = {field.name for field in dataclasses.fields(observation)}
                    inputs.append(self.assert_fields)
                    return teacher.act(observation)
            return Spy()

        trial = run_meta_trial(self.rule(), teacher_factory, seed=7, steps=8,
                               opponent_profile='steady')
        self.assertEqual([len(trial[k]) for k in ('exploration','assessment','erased_assessment')], [8,8,8])
        self.assertEqual(trial['environment_steps_with_control'],24)
        self.assertEqual(trial['phase_steps'], {'exploration':8,'assessment':8,'erased_assessment':8})
        self.assertEqual(trial['opponent_before_assessment'],trial['erased_opponent_before_assessment'])
        self.assertEqual(trial['student_before_assessment']['rng_digest'],
                         trial['erased_student_before_assessment']['rng_digest'])
        self.assertEqual(trial['erased_student_before_assessment']['entries'],[])
        self.assertEqual(trial['student_before_assessment']['updates'],8)
        self.assertEqual(trial['opponent_before_assessment']['updates'],8)
        self.assertEqual(trial['opponent_final']['updates'],16)
        self.assertEqual(trial['erased_opponent_final']['updates'],16)
        self.assertEqual(len(inputs), 8)
        self.assertTrue(all(fields == {'previous_own_action','previous_opponent_action',
                                       'previous_own_reward','episode_start_flag'} for fields in inputs))
        for phase in ('exploration','assessment','erased_assessment'):
            for i, step in enumerate(trial[phase]):
                observations, actions, rewards, next_obs = (step[k] for k in
                    ('observations','actions','rewards','next_observations'))
                p0,p1 = observations
                self.assertEqual(p0['previous_own_action'],p1['previous_opponent_action'])
                self.assertEqual(p0['previous_opponent_action'],p1['previous_own_action'])
                self.assertEqual(rewards, [float(actions[0] == actions[1])]*2)
                self.assertEqual(next_obs[0]['previous_own_action'],actions[0])
                self.assertEqual(next_obs[0]['previous_opponent_action'],actions[1])
                self.assertEqual(next_obs[0]['previous_own_reward'],rewards[0])
                self.assertEqual(p0['episode_start_flag'], i == 0)
                if i == 0:
                    self.assertEqual(p0['previous_own_reward'],0)
        # Same paired branch begins at the same environment state and opponent RNG.
        self.assertEqual(trial['assessment'][0]['observations'],
                         trial['erased_assessment'][0]['observations'])
        self.assertEqual(trial['assessment'][0]['actions'][1],
                         trial['erased_assessment'][0]['actions'][1])
        # Internal truncation bootstraps from the next episode's actual reset.
        self.assertFalse(trial['exploration'][-1]['terminal'])
        self.assertEqual(trial['exploration'][-1]['learning_next_observations'],
                         trial['assessment'][0]['observations'])
        self.assertTrue(trial['assessment'][-1]['terminal'])
        self.assertTrue(trial['erased_assessment'][-1]['terminal'])
        self.assertAlmostEqual(trial['exploration_mean'],
                               sum(s['rewards'][0] for s in trial['exploration']) / 8)
        self.assertAlmostEqual(trial['assessment_mean'],
                               sum(s['rewards'][0] for s in trial['assessment']) / 8)
        self.assertAlmostEqual(trial['erased_mean'],
                               sum(s['rewards'][0] for s in trial['erased_assessment']) / 8)
        self.assertAlmostEqual(trial['meta_reward'],trial['assessment_mean']-trial['erased_mean'])
        json.dumps(trial, allow_nan=False)

    def test_real_opponent_learning_and_possible_paired_divergence(self):
        from mamrl.meta_exploration_rollout import run_meta_trial

        # Independent deterministic witnesses: learned-vs-erased first action
        # diverges at seed 51; opponent continuation diverges at seed 1.
        trials=[run_meta_trial(self.rule(), self.factory, seed=s, steps=8,
                               opponent_profile='steady') for s in range(40)]
        trials.append(run_meta_trial(self.rule(), self.factory, seed=51, steps=8,
                                     opponent_profile='steady'))
        self.assertTrue(any(t['assessment'][0]['actions'][0] != t['erased_assessment'][0]['actions'][0]
                            for t in trials))
        self.assertTrue(any([s['actions'][1] for s in t['assessment']] !=
                            [s['actions'][1] for s in t['erased_assessment']] for t in trials))
        # Detect actual strategy changes directly from pre/post policy distributions,
        # not just from the fact that the update method was invoked.
        self.assertTrue(any(t['opponent_before_assessment']['policy_l1_change'] > 0 for t in trials))

    def test_determinism_and_global_rng_preservation(self):
        from mamrl.meta_exploration_rollout import run_meta_trial

        before=random.getstate()
        first=run_meta_trial(self.rule(),self.factory,seed=42,steps=4,opponent_profile='rapid')
        second=run_meta_trial(self.rule(),self.factory,seed=42,steps=4,opponent_profile='rapid')
        self.assertEqual(first,second)
        self.assertEqual(before,random.getstate())
        with self.assertRaises((TypeError,ValueError)):
            run_meta_trial(self.rule(),self.factory,seed=42,steps=0,opponent_profile='rapid')
        with self.assertRaises((TypeError,ValueError)):
            run_meta_trial(self.rule(),self.factory,seed=42,steps=4,opponent_profile='unknown')

    def test_hand_calculated_memory_one_player_local_reward_indices(self):
        from mamrl.meta_exploration_rollout import run_meta_trial

        rule=RuleSpec(3,(RuleTerm('repeat_own',2),RuleTerm('copy_opponent',-1),
                         RuleTerm('cyclic_zero_sum',1)))
        trial=run_meta_trial(rule,self.factory,seed=432,steps=12,opponent_profile='rapid')
        for phase in ('exploration','assessment','erased_assessment'):
            for step in trial[phase]:
                for player in (0,1):
                    before=step['observations'][player]
                    own=step['actions'][player]
                    other=step['actions'][1-player]
                    delta=(own-other)%3
                    expected=(2*(own==before['previous_own_action'])
                              -(own==before['previous_opponent_action'])
                              +(1 if delta==1 else (-1 if delta==2 else 0)))
                    self.assertEqual(step['rewards'][player],expected)


@unittest.skipUnless(torch is not None,'optional torch dependency')
class MetaOptimizationContract(unittest.TestCase):
    def test_full_sequence_replay_reproduces_policy_likelihood_including_n4_n5(self):
        from mamrl.action_set import ActionSetActorCritic
        from mamrl.meta_exploration import collect_exploration, replay_exploration

        for n in (3,4,5):
            with self.subTest(n=n):
                model=ActionSetActorCritic(hidden_size=4,seed=17)
                rule=RuleSpec(n,(RuleTerm('coordination',1),))
                rollout=collect_exploration(model,rule,seed=34,steps=5,opponent_profile='rapid')
                logs,values,entropies=replay_exploration(model,rollout)
                self.assertTrue(torch.allclose(logs,torch.tensor(rollout.old_log_probs),atol=1e-6,rtol=1e-6))
                self.assertTrue(torch.allclose(values,torch.tensor(rollout.old_values),atol=1e-6,rtol=1e-6))
                self.assertEqual(len(entropies),5)
                self.assertEqual(rollout.trial['n'],n)
                self.assertEqual(len(rollout.events),5)

    def test_meta_targets_ignore_exploration_rewards_and_change_actor_gradient(self):
        from mamrl.action_set import ActionSetActorCritic
        from mamrl.meta_exploration import collect_exploration, replay_exploration, meta_targets
        from mamrl.ppo import ppo_loss

        model=ActionSetActorCritic(hidden_size=4,seed=5)
        rule=RuleSpec(3,(RuleTerm('coordination',1),))
        rollout=collect_exploration(model,rule,seed=12,steps=5,opponent_profile='steady')
        # Keep the same history, old likelihoods and downstream reward while changing
        # every recorded exploration reward. Target must remain identical.
        modified_trial=dict(rollout.trial)
        modified_trial['exploration']=[{**step,'rewards':[1000.0,-1000.0]}
                                       for step in rollout.trial['exploration']]
        changed=dataclasses.replace(rollout,trial=modified_trial)
        self.assertTrue(torch.equal(meta_targets([rollout]),meta_targets([changed])))
        pos=dataclasses.replace(rollout,meta_reward=1.0)
        neg=dataclasses.replace(rollout,meta_reward=-1.0)
        self.assertEqual(meta_targets([pos]).tolist(),[1.0]*5)
        self.assertEqual(meta_targets([neg]).tolist(),[-1.0]*5)

        def actor_gradient(candidate):
            logs,values,entropy=replay_exploration(model,candidate)
            targets=meta_targets([candidate])
            loss,_=ppo_loss(logs,torch.tensor(candidate.old_log_probs),values,targets,targets,entropy,
                            clip_ratio=.2,value_coefficient=0,entropy_coefficient=0)
            return torch.autograd.grad(loss,model.actor[2].weight)[0]
        positive,negative=actor_gradient(pos),actor_gradient(neg)
        self.assertGreater(float(positive.abs().max()),1e-8)
        self.assertTrue(torch.allclose(positive,-negative,atol=1e-7,rtol=1e-6))

    def test_tiny_cli_checkpoint_reproduces_trials_and_no_overwrite(self):
        from mamrl.meta_exploration import evaluate_meta,meta_tasks,neural_factory
        from mamrl.recurrent import load_checkpoint

        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)/'experiment'
            cmd=[sys.executable,'-B','-m','mamrl.meta_exploration','--output-dir',str(directory),
                 '--seeds','7','--updates','1','--trials-per-update','1','--steps','2',
                 '--hidden-size','4','--reps','1']
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
            completed=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=180)
            self.assertEqual(completed.returncode,0,completed.stderr[-5000:])
            from mamrl.experiment_artifacts import load_meta_experiment
            record=load_meta_experiment(directory/'experiment.json')
            protocol=json.loads((directory/'protocol.json').read_text())
            manifest=json.loads((directory/'manifest.json').read_text())
            self.assertEqual(len(record['runs']),1)
            self.assertEqual(protocol['inner_training_gameplay_steps_per_seed'],4)
            self.assertEqual(protocol['inner_training_control_steps_per_seed'],2)
            self.assertEqual(protocol['teacher_training_decisions_per_seed'],2)
            for filename,digest in protocol['source_sha256'].items():
                self.assertEqual(hashlib.sha256((directory/'sources'/filename).read_bytes()).hexdigest(),digest)
            row=record['runs'][0]
            self.assertEqual(json.loads((directory/'seed7'/'evaluation.json').read_text()),row)
            model=load_checkpoint(directory/row['checkpoint'])
            tasks=meta_tasks(manifest,'fresh_rules')
            actual=evaluate_meta(neural_factory(model),tasks,steps=2,reps=1)
            self.assertEqual(actual,row['evaluation']['trained']['fresh_rules'])
            # Every algorithm is evaluated from the same per-task random seed.
            reference=record['baselines']['random']['fresh_rules']['trials']
            compare=row['evaluation']['trained']['fresh_rules']['trials']
            self.assertEqual([(t['task_id'],t['rep'],t['seed']) for t in reference],
                             [(t['task_id'],t['rep'],t['seed']) for t in compare])
            self.assertEqual({t['opponent_profile'] for t in actual['trials']},{'steady','rapid'})
            self.assertGreater(sum(t['opponent_final']['policy_changes'] for t in actual['trials']),0)
            self.assertIn('manifest.json',(directory/'report.html').read_text())
            before=(directory/'protocol.json').read_bytes()
            failed=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=180)
            self.assertNotEqual(failed.returncode,0)
            self.assertEqual(before,(directory/'protocol.json').read_bytes())


if __name__ == '__main__':
    unittest.main()
