# Absolute-return objectives for learning to explore

Prospective follow-up using fresh games and new initializations. The preceding D-031 and active-probe panels are open. The question is whether the action chosen for experience yields high absolute performance on a new game and whether that action's usefulness depends on its feedback.

## Design

Use the real three-action, depth-three structured games and four equally weighted motifs. In each game, start a fresh Q-learning opponent and collect four shared uniformly random calibration interactions. Then assign one opening move and one terminal response. All three opening actions are available. Training labels enumerate openings and integrate terminal opponent probabilities. The policies only see joint actions, focal rewards available in their channel, and public state/history. Hidden rules, private state, opponent Q and simulator action values are excluded from all inputs. The carried-state evaluation preserves the new game and opponent state after the opening.

In each block, train one terminal reward predictor on fresh terminal-training games with paired visible and masked versions of the numerical opening reward. Reuse the existing active_probe two-output predictor for the live state and a restored-state diagnostic. The shared terminal predictor receives the opening's joint action and public state in both channels. It has separate outputs for live and rewind. Rewind restores the physical game and Q state to the pre-opening state while the agent retains its opening transcript. The original scored first reward remains the same. This setup distinguishes the effect of contingent feedback from preparing a favorable state or making the opponent learn.

Freeze the terminal predictor. On disjoint opening-training games learn four opening predictors with identical architecture (two 64-unit SiLU layers), random initialization and minibatch order (one paired seed per block), observables, number of samples and optimizer steps. Fit per-action targets by mean squared error with the same target-standardization procedure. Apply the resulting argmax to opening observations. The score definitions are:

| Policy | Training target for a candidate opening |
|---|---|
| return | Expected first reward + the expected live reward of the visible terminal choice |
| terminal | Expected live reward of the visible terminal choice |
| gain | Expected live reward of the visible choice minus that of the masked choice, against the same post-opening state and opponent |
| myopic | Expected first reward |

The first term uses the exact current opponent mixture. The continuation label uses a paired sampled first opponent move and gate, integrating the terminal mixture. Optimizing the undiscounted sum avoids rewarding a lower masked score. Gain-only resembles a retained-minus-erased objective, but masks the latest numerical reward in a learned decoder instead of clearing D-031's Q table. It can be increased by either raising visible payoff or lowering masked payoff after a chosen opening. Prefix rewards are common to all methods and cancel from all return contrasts. Expected terminal reward is the absolute terminal performance at the same fixed experience budget. Also report opening plus terminal reward with actual first reward and exact expected second. This study uses supervised optimization on simulator branches. Its horizon comprises two learned decisions.

## Paired tests

All evaluated openers use the same held-out tasks, resets, calibration histories, first opponent/gate random variables, terminal responder and reward conventions. The observed history of one game may point to a different probe than that of another. Test this directly by applying the return scorer to a donor history from another game within the same motif. The recipient's world, rewards and terminal policy inputs remain its own. Donor games form a deterministic one-game cyclic shift within each motif, preserving reset slots and the marginal opener action distribution. Report random-opening expectations by integrating all three moves with the common responder.

Prespecified two-sided primary contrasts:

1. Live total return difference, cumulative-return opener minus gain-only opener.
2. Live total return difference, cumulative-return opener minus myopic opener.
3. In rewind, the difference in terminal reward visibility effects of the cumulative-return and myopic openers. Use their original live-trained choice. Within each opening compute visible minus masked payoff; both methods face the exact same restored physical and opponent state. This tests whether the chosen probe improves use of its reward signal relative to a myopic choice.

Report all cell means and a decomposition into actual initial reward, visible terminal reward, masked terminal reward, and feedback-dependent terminal gain. The matched minus donor-history contrast, terminal-only objective, random openings and per-motif outcomes are descriptive. A positive primary 1 alone shows a better utility target. A positive primary 2 additionally shows value over optimizing immediate reward. Evidence for specifically selecting reward-informative experience requires primary 3; compare the donor-history diagnostic for context dependence. None of the objectives requires treating predictive gain as an intrinsic reward. The diagnostic tests use of the reward signal by a fixed responder, with no claim that the agent identifies a human-interpretable variable.

## Scale and inference

Twenty-four independent blocks. Each draws 512 terminal-training games × 4 resets, 768 opening-training games × 4 resets, and 256 held-out games × 2 resets. Terminal optimization 800 steps × minibatch 384. Openers 600 × 256. AdamW .002 and weight decay .0001, with scalar-target normalization computed separately on the training set. Joint source configuration, checkpoints, branching data, trained scores and analysis are saved. Both train splits receive the identical count of games for every opener and test portfolios. Hold out serialized game fingerprints across blocks and splits. Report exact fingerprint disjointness relative to accessible prior panels separately.

Development seed 8420129. Final seed 29380741. Each contrast averages over held-out games and resets within each block. Bootstrap the 24 independent block vectors with 20,000 paired draws from NumPy seed 912853. Use per-contrast 98.3333% equal-tailed intervals for three prespecified contrasts, giving nominal familywise 95% coverage. Descriptive summaries get no familywise claim. No outcome-dependent extension or method selection.
