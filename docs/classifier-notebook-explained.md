# `notebooks/classifier.ipynb`: what this notebook does

This notebook is not used to train the generator itself. Its purpose is independent evaluation.
It trains a baseline external multi-label AMP classifier that is separate from the cVAE and then uses it as a proxy oracle to check whether generated peptides actually satisfy the requested activity conditions.

## Why it is needed

`notebooks/cvae.ipynb` already reports the basic generative metrics:

- validity
- uniqueness
- novelty

However, these metrics are not enough to support the claim that "the model can truly generate peptides with the requested biological activity."

`notebooks/classifier.ipynb` closes that gap:

1. it trains an external sequence-only classifier on the same AMP dataset;
2. it tunes decision thresholds on the validation set;
3. it loads a cVAE checkpoint;
4. it generates peptides under different conditions;
5. it runs those peptides through the external classifier;
6. it computes condition-adherence metrics.

## Notebook logic by section

### 1. Data loading

The notebook reads the same processed CSV as the cVAE pipeline and uses:

- the amino acid sequence column;
- sequence length;
- six binary target labels:
  - `is_anti_gram_positive`
  - `is_anti_gram_negative`
  - `is_antifungal`
  - `is_antiviral`
  - `is_antiparasitic`
  - `is_anticancer`

### 2. Vocabulary and tokenization

The classifier uses its own vocabulary pipeline, but it reuses `models/vocab.pkl` whenever possible to stay aligned with the cVAE in terms of alphabet and `max_len`.

The standard special tokens are used:

- `<PAD>`
- `<UNK>`
- `<SOS>`
- `<EOS>`

### 3. Data split

The current notebook uses a deterministic random split:

- 80% train
- 10% validation
- 10% test

The notebook also states explicitly that a publishable setup should replace this with a homology-aware split.

### 4. External classifier

The classifier architecture is:

- embedding
- BiLSTM encoder
- masked mean pooling
- masked max pooling
- MLP classification head

This is a separate discriminative model and does not share weights with the generator.

### 5. Classifier training

During training, the notebook:

- handles class imbalance via `pos_weight`;
- optimizes `BCEWithLogitsLoss`;
- logs train and validation dynamics;
- saves the best checkpoint to `models/best_external_amp_classifier.pt`.

### 6. Classification metrics

The notebook computes standard multi-label classification metrics:

- per-label AUROC
- per-label AUPRC
- per-label F1 / precision / recall
- per-label MCC
- macro AUROC
- macro AUPRC
- macro F1
- macro precision
- macro recall
- macro MCC
- macro balanced accuracy
- micro F1

According to the saved artifacts in `models/external_amp_classifier_artifacts.json`, the current test summary is:

- default thresholds:
  - `macro_auroc = 0.8203`
  - `macro_auprc = 0.6414`
  - `macro_f1 = 0.6144`
  - `macro_mcc = 0.4305`
  - `macro_balanced_acc = 0.7256`
  - `micro_f1 = 0.8347`
- tuned thresholds:
  - `macro_auroc = 0.8203`
  - `macro_auprc = 0.6414`
  - `macro_f1 = 0.6288`
  - `macro_mcc = 0.4244`
  - `micro_f1 = 0.8371`

This means the notebook already provides more than "some extra metrics": it establishes a baseline external evaluation setup, not a final oracle.

### 7. Threshold tuning

The threshold is not fixed to `0.5`.
The notebook tunes label-specific thresholds on the validation set and stores them in the artifacts.

This is important because for downstream condition-adherence evaluation, a single shared threshold is usually too crude.

### 8. Connecting the generator

After the classifier section, the notebook reuses the cVAE architecture and loads:

- `models/best_vae.pt`
- or `models/best_cvae.pt`

From that point, the generator is used only for inference:

- sampling from latent space;
- decoding with a condition vector;
- batch generation for different activity profiles.

### 9. Evaluating generation with the classifier

This is the main point of the notebook.

For each condition, the notebook:

1. generates a batch of sequences;
2. predicts probabilities for all 7 activities;
3. aggregates these probabilities into generation metrics.

It reports:

- `all_requested_hit_rate`
- `any_requested_hit_rate`
- `exact_match_rate`
- `off_target_rate`
- `mean_requested_prob`
- `mean_offtarget_prob`
- `macro_precision_vs_request`
- `macro_recall_vs_request`
- `macro_f1_vs_request`

So the notebook already moves beyond string-level similarity and asks whether the requested condition is actually satisfied.

### 10. Ranking candidates

A separate block ranks generated peptides so that the top candidates are those with:

- high probabilities for the requested labels;
- low probabilities for off-target activity;
- the best balance between requested and unwanted activity.

This is useful for post hoc reranking and manual candidate selection.

### 11. Artifacts

The notebook saves:

- `models/best_external_amp_classifier.pt`
- `models/external_amp_classifier_artifacts.json`
- `models/condition_adherence_metrics.csv`

From the current `models/condition_adherence_metrics.csv`, it is already clear that conditional control is uneven:

- there are strong single-label results, for example for `Gram-` and `Gram+`;
- compositional conditions work worse;
- one especially weak case is `Antifungal + Anticancer`, where `all_requested_hit_rate = 0.22`.

This supports the current interpretation: this is a baseline evaluation pipeline, and the next step is to make it stricter and tie it to an external predictor API.

## Interpretation of the training plots

The training plots show that the baseline external multi-label classifier already performs reasonably well, but it is trained for too many epochs and starts to overfit.

### 1. Loss curves

In the upper-left panel:

- train loss decreases monotonically, roughly from `0.93` to `0.17`;
- validation loss decreases at first, reaching about `0.70-0.72`;
- after roughly epoch `7-9`, validation loss starts to rise consistently and reaches almost `1.9` by the end.

This means:

- the model keeps fitting the training set very well;
- after the early epochs it starts to generalize worse on the validation set;
- this is a textbook overfitting pattern.

Main takeaway:

- the best checkpoint is not the final epoch;
- the model should be selected around epoch `6-9`, where validation loss is lowest.

So the last epoch is not the best one, even if the train loss still looks good.

### 2. Validation metrics

In the upper-right panel:

- macro AUROC rises to about `0.85`;
- micro F1 rises to about `0.84-0.85`;
- macro F1 rises to about `0.64-0.65`.

#### What macro AUROC around 0.85 means

This is a good result.
It means that, on average across labels, the classifier ranks positive examples above negative ones quite well.

Importantly, AUROC does not say "how good the classifier is at threshold 0.5." It measures how well the model separates classes by score overall.

A practical interpretation is:

- `0.5`: near-random ranking
- `0.7-0.8`: acceptable
- `0.8-0.9`: good
- `> 0.9`: very strong

In this case:

- the classifier is already reasonably strong as a baseline external evaluator;
- it can be used for condition-adherence analysis.

#### Why micro F1 is high while macro F1 is noticeably lower

This is important.

- micro F1 pools all label decisions together;
- macro F1 averages F1 equally across labels.

If `micro F1 >> macro F1`, that usually means:

- the dataset is class-imbalanced;
- frequent classes are predicted well;
- rare classes are handled substantially worse.

So the model is decent overall across all predictions, but not all labels are covered equally well.

Main implication:

- performance is solid for frequent labels;
- weaker performance is likely concentrated in rarer labels.

### 3. Per-label AUROC on the test set

The lower-left panel is approximately:

- `is_anti_gram_negative`: about `0.87`
- `is_anti_gram_positive`: about `0.86`
- `is_antiparasitic`: about `0.86`
- `is_antifungal`: about `0.79`
- `is_antiviral`: about `0.77`
- `is_anticancer`: about `0.68`

How to read this:

#### Strong classes

- antibacterial
- gram-negative
- gram-positive
- antiparasitic

For these labels, the classifier can be used with relatively high confidence as a baseline external evaluator of generation quality.

#### Moderate classes

- antifungal
- antiviral

The model is already useful for these labels, but conclusions should be made more carefully.

#### Weak class

- anticancer at about `0.68`

This is not catastrophic, but clearly weaker than the others.
For this class, the classifier is a noisy oracle:

- it will make more mistakes;
- anticancer condition-adherence estimates will be less reliable.

Practical takeaway:

- the generator can already be evaluated confidently for antibacterial, gram-positive, gram-negative, and antiparasitic activity;
- antifungal and antiviral should be treated more cautiously;
- anticancer should be treated much more cautiously.

### 4. Per-label F1 before vs after threshold tuning

The lower-right panel compares:

- the default threshold, usually `0.5`;
- a tuned, label-specific threshold.

This is the correct comparison to include, because in multi-label classification a universal threshold of `0.5` is almost never optimal.

What is visible:

#### Antibacterial

F1 improves further, almost to `0.97-0.98`.
This indicates a very strong class.

#### Gram-positive and Gram-negative

These also remain very strong, around `0.88-0.91`.
They are stable and reliable labels.

#### Antiparasitic

There is a noticeable increase after threshold tuning, roughly from `0.49` to `0.60`.
This suggests the model had useful signal, but the default threshold of `0.5` was suboptimal.

#### Antifungal

There is little to no gain, visually around `0.57-0.60`.
This suggests the limitation is not only the threshold, but also the intrinsic separability of the class.

#### Anticancer

F1 stays low, around `0.22`.
So this label is still predicted poorly as a hard binary decision.

#### Antiviral

F1 also remains very low, around `0.16-0.18`.
This is interesting because AUROC for antiviral is around `0.77`, yet F1 is poor.

That usually means:

- the scores are not ranked terribly;
- but choosing a concrete threshold still does not isolate positives well;
- the class may be strongly imbalanced or poorly calibrated.

## The main meaning of the figure

### 1. The model is already useful as a baseline ranking-based evaluator

By AUROC, the classifier is already strong enough for several labels.
That means it can already be used for:

- `mean_requested_prob`
- condition adherence
- reranking generated peptides
- comparing generated vs real peptides by predicted activity profiles

### 2. It is not equally reliable as a hard binary classifier

The F1 scores show that not all labels are equally strong.

The clearest problems are:

- anticancer
- antiviral

So:

- not all predicted labels should be trusted equally;
- labels should be grouped into strong / medium / weak categories.

### 3. The training procedure is overfitting

The loss curves show that training continues for too long.

This means the notebook should:

- enable early stopping based on validation loss;
- keep the best checkpoint rather than the final checkpoint;
- potentially use stronger regularization.

## What this means for the AMP project

If the goal is to use this classifier as an external evaluator for the generator, then the current setup is already useful as a baseline.

### Already reliable enough

For the stronger labels:

- antibacterial
- gram-positive
- gram-negative
- antiparasitic

it is reasonable to report:

- hit rate
- requested-label probability
- off-target rate
- exact match for these labels

### Use with caution

- antifungal
- antiviral

### Current weak point

- anticancer

For anticancer, strong biological claims should be avoided at this stage.

## How to interpret the metrics in plain language

### AUROC

"How well the model separates positives from negatives overall, without fixing a single threshold."

### F1

"How well the model makes an actual 0/1 decision at a chosen threshold."

### Macro F1

"Average quality across classes when every class is treated as equally important."

### Micro F1

"Overall quality across all label decisions together, where frequent classes contribute more."

### Threshold tuning

"Choosing a separate decision threshold for each class to improve binary predictions."

## Concrete conclusion for the current result

A concise way to summarize the current result is:

> The baseline external AMP classifier demonstrates strong ranking performance for antibacterial and Gram-specific labels, moderate performance for antifungal and antiviral labels, and weak performance for anticancer. Threshold tuning improves binary classification for several labels, especially antiparasitic, but class imbalance and label difficulty remain visible through the gap between micro- and macro-F1. This should be treated as a baseline result rather than a final evaluation model. Validation loss indicates substantial overfitting after the early epochs, so best-checkpoint selection and stronger regularization are necessary.

In plain terms:

- the baseline is already workable;
- it is already useful for condition-adherence evaluation;
- it should be presented explicitly as a baseline, not as a definitive oracle;
- it should not be trusted equally across all labels;
- and training should stop earlier.

## What the notebook already shows well

- There is an independent evaluator, not only internal generative metrics.
- There is a multi-label classification baseline.
- There is threshold tuning.
- There is a direct bridge between cVAE generation and downstream activity prediction.
- Condition adherence can already be measured, not just validity or novelty.

## What it still does not cover

The notebook still has methodological limitations:

- random split instead of homology-aware split;
- no independent non-AMP negative background;
- no external API or external database oracle;
- no toxicity, hemolysis, or stability predictors;
- no calibration analysis;
- novelty is not measured via sequence-identity search against external AMP databases.

## Why a new notebook with an external classifier API is needed

The new notebook in `notebooks/` should have a different focus:

- not to train another local classifier;
- but to use an external AMP classifier API as an independent evaluator;
- to evaluate cVAE generation through an API-based oracle;
- to compute the same generation-side metrics without tying the evaluation to a local discriminative model from this repository.

That is the natural next step after the current `classifier.ipynb`, and it moves the evaluation setup closer to a publishable framing.
