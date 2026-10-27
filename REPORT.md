# Assignment 2 Report

> **How to use this draft.** The tables and facts below come straight from the
> pipeline's outputs (`results/`) and are correct for the committed run. Every
> line marked ✍️ is yours to write in your own words, then delete the ✍️
> prompt. The cost figures are still the placeholders in `assumptions.json`:
> once you set your own and run `uv run python decision.py`, the tables in the
> last three sections change and need updating. Delete this box when done.

- **Name**:
- **Student ID**:
- **Email**:
- **Group**: [BBADBA 5A | BBADBA 5B | PPLEDBA 5A | BDBA 3A]

## Business problem, and the cost of being wrong

**Decision.** On winter nights, a road-maintenance control room decides for
each stretch of road whether to send a salt/grit truck. The model looks at a
dashcam image of the stretch and gives one of three outcomes: **pass** (do
nothing), **operator review** (a person looks at the image and decides), or
**send a truck**.

- **One case** = one image of one road segment, checked once per 2-hour cycle.
- **Positive class** = the road *needs treatment*: the image shows **Snow** or
  **Slush**. Negative = **Clear** or **Wet**.
- **False negative**: a snowy road is left untreated.
- **False positive**: a truck is sent to a road that did not need it.

✍️ Who acts on the decision today, and how they make it.
✍️ What a missed snowy road costs and why; what an unnecessary truck run costs
and why; why the first is worse and roughly by how much.
✍️ Why Wet counts as "no treatment needed" in your framing.

## Dataset

- **Roboflow Universe:** workspace `udot-classification`, project
  `udot-winter-road-conditions`, **version 1**, used as published (not forked).
- **Link:** https://universe.roboflow.com/udot-classification/udot-winter-road-conditions/dataset/1
- **Licence:** CC BY 4.0. The example images in `results/examples/` come from
  this dataset.
- **Task:** image classification, 4 classes: Clear, Slush, Snow, Wet.
- **One image:** a frame from a vehicle dashcam driving on winter roads,
  with the car's hood visible at the bottom. 5,232 images are 3840×2160 video
  frames and 1,013 are 640×640 with black letterbox bars.

✍️ Why this dataset serves the decision rather than just resembling it, and
the one way it differs from your scenario (dashcam frames, not fixed cameras).

## What the dataset's composition implies

| Class | Train | Valid | Test | Total | Share taken at night |
|---|---|---|---|---|---|
| Clear | 410 | 117 | 52 | 579 | 24.9% |
| Slush | 762 | 203 | 107 | 1,072 | 9.7% |
| Snow | 2,289 | 678 | 344 | 3,311 | 53.4% |
| Wet | 902 | 263 | 118 | 1,283 | 11.6% |
| **Total** | **4,363** | **1,261** | **621** | **6,245** | |

- Split: 69.9% train / 20.2% valid / 9.9% test, no augmentation applied.
- Positive (Snow + Slush) share: 69.9% of train, 72.6% of test.
- Object size: not applicable, as this is classification and the whole image
  is the case.
- **Night images:** "night" = mean grey level below 45 out of 255. Over half
  of the Snow images are at night, against 10–25% for the other classes.
- **Near-identical frames across splits:** many images are consecutive video
  frames. Comparing a 256-bit fingerprint of every image (a test image counts
  as a twin if it differs from some training image in at most 25 bits), 22.8%
  of validation images and 21.1% of test images (131 of 621) have a
  near-identical twin in train.

✍️ What these imply: Clear has only 52 test images (how much can you trust
its numbers?); night images could let the model learn "dark = snow"; twins
could inflate test accuracy. Say how you checked each one (see the subsets in
the next section).
✍️ How the test split's 72.6% snow differs from real winter nights, and why
that is the reason for the base rate in the costing section.

## Finetuning and baselines

**Model:** `yolo11n-cls` (1.5M parameters), finetuned from its
ImageNet-pretrained checkpoint.

**Where and how it was trained:** a 4-core Intel Xeon CPU (2.10 GHz) cloud
machine, PyTorch 2.11 (CPU build), Ultralytics 8.4.173, Python 3.12. 30
epochs, image size 224, batch 64, seed 0, otherwise Ultralytics defaults
(including its default augmentation). Training took 47 minutes; the full
pipeline took 53. Before training, every image was shrunk to 640 px on its
longest side (labels and splits untouched) because the model only sees 224 px.
The best epoch on validation was epoch 28 (validation top-1 accuracy 0.890).

**Results on the same test split (621 images):**

| Approach | 4-class accuracy | "Needs treatment" precision | "Needs treatment" recall |
|---|---|---|---|
| Finetuned `yolo11n-cls` (threshold 0.51) | **0.907** | **0.967** | **0.978** |
| Majority class: always "Snow" (= always send a truck) | 0.554 | 0.726 | 1.000 |
| Predict nothing (= never send a truck) | – | – | 0.000 |
| Pretrained model, zero-shot (threshold 0.01) | – | 0.750 | 0.911 |

The pretrained model only knows the 1,000 ImageNet labels, so it cannot output
Clear/Slush/Snow/Wet. Its "snow score" is the total probability it gives to
snow-related labels (alp, snowplow, snowmobile, ski, dogsled, bobsled,
shovel). Its most common top labels on the test images were alp (74), snowplow
(70), snowmobile (63), traffic light (60) and car mirror (50).

**Same finetuned model on harder subsets of the test split:**

| Test subset | Images | Accuracy |
|---|---|---|
| All | 621 | 0.907 |
| No near-identical twin in train | 490 | 0.892 |
| Daytime | 388 | 0.889 |
| Night | 233 | 0.936 |

✍️ Why classification and `yolo11n-cls` fit this decision.
✍️ Whether the gap to the baselines is what you expected, and what the subsets
tell you about the leakage and night worries from the previous section.

## Per-class evaluation and failure modes

**Positive class:** "needs treatment" = Snow or Slush.

| Class | Test images | Precision | Recall |
|---|---|---|---|
| Clear | 52 | 0.788 | 0.788 |
| Slush | 107 | 0.812 | 0.850 |
| Snow | 344 | 0.971 | 0.971 |
| Wet | 118 | 0.858 | 0.822 |

**Confusion matrix on test** (rows = actual, columns = predicted):

| Actual \ Predicted | Clear | Slush | Snow | Wet |
|---|---|---|---|---|
| Clear | **41** | 2 | 2 | 7 |
| Slush | 0 | **91** | 7 | 9 |
| Snow | 1 | 9 | **334** | 0 |
| Wet | 10 | 10 | 1 | **97** |

Collapsed to the business decision at threshold 0.51 (test images):

| | Truck sent | No truck |
|---|---|---|
| Actually needs treatment (451) | 441 (TP) | 10 (FN) |
| Actually fine (170) | 15 (FP) | 155 (TN) |

✍️ Which class carries the headline accuracy, which one is dragging, and what
the model confuses with what (Clear ↔ Wet, Wet → Slush, Slush → Wet).
✍️ Which of those confusions actually matter for the decision: a Slush
image called Snow still gets a truck; a Slush image called Wet does not.

## Cost matrix and operating point

**Current figures (`assumptions.json`, placeholders to replace):**

| Item | Value |
|---|---|
| Truck sent, road needed it (TP) | €80 |
| Truck sent, road was fine (FP) | €80 |
| Snowy road left untreated (FN) | €1,000 |
| Nothing done, road fine (TN) | €0 |
| Operator review | 15 s at €25/h = €0.10 per image |
| Base rate: share of images that really need treatment | 15% |
| Images per day | 10,000 |

✍️ Where each figure comes from and why. "I assumed X because Y."

**How the expected cost is computed:** on the validation split, measure for
each threshold what share of snowy images and what share of fine images get a
truck. Weight those shares by the base rate (15%), not by the dataset's 70%
snow, and multiply by the costs to get euros per 1,000 real images. Thresholds
from 0.00 to 1.00 in steps of 0.01 are tried, and the cheapest one is chosen.
It is chosen on validation and reported once on test.

![Cost against threshold](results/cost_curve.png)

- **Chosen threshold (model decides alone): 0.51.** Expected cost on test:
  €21,060 per 1,000 images.
- **Cross-check:** a truck is worth sending when P(snowy) × €1,000 >
  €80, i.e. P(snowy) > 0.08. The model was trained on 70% snow, so its
  score overstates real-world P(snowy). Converting 0.08 into the model's
  score scale gives **0.53**, close to the 0.51 the sweep found.

**Sensitivity** (one figure changed at a time; threshold re-chosen on
validation, priced on test; full table in `results/sensitivity.csv`):

| Changed | Value | Best threshold | € per 1,000 (model alone) |
|---|---|---|---|
| Missed snowy road | €250 | 0.90 | 15,814 |
| Missed snowy road | €500 | 0.53 | 19,397 |
| Missed snowy road | **€1,000 (current)** | **0.51** | **21,060** |
| Missed snowy road | €2,000 | 0.43 | 24,786 |
| Missed snowy road | €4,000 | 0.05 | 29,304 |
| Truck run | €40 | 0.43 | 12,393 |
| Truck run | €320 | 0.90 | 63,254 |
| Base rate | 5% | 0.53 | 11,726 |
| Base rate | 25% | 0.43 | 30,747 |
| Base rate | 40% | 0.05 | 44,110 |

✍️ The assumption that would most change your answer, and the value at which
it flips ("the answer changes if X drops below Z").

## What a person still has to check

**Thresholds (chosen on validation within the operators' capacity):** pass
below **0.03**, send a truck from **1.00**, operator review in between.
Capacity = 2 operators × 8 h × 240 images/h = 3,840 reviews a day.

| On the test images | Pass | Operator review | Send a truck |
|---|---|---|---|
| Actually fine (170) | 129 | 41 | 0 |
| Actually needs treatment (451) | 1 | 421 | 29 |

At the 15% base rate, **34.5%** of images go to an operator: **3,450 a day**,
15 s each, **1.8 operators** of 8 hours. That is within the 2 available.

**Priced on the test split, per 1,000 images:**

| Policy | € per 1,000 | Reviewed | Operators | Missed snowy /1,000 | Needless trucks /1,000 |
|---|---|---|---|---|---|
| Perfect model (reference) | 12,000 | 0% | 0 | 0 | 0 |
| Do nothing | 150,000 | 0% | 0 | 150 | 0 |
| Send a truck everywhere | 80,000 | 0% | 0 | 0 | 850 |
| Operator checks every image | 12,104 | 100% | 5.2 | 0 | 0 |
| Pretrained model, zero-shot, alone | 79,039 | 0% | 0 | 13.3 | 685 |
| Finetuned model alone | 21,060 | 0% | 0 | 3.3 | 75 |
| **Finetuned model + operator review** | **12,342** | 34.5% | 1.8 | 0.3 | 0 |

✍️ Is 3,450 reviews a day with 2 operators realistic? What happens on a heavy
snow night?
✍️ Compare your design with the three alternatives (do nothing, a person
checks everything, model alone), and say why "a person checks everything" is
not the answer even though it is slightly cheaper per image.
✍️ Why the operator ends up confirming almost every truck (a review costs
€0.10, a wrong truck €80), and whether that changes with your own figures.

## Limitations & next steps

✍️ Real limitations, each with a concrete fix. Candidates from the results:
dashcam frames rather than fixed cameras; one region and one season; near-twin
frames across splits; only 52 Clear test images; operator assumed always
right; one image treated as independent of the segment's previous checks.

## Generative AI use disclosure

✍️ What AI did and what you did. Facts to include: an AI assistant (Claude)
suggested the topic options and the dataset, wrote the pipeline and dashboard
code, and executed the training run; say which decisions and analysis are yours.
