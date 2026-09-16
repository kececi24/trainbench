# Benchmarking the Learning Efficiency of LLM Fine-Tuning

## 1. Core Research Idea

### Main question

How do **full fine-tuning and parameter-efficient fine-tuning methods differ in actual learning efficiency**, rather than merely final downstream performance?

The project should distinguish between:

- **how fast training runs**
- **how quickly the model learns**
- **how much data the model needs**
- **how much computation is required**
- **how much memory is required**
- **how many parameters need to change**
- **what final capability is achieved**

This gives a much richer comparison than simply reporting:

> Accuracy after 3 epochs.

---

# 2. Main Experimental Comparison

Start with four principal methods:

### A. Full Fine-Tuning — FFT

Train every parameter.

This acts as the main reference point.

---

### B. LoRA

Freeze the pretrained model and learn low-rank updates.

Suggested initial configuration:

- rank \(r = 16\)
- alpha = 32
- dropout = 0.05
- target:
  - q_proj
  - k_proj
  - v_proj
  - o_proj

An expanded experiment can additionally adapt MLP layers.

---

### C. QLoRA

Quantize the backbone to 4-bit and train LoRA adapters.

Treat this separately from LoRA because QLoRA primarily changes the **memory/computation environment**, rather than representing an entirely different adaptation mechanism.

Use:

- NF4
- BF16 computation if hardware permits
- double quantization if supported
- same LoRA configuration as the corresponding LoRA experiment

Keeping the adapter configuration identical makes the comparison much cleaner.

---

### D. DoRA

Include DoRA as a stronger PEFT comparison.

DoRA separates the magnitude and directional components of the weight update and can particularly improve low-rank adaptation performance.

---

# 3. Optional Expanded PEFT Comparison

Once the basic pipeline works, add:

### AdaLoRA

Instead of assigning identical rank everywhere, dynamically allocates the available rank budget to more important matrices.

Interesting question:

> Does adaptive rank allocation improve learning efficiency per trainable parameter?

---

### IA³

Instead of learning low-rank matrices, learn scaling vectors for Transformer activations.

IA³ has an extremely small parameter footprint.

Interesting question:

> How much performance can be obtained per trainable parameter?

---

### Possible later methods

Only add these after the main experiments work:

- LoHa
- LoKr
- OFT
- BOFT
- VeRA
- PiSSA
- rsLoRA
- other recent LoRA variants

Do not start with ten PEFT algorithms.

A controlled comparison between:

\[
\text{FFT, LoRA, QLoRA, DoRA}
\]

is substantially better than an uncontrolled comparison of fifteen methods.

---

# 4. Phase I — Select the Model

The first benchmark should use a relatively small decoder-only LLM.

Ideal size:

\[
\boxed{0.5B-3B\ parameters}
\]

Reasons:

- FFT remains computationally feasible.
- multiple seeds become possible.
- many checkpoints can be evaluated.
- hyperparameter experiments become affordable.
- learning curves can be densely sampled.

Avoid beginning with a 7B–70B model.

The purpose is not to demonstrate that LoRA uses less memory—that is already known.

The interesting part is measuring the **learning dynamics** carefully.

---

# 5. Model Scaling Experiment

Once the pipeline works, use approximately three scales:

\[
\text{small} < \text{medium} < \text{large}
\]

For example:

- ~0.5B
- ~1.5B
- ~3B

Then investigate:

> Does the relative advantage of PEFT change with model scale?

Possible outcome:

| Model size | FFT | LoRA | QLoRA |
|---|---:|---:|---:|
| 0.5B | strong | competitive | competitive |
| 1.5B | strong | closer | closer |
| 3B | expensive | highly efficient | highly efficient |

This scaling dimension makes the project much more interesting.

---

# 6. Phase II — Select Training Tasks

Do not benchmark on only one dataset.

Use tasks with different adaptation requirements.

A strong design would contain roughly three categories.

## Task A — Instruction Following

Supervised instruction tuning.

Tests whether PEFT efficiently teaches:

- response format
- instruction following
- conversational behavior
- task compliance

---

## Task B — Knowledge / Domain Adaptation

Use domain-specific text such as:

- medicine
- law
- finance
- scientific literature
- programming

Question:

> Can PEFT inject domain knowledge as effectively as full fine-tuning?

This may behave very differently from instruction tuning.

---

## Task C — Reasoning or Structured Task

Examples:

- mathematical reasoning
- logical reasoning
- code generation
- structured extraction
- classification expressed generatively

Question:

> Does learning a more substantial behavioral transformation require full parameter updates?

---

# 7. Dataset Scaling

This should be one of the central experiments.

Instead of training each method once, use multiple dataset fractions:

\[
1\%, 5\%, 10\%, 25\%, 50\%, 100\%
\]

or, preferably, fixed token budgets:

\[
1M,\quad
5M,\quad
10M,\quad
25M,\quad
50M,\quad
100M
\]

tokens.

This produces:

\[
\text{performance} = f(\text{training tokens})
\]

for each method.

---

# 8. Why Token Budgets Are Better Than Epochs

Avoid treating epochs as the primary comparison.

An epoch is dataset-dependent.

Instead report:

\[
N_{\text{tokens}}
\]

processed.

Your x-axis should frequently be:

> Training tokens seen

rather than:

> Epoch

This allows meaningful comparisons across datasets and configurations.

---

# 9. Central Benchmark: Learning Curves

For every method, record validation performance throughout training.

You want curves such as:

## Loss vs Tokens

\[
L(N)
\]

where \(N\) is training tokens consumed.

---

## Loss vs Wall-Clock Time

\[
L(t)
\]

---

## Loss vs FLOPs

\[
L(C)
\]

where \(C\) represents compute.

---

## Downstream Score vs Tokens

\[
Q(N)
\]

---

## Downstream Score vs GPU-Hours

\[
Q(H)
\]

These curves contain more information than the final checkpoint.

---

# 10. Benchmark Family I — Raw Training Throughput

Measure:

### Tokens per second

\[
TPS =
\frac{N_{\text{tokens}}}{T}
\]

Report:

- total tokens/s
- tokens/s/GPU
- sequence/s
- optimizer steps/s

---

### Effective tokens per second

Padding should ideally not count.

\[
TPS_{\text{effective}}
=
\frac{\text{non-padding tokens}}
{\text{training time}}
\]

This prevents different packing/padding strategies from artificially altering throughput.

---

# 11. Benchmark Family II — Memory Efficiency

Record:

### Peak GPU memory

\[
M_{\max}
\]

in GB.

---

### Memory per token

\[
\frac{M_{\max}}
{\text{tokens per batch}}
\]

---

### Maximum feasible batch size

Determine the largest microbatch fitting within a fixed GPU.

This is particularly useful for comparing:

- FFT
- LoRA
- QLoRA

---

# 12. Benchmark Family III — Parameter Efficiency

Record:

\[
P_{\text{train}}
\]

and:

\[
\frac{P_{\text{train}}}{P_{\text{total}}}
\]

Example:

| Method | Total params | Trainable params | Trainable % |
|---|---:|---:|---:|
| FFT | 1.5B | 1.5B | 100% |
| LoRA | 1.5B | 15M | 1% |
| IA³ | 1.5B | 1M | 0.07% |

---

# 13. Benchmark Family IV — Data Efficiency

This is much more interesting.

Define:

## Tokens-to-Target

For a target validation loss:

\[
N^*(L_t)
=
\min_N \{N:L(N)\le L_t\}
\]

Example:

| Method | Tokens required to reach loss 1.8 |
|---|---:|
| FFT | 18M |
| LoRA | 22M |
| DoRA | 19M |
| QLoRA | 23M |

Lower is better.

---

For downstream performance:

\[
N^*(Q_t)
=
\min_N\{N:Q(N)\ge Q_t\}
\]

Example:

> Tokens required to reach 60% benchmark accuracy.

This is an excellent metric.

---

# 14. Benchmark Family V — Time-to-Target

Replace tokens with wall-clock time:

\[
T^*(Q_t)
\]

Example:

| Method | Time to 60% |
|---|---:|
| FFT | 82 min |
| LoRA | 51 min |
| DoRA | 55 min |
| QLoRA | 62 min |

This captures both:

- learning efficiency
- hardware efficiency

simultaneously.

---

# 15. GPU-Hours-to-Target

For distributed experiments:

\[
H =
T_{\text{hours}}
\times
N_{\text{GPU}}
\]

Then:

\[
H^*(Q_t)
\]

This is preferable to wall-clock time if GPU counts differ.

---

# 16. Compute-to-Target

If reliable FLOP estimates are available:

\[
C^*(Q_t)
\]

= FLOPs required to reach performance \(Q_t\).

This separates algorithmic efficiency from hardware speed.

---

# 17. Your “Learned Tokens per Second” Idea

There is no straightforward quantity called a learned token because a token cannot cleanly be labeled as learned or unlearned.

But the underlying idea can be formalized.

One approach:

## Loss Improvement per Second

\[
LER_t =
-\frac{\Delta L}{\Delta t}
\]

where LER stands for Learning Efficiency Rate.

---

## Loss Improvement per Token

\[
LER_N =
-\frac{\Delta L}{\Delta N}
\]

---

## Quality Improvement per Token

\[
QER_N =
\frac{\Delta Q}{\Delta N}
\]

---

## Quality Improvement per Second

\[
QER_t =
\frac{\Delta Q}{\Delta t}
\]

These should usually be evaluated over intervals rather than single optimizer steps because training is noisy.

---

# 18. A Better Generalization: Equivalent Learned Tokens

You could create a reference-model-based metric.

Suppose FFT is the reference.

Define:

\[
N_{\text{FFT}}(Q)
\]

as the number of tokens FFT requires to achieve quality \(Q\).

For another method \(m\), at time \(t\):

\[
ELT_m(t)
=
N_{\text{FFT}}\left(Q_m(t)\right)
\]

Interpretation:

> How many FFT training tokens is the current model's learned capability equivalent to?

Then define:

\[
ELTPS_m
=
\frac{d\,ELT_m(t)}{dt}
\]

or approximately:

\[
ELTPS_m
=
\frac{\Delta ELT_m}
{\Delta t}
\]

This gives something very close to:

\[
\boxed{\text{Equivalent Learned Tokens per Second}}
\]

Conceptually:

If LoRA processes:

\[
100,000\ tokens/s
\]

but its progress corresponds to FFT learning from only:

\[
80,000\ tokens/s
\]

then:

\[
ELTPS \approx 80k
\]

If another method processes only 90k physical tokens/s but learns at the equivalent rate of 85k FFT tokens/s, it is arguably the more efficient learner.

This could become an interesting methodological contribution if defined carefully.

---

# 19. Alternative Metric: Effective Learning Throughput

A simpler metric:

\[
ELT =
TPS
\times
\eta_{\text{learning}}
\]

where:

\[
\eta_{\text{learning}}
=
\frac{\Delta Q/\Delta N}
{(\Delta Q/\Delta N)_{\text{reference}}}
\]

Thus:

\[
ELT =
\frac{\text{tokens}}{\text{s}}
\times
\frac{\text{quality gained/token}}
{\text{reference quality gained/token}}
\]

This combines:

- hardware throughput
- statistical efficiency

into one quantity.

Use cautiously because it depends on:

- target metric
- training region
- reference model

Therefore it should supplement—not replace—the raw curves.

---

# 20. Benchmark Family VI — Area Under the Learning Curve

Instead of choosing one arbitrary target:

\[
AULC =
\int_0^B Q(N)dN
\]

where \(B\) is a fixed token budget.

Normalize:

\[
NAULC =
\frac{1}{B}
\int_0^B Q(N)dN
\]

This answers:

> Across the entire training process, how good was the model on average?

A method that learns quickly but eventually plateaus may outperform another method early while losing at the final checkpoint.

AULC captures this.

---

# 21. Compute-Normalized AULC

Even more useful:

\[
AULC_C =
\frac{1}{C_{\max}}
\int_0^{C_{\max}}Q(C)dC
\]

This compares quality throughout a fixed compute budget.

---

# 22. Parameter-Normalized Efficiency

For PEFT methods:

\[
E_P =
\frac{\Delta Q}
{P_{\text{train}}}
\]

Since values become extremely small, report:

\[
\frac{\Delta Q}
{\text{million trainable parameters}}
\]

---

Another useful metric:

\[
E_{PH}
=
\frac{\Delta Q}
{P_{\text{train}}\times GPUHours}
\]

This represents adaptation performance obtained relative to both:

- optimization footprint
- training compute

---

# 23. Hardware Utilization

Record GPU utilization where possible.

Important metrics:

### Model FLOPs Utilization

\[
MFU =
\frac{\text{achieved model FLOPs/s}}
{\text{theoretical hardware FLOPs/s}}
\]

Also record:

- average GPU utilization
- average power
- memory utilization
- memory bandwidth if available

These are especially useful when a method unexpectedly has poor tokens/s.

---

# 24. Energy Efficiency

If GPU power telemetry is available:

\[
E =
\int P(t)dt
\]

Measure in:

\[
kWh
\]

Then:

\[
EnergyToTarget(Q_t)
\]

can be reported.

You can also calculate:

\[
\frac{\Delta Q}{kWh}
\]

This gives the project a useful sustainability angle.

---

# 25. Training Stability

Efficiency is meaningless if training is unstable.

Track:

- training loss
- validation loss
- gradient norm
- learning rate
- weight norm
- update norm
- NaN / overflow events

Optional:

\[
\frac{\|\Delta W\|}
{\|W\|}
\]

to measure relative update magnitude.

---

# 26. Weight-Space Analysis

This could become one of the more research-oriented extensions.

For FFT:

\[
\Delta W = W_{FT}-W_0
\]

For LoRA:

\[
\Delta W = BA
\]

Compare:

- Frobenius norm
- spectral norm
- effective rank
- singular-value distribution
- cosine similarity of updates

Question:

> Is LoRA learning a low-rank approximation of essentially the same update that FFT learns?

---

# 27. LoRA Rank Study

Run:

\[
r \in
\{2,4,8,16,32,64\}
\]

Measure:

\[
Q(r)
\]

and:

\[
N^*(Q_t,r)
\]

Then investigate:

> At what rank does increasing parameter count stop meaningfully improving learning efficiency?

---

# 28. Trainable-Parameter Budget Matching

A particularly strong experiment:

Give each PEFT method approximately the same number of trainable parameters.

For example:

\[
P_{\text{train}}\approx10M
\]

Then compare:

- LoRA
- DoRA
- AdaLoRA
- IA³ where feasible

This makes the question:

> Given the same adaptation parameter budget, which parameterization learns most efficiently?

Much cleaner scientifically.

---

# 29. Compute-Budget Matching

Another experiment:

Give each method exactly:

\[
10\ GPU-hours
\]

and compare final quality.

Or:

\[
10^{18}\ FLOPs
\]

if compute measurement is reliable.

Then:

\[
Q(C=\text{constant})
\]

becomes the comparison.

---

# 30. Data-Budget Matching

Give all methods exactly:

\[
50M
\]

training tokens.

Compare quality.

This measures statistical efficiency.

---

# 31. Memory-Budget Matching

An interesting practical comparison:

Give every method the same hardware constraint.

Example:

\[
24GB\ VRAM
\]

Allow each method to choose its maximum feasible batch size.

Then ask:

> Under real consumer-GPU constraints, which strategy reaches the target quality fastest?

This is particularly favorable for testing QLoRA's practical value.

---

# 32. Evaluation Metrics

Training loss alone is insufficient.

Use three levels.

## Level 1 — Held-Out Loss

- cross entropy
- perplexity

\[
PPL=e^L
\]

---

## Level 2 — Task Performance

Depending on task:

- exact match
- accuracy
- F1
- ROUGE
- BLEU
- pass@k

---

## Level 3 — General Capability Retention

Evaluate the base model before and after fine-tuning on unrelated tasks.

Define:

\[
Forgetting =
Q_{\text{base}}
-
Q_{\text{after FT}}
\]

Question:

> Does PEFT preserve pretrained capabilities better than FFT?

This could be important.

---

# 33. Learning–Forgetting Tradeoff

You can define:

\[
AdaptationGain
=
Q_{\text{target,after}}
-
Q_{\text{target,before}}
\]

and:

\[
ForgettingLoss
=
Q_{\text{general,before}}
-
Q_{\text{general,after}}
\]

Then plot:

\[
AdaptationGain
\quad vs \quad
ForgettingLoss
\]

This could reveal important differences between FFT and PEFT.

---

# 34. Evaluation Checkpoints

Do not evaluate only at the end.

For example evaluate after:

\[
0,
1M,
2M,
5M,
10M,
20M,
50M
\]

tokens.

Or approximately log-spaced checkpoints.

Early training needs denser evaluation because the learning curve changes rapidly.

---

# 35. Seeds

Initial exploratory experiments:

\[
1\ seed
\]

Final experiments:

\[
\boxed{3\ seeds}
\]

Minimum.

For highly noisy datasets, preferably:

\[
5
\]

Report:

\[
mean\pm standard\ deviation
\]

---

# 36. Fairness Controls

The comparison can become invalid very easily.

Keep identical where possible:

- base checkpoint
- tokenizer
- train dataset
- validation dataset
- sample ordering
- tokenization
- sequence length
- packing strategy
- optimizer family
- evaluation code
- precision

Hyperparameters such as learning rate may legitimately differ between FFT and PEFT.

Therefore perform a small tuning sweep separately for each method.

---

# 37. Learning-Rate Sweep

For example:

FFT:

\[
\{5\times10^{-6},1\times10^{-5},2\times10^{-5},5\times10^{-5}\}
\]

PEFT:

\[
\{5\times10^{-5},1\times10^{-4},2\times10^{-4},5\times10^{-4}\}
\]

Do not assume they share an optimal LR.

Choose based on validation performance.

Then freeze those settings for the main experiments.

---

# 38. Optimizer Control

Initially use one optimizer across experiments:

AdamW.

Record:

- LR
- beta1
- beta2
- epsilon
- weight decay
- scheduler
- warmup

Optimizer differences should not become a confound.

---

# 39. Sequence-Length Control

Sequence length strongly affects throughput because attention cost depends strongly on context length.

Begin with:

\[
L=512
\]

or:

\[
L=1024
\]

Then optionally test:

\[
512,\quad1024,\quad2048,\quad4096
\]

Question:

> Does PEFT's computational advantage change with context length?

This is a valuable systems experiment.

---

# 40. Batch-Size Control

Report both:

### Micro batch

Sequences simultaneously resident on GPU.

### Effective batch

\[
B_{effective}
=
B_{micro}
\times
gradientAccumulation
\times
N_{GPU}
\]

Keep effective batch size approximately constant when comparing learning behavior.

For memory-constrained experiments, separately allow maximum feasible batch size.

These are two different experiments.

---

# 41. Checkpoint Everything Needed for Curve Reconstruction

Each logging point should contain something similar to:

```text
run_id
method
model
dataset
seed

global_step
tokens_seen
examples_seen

train_loss
validation_loss
evaluation_score

wall_clock_seconds
gpu_hours
estimated_flops

tokens_per_second
samples_per_second

peak_vram
allocated_vram

trainable_parameters
total_parameters

learning_rate
gradient_norm
```

---

# 42. Main Result Table

Your final benchmark table could look like:

| Method | Score ↑ | PPL ↓ | Tok/s ↑ | Peak VRAM ↓ | Trainable % ↓ | Tokens-to-target ↓ | GPU-h-to-target ↓ | AULC ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FFT | | | | | 100 | | | |
| LoRA | | | | | | | | |
| QLoRA | | | | | | | | |
| DoRA | | | | | | | | |
| AdaLoRA | | | | | | | | |
| IA³ | | | | | | | | |

---

# 43. Essential Figures

## Figure 1

\[
Validation\ Loss
\quad vs \quad
Tokens
\]

Measures data efficiency.

---

## Figure 2

\[
Validation\ Loss
\quad vs \quad
Wall\ Time
\]

Measures practical training efficiency.

---

## Figure 3

\[
Downstream\ Score
\quad vs \quad
GPU\ Hours
\]

Probably one of the strongest plots.

---

## Figure 4

\[
Score
\quad vs \quad
Trainable\ Parameters
\]

Measures parameter efficiency.

---

## Figure 5

\[
Score
\quad vs \quad
Peak\ VRAM
\]

Measures hardware efficiency.

---

## Figure 6

Pareto frontier:

\[
Quality
\quad vs \quad
Compute
\]

Highlight nondominated methods.

---

# 44. Pareto Analysis

A method dominates another if it achieves:

- higher quality
- lower compute
- lower memory

simultaneously.

Construct Pareto frontiers such as:

\[
Quality \leftrightarrow GPUHours
\]

\[
Quality \leftrightarrow VRAM
\]

\[
Quality \leftrightarrow TrainableParams
\]

This may tell the story better than ranking methods with one arbitrary composite metric.

---

# 45. Minimum Viable Experiment

Do this first.

### Model

One ~0.5–1.5B decoder LLM.

### Dataset

One instruction dataset.

### Methods

\[
FFT,\ LoRA,\ QLoRA,\ DoRA
\]

### Seeds

1 initially.

### Token budgets

\[
1M,\quad5M,\quad10M,\quad25M
\]

### Metrics

- training loss
- validation loss
- downstream score
- tokens/s
- tokens seen
- wall-clock time
- peak VRAM
- trainable parameters

### Main plots

\[
Loss\ vs\ Tokens
\]

\[
Score\ vs\ Tokens
\]

\[
Score\ vs\ Time
\]

If those already produce interesting differences, continue.

---

# 46. Intermediate Benchmark

Add:

- 3 seeds
- 2 models
- 2 datasets
- AdaLoRA
- IA³
- LoRA rank sweep
- GPU-hours
- FLOPs
- forgetting evaluation
- AULC

This is already potentially substantial.

---

# 47. Full Research Version

A stronger project would contain:

### Models

3 scales.

### Tasks

3 adaptation types.

### Methods

6–8 methods.

### Data budgets

5–6 budgets.

### Seeds

3.

### Resource dimensions

- tokens
- time
- FLOPs
- memory
- energy
- trainable parameters

### Analysis

- time-to-target
- tokens-to-target
- compute-to-target
- AULC
- Pareto frontiers
- forgetting
- weight-update rank analysis

That becomes much more than a simple PEFT comparison.

---

# 48. Suggested Experimental Order

## Stage 0 — Pipeline validation

Train one tiny model using LoRA.

Verify:

- loss decreases
- evaluation works
- logging works
- tokens are counted correctly

---

## Stage 1 — FFT vs LoRA

Only these two.

This validates the central benchmark.

---

## Stage 2 — Add QLoRA and DoRA

Establish the main four-method benchmark.

---

## Stage 3 — Dataset scaling

Run multiple token budgets.

Produce learning curves.

---

## Stage 4 — Rank sweep

Investigate:

\[
r=2,4,8,16,32,64
\]

---

## Stage 5 — Multiple seeds

Repeat selected configurations three times.

Do not waste compute running every exploratory configuration with three seeds.

---

## Stage 6 — Multiple tasks

Add domain adaptation and reasoning.

---

## Stage 7 — Model scaling

Repeat important experiments across model sizes.

---

## Stage 8 — Advanced analysis

Add:

- forgetting
- weight-space analysis
- FLOPs
- energy
- AULC
- Pareto analysis
- equivalent learned-token metrics

---

# 49. Suggested Folder Structure

```text
llm-efficiency-benchmark/
│
├── configs/
│   ├── fft/
│   ├── lora/
│   ├── qlora/
│   └── dora/
│
├── data/
│
├── src/
│   ├── models.py
│   ├── peft.py
│   ├── train.py
│   ├── evaluate.py
│   ├── metrics.py
│   └── profiler.py
│
├── scripts/
│   ├── run_fft.sh
│   ├── run_lora.sh
│   └── run_sweep.sh
│
├── logs/
│
├── checkpoints/
│
├── results/
│   ├── raw/
│   ├── aggregated/
│   └── tables/
│
└── plots/
```

---

# 50. Recommended Implementation Stack

A practical implementation can use:

- PyTorch
- Hugging Face Transformers
- Hugging Face PEFT
- Accelerate
- bitsandbytes
- datasets
- evaluate
- Weights & Biases or TensorBoard
- NVIDIA NVML / pynvml for hardware telemetry

Hugging Face `Trainer` already supplies a complete Transformers training loop and integrates with Accelerate for distributed execution, while PEFT integrates directly with Transformers. This makes it reasonable to maintain one common training pipeline and switch adaptation strategies through configuration rather than implementing separate trainers.

---

# 51. QLoRA Implementation Note

For a controlled experiment:

### LoRA

```text
BF16 backbone
+
LoRA
```

### QLoRA

```text
4-bit NF4 backbone
+
same LoRA
```

Keep:

- rank
- alpha
- target modules
- dataset
- sequence length

identical.

The current bitsandbytes/Hugging Face implementation supports 4-bit LoRA training and NF4 quantization specifically for this style of setup.

This isolates the effect of quantization reasonably well.

---

# 52. DoRA Implementation Note

DoRA can be treated as a LoRA variant using the same general target-module framework.

This makes paired experiments particularly convenient:

```text
LoRA r=8
vs
DoRA r=8
```

Current PEFT support exposes DoRA through the LoRA configuration.

---

# 53. AdaLoRA Experiment

Do not compare arbitrary adapter sizes.

Match approximately equal parameter budgets.

AdaLoRA redistributes rank based on learned importance during training rather than assigning the same rank throughout the model.

Therefore compare:

\[
LoRA(P)
\quad vs \quad
AdaLoRA(P)
\]

at approximately the same adapter budget \(P\).

---

# 54. Primary Hypotheses

Write these before performing the final experiments.

### H1

PEFT requires fewer trainable parameters and less memory than FFT while retaining comparable downstream quality.

---

### H2

Raw training throughput alone will not predict time-to-target performance.

---

### H3

Different PEFT methods will exhibit different statistical efficiency:

\[
\frac{\Delta Q}{\Delta tokens}
\]

despite similar adapter parameter counts.

---

### H4

FFT may achieve higher asymptotic performance on tasks requiring substantial representation change while PEFT may dominate under limited resource budgets.

---

### H5

PEFT may produce more favorable:

\[
Quality/GPUHour
\]

during early and medium training.

---

### H6

The optimal adaptation strategy changes depending on:

- training-data size
- model scale
- target task
- available memory

---

# 55. Potential Stronger Research Question

Instead of:

> Is LoRA better than full fine-tuning?

use:

> Under what combinations of model scale, dataset size, adaptation task, and compute budget does parameter-efficient fine-tuning dominate full fine-tuning?

That is a substantially better research question.

It permits the answer:

> Neither is universally superior.

Instead, you discover the regimes in which each method is preferable.

---

# 56. Potential Main Contribution

A strong final framing would be:

> **Benchmarking Fine-Tuning as Learning Efficiency Rather Than Endpoint Performance**

The contribution becomes a multidimensional evaluation framework based on:

\[
\boxed{
\text{Quality}
\times
\text{Data Efficiency}
\times
\text{Compute Efficiency}
\times
\text{Memory Efficiency}
\times
\text{Parameter Efficiency}
}
\]

rather than merely another:

> LoRA vs QLoRA vs FFT accuracy table.

---

# 57. Final Priority Order

If resources are limited, prioritize these measurements:

1. **Validation/downstream quality**
2. **Tokens seen**
3. **Wall-clock training time**
4. **Tokens/s**
5. **Peak VRAM**
6. **Trainable parameter count**
7. **Tokens-to-target**
8. **Time-to-target**
9. **GPU-hours-to-target**
10. **AULC**
11. FLOPs
12. energy consumption
13. update-space analysis

The first ten already produce a very strong benchmark.

---

# 58. Recommended First Experiment

Start with:

\[
\boxed{
1\ model
\times
1\ dataset
\times
4\ methods
\times
4\ token\ budgets
}
\]

Methods:

\[
\boxed{
FFT,\ LoRA,\ QLoRA,\ DoRA
}
\]

Record every few hundred optimizer steps:

\[
\boxed{
loss,\ validation\ loss,\ score,\ tokens,\ seconds,\ VRAM
}
\]

The first question to answer is simply:

> **When all four methods are plotted as quality versus tokens and quality versus time, do they actually follow different learning trajectories?**

If the answer is yes, you have the foundation of the entire project.