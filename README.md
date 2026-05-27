# When and How Human Curation Backfires: Experiments

This repository contains the experimental code for the paper __When and How Human Curation Backfires: Preference Alignment under Multi-Model Self-Consuming Loop__.

The experiments study preference alignment in a multi-model self-consuming loop. In this setting, multiple models are repeatedly updated using mixtures of real data, synthetic data, and human-curated synthetic data. The main empirical goal is to show that human curation does not always improve long-term alignment when models interact with each other through recycled training data.

The repository contains three experiments:

1. **Gaussian mechanism experiment**: a controlled synthetic experiment for validating the theoretical mechanism.
2. **CIFAR-10 experiment**: a diffusion-model experiment with conflicting visual preferences.
3. **Qwen2.5-0.5B experiment**: a language-model experiment for preference domain mismatch.

## Repository Structure

```text
curationBackfires/
├── cifar10_experiment/
│   ├── figures/
│   │   ├── curve_allrewards.png
│   │   ├── curve_allrewards.svg
│   │   ├── curve_converg_A4_cu6_Jp_Jq_A1_6.png
│   │   └── curve_converg_A4_cu6_Jp_Jq_A1_6.svg
│   ├── scripts/
│   │   ├── plot.ipynb
│   │   └── utils.py
│   └── cifar10_experiment.py
│   
│
├── guassian_experiment/
│   ├── guassian_mechanism_outputs/
│   │   ├── exact_consistency_summary_hd.txt
│   │   ├── master_figure_hd.png
│   │   └── master_figure_hd.svg
│   └── gaussian_mechanism.py
│
├── qwen_experiment/
|   ├── data/
|   │   ├── para_train.jsonl
|   │   ├── para_val.jsonl
|   │   ├── sum_train.jsonl
|   │   └── sum_val.jsonl
|   ├── src/
|   │   ├── data.py
|   │   ├── modeling.py
|   │   ├── reward.py
|   │   ├── sft.py
|   │   └── utils.py
|   ├── iterative_loop.py
|   ├── plot.ipynb
|   ├── prepare_data.py
|   └── train_lora.py
└── requirements.txt
```

## Experiment Overview

### 1. Gaussian Mechanism Experiment

The Gaussian experiment validates the mechanism behind self-influence and cross-influence in an analytically tractable system. It uses two coupled Gaussian models with quadratic losses and explicitly computable stable points.

This experiment is designed to show that the sensitivity matrices induced by cross-model interaction can amplify, attenuate, or reverse the effect of human-curated data. As a result, even when the curation-induced update direction is locally aligned with the reward-improving direction, the final long-term effect on alignment can become negative.

### 2. CIFAR-10 Experiment

The CIFAR-10 experiment uses two class-conditional diffusion models. Both models generate images conditioned on CIFAR-10 class labels. Human curation is implemented using hue-based reward functions with conflicting preferences:

- model $\theta$ prefers warm-toned images;
- model $\phi$ prefers cool-toned images.

As shown in the main paper, the experiment evaluates six interaction settings, denoted by **A1-A6**. These settings differ in the fractions of cross-model synthetic data used during iterative retraining. The purpose is to study how cross-model interaction changes the effect of increasing the human-curated data ratio.

### 3. Qwen2.5-0.5B Experiment

The Qwen experiment studies preference domain mismatch using two LoRA-adapted language models initialized from **Qwen2.5-0.5B-Instruct**:

- model $\theta$ is trained for summarization;
- model $\phi$ is trained for paraphrasing.

The experiment considers a cross-domain setting where each model is trained using data induced by the other task. This setup is used to examine whether cross-model adaptation transfers to the target evaluation domain.

## Environment Setup

Create a Python environment:

```bash
conda create -n curation-backfires python=3.12
conda activate curation-backfires
pip install -r requirements.txt
```
Install the packages required by the experiments you want to run. A typical environment should include:

```bash
pip install numpy scipy pandas matplotlib tqdm jupyter
pip install torch torchvision
pip install diffusers transformers datasets accelerate peft trl
```

## Running the Experiments

### Gaussian Experiment

The Gaussian experiment is self-contained. It performs both simulation and visualization in a single script.

```bash
cd guassian_experiment
python gaussian_mechanism.py
```

The outputs will be saved to:

```text
guassian_mechanism_outputs/
```

Expected output files (the figure is identical to Figure 3 in the main paper) include:

```text
exact_consistency_summary_hd.txt
master_figure_hd.png
master_figure_hd.svg
```
### CIFAR-10 Experiment

The CIFAR-10 experiment has three stages:

1. train the baseline CIFAR-10 diffusion model using real data;
2. continue training under A1-A6 interaction settings with model interactions;
3. generate result visualizations.

Go to the CIFAR-10 experiment directory:

```bash
cd cifar10_experiment
```

#### Step 1: Train the baseline model

First, configure `cifar10_experiment.py` for baseline training. The baseline model is trained for 5 epochs and saved as:

```text
runs/baseline_cifar10
```

Run:

```bash
python cifar10_experiment.py --out_dir runs/baseline_cifar10 --iters 5 --trainset_size_p 25000 --trainset_size_q 25000 --frac_real_p 1.0 --frac_real_q 1.0 --frac_raw_p 0.0 --frac_raw_q 0.0 --frac_self_in_raw_p 0.0 --frac_self_in_raw_q 0.0 --frac_self_in_cur_p 0.0 --frac_self_in_cur_q 0.0 --real_train_size 25000 --real_test_size 5000 --train_steps_p 16000 --train_steps_q 16000 --train_steps_q 16000
```

After this step, check that the baseline checkpoint exists under `runs/baseline_cifar10`.

#### Step 2: Run A1-A6 iterative training

Starting from the baseline model, run six iterative training settings, **A1-A6**. Each setting continues training for 55 epochs with a different cross-model synthetic data configuration.

For each setting, modify the corresponding configuration, then run `cifar10_experiment.py`. For example, for the setting A5 with curation ratio 50%:

```bash
python cifar10_experiment.py --out_dir runs/A5_curation05 --iters 55 --trainset_size_p 8000 --trainset_size_q 8000 --eval_real_samples 5000 --frac_real_p 0.5 --frac_real_q 0.5 --frac_raw_p 0.0 --frac_raw_q 0.0 --frac_self_in_raw_p 0.0 --frac_self_in_raw_q 0.0 --frac_self_in_cur_p 0.8 --frac_self_in_cur_q 0.7 --real_train_size 50000 --real_test_size 5000 --train_steps_p 4000 --train_steps_q 4000 --lr_decay_start 40 --init_p_path runs/baseline_cifar10/ckpt/p_unet_iter_004.pt --init_q_path runs/baseline_cifar10/ckpt/q_unet_iter_004.pt --color_alpha 0.3 --realism_beta 0.3 --bt_tau 1.0 
```

The settings A1-A6 are used to compare how different cross-model synthetic data fractions affect the final rewards of the two models.

#### Step 3: Generate CIFAR-10 figures

After all A1-A6 runs are completed, open and execute:

```text
scripts/plot.ipynb
```

The figures will be saved under:

```text
figures/
```

Expected output files (the figures are identical to Figure 4 and 6 in the paper) include:

```text
curve_allrewards.png
curve_allrewards.svg
curve_converg_A4_cu6_Jp_Jq_A1_6.png
curve_converg_A4_cu6_Jp_Jq_A1_6.svg
```

### Qwen2.5-0.5B Experiment

Go to the Qwen experiment directory:

```bash
cd qwen_experiment
```

#### Step 1: Prepare datasets

Run:

```bash
python prepare_data.py
```

This generates four JSONL datasets under `data/`:

```text
data/
├── para_train.jsonl
├── para_val.jsonl
├── sum_train.jsonl
└── sum_val.jsonl
```

The `sum_*` files are used for summarization, and the `para_*` files are used for paraphrasing.

#### Step 2: Train LoRA base models

Run `train_lora.py` twice with different task configurations.

First, train the summarization LoRA adapter:

```bash
python train_lora.py --task summarize \
  --train_jsonl data/sum_train.jsonl --val_jsonl data/sum_val.jsonl \
  --base_model Qwen/Qwen2.5-0.5B-Instruct --out_dir runs/p_summarize_lora --num_epochs 1
```

Then, train the paraphrasing LoRA adapter:

```bash
python train_lora.py --task paraphrase \
  --train_jsonl data/para_train.jsonl --val_jsonl data/para_val.jsonl \
  --base_model Qwen/Qwen2.5-0.5B-Instruct --out_dir runs/q_para_lora --num_epochs 1
```

Both adapters are initialized from `Qwen2.5-0.5B-Instruct`.

The two runs should be configured independently so that one output checkpoint corresponds to the summarization model and the other corresponds to the paraphrasing model.

#### Step 3: Run the iterative self-consuming loop

After the two task-specific LoRA base models are trained, run the iterative self-consuming loop. For example, for the setting with cross-model data ratio = 100% and

```bash
python iterative_loop.py --out_dir qwen_r0s0c1_noCur --base_model Qwen/Qwen2.5-0.5B-Instruct --p_lora runs/p_summarize_lora --q_lora runs/q_para_lora --sum_train_jsonl data/sum_train.jsonl --sum_val_jsonl data/sum_val.jsonl --para_train_jsonl data/para_train.jsonl --para_val_jsonl data/para_val.jsonl --iters 13 --samples_per_iter 1024 --lambda_real_p 0.0 --lambda_self_p 0.0 --lambda_cross_p 1.0 --lambda_real_q 0.0 --lambda_self_q 0.0 --lambda_cross_q 1.0 --rho_self_cur_p 0.0 --rho_cross_cur_p 0.0 --rho_self_cur_q 0.0 --rho_cross_cur_q 0.0 --train_steps_per_iter 512 --max_seq_len_p 1024 --max_seq_len_q 1024 --reward_tra_eval_same 1 --reward_type 1 --eval_num_samples 128 --load_in_4bit
```

This script performs the iterative multi-model self-consuming training loop and saves the resulting model checkpoints and reward records according to the paths specified in the script.

#### Step 4: Generate Qwen figures

Open and execute `plot.ipynb`. This notebook visualizes the target-domain and cross-domain reward curves from the Qwen experiment. The output figure is identical to Figure 5 in the main paper.

## Expected Results

### Gaussian Experiment

The Gaussian experiment should reproduce the mechanism-level findings:

- self-influence and cross-influence can be decomposed across different dimension blocks;
- the sensitivity matrices can amplify, attenuate, or reverse curation effects;
- finite-sample estimates approach the theoretical values as the sample size increases;
- the observed reward derivatives match the closed-form theoretical predictions.

### CIFAR-10 Experiment

The CIFAR-10 experiment should show that:

- the iterative retraining dynamics converge under the tested settings;
- increasing human curation improves alignment when cross-model interaction is weak;
- increasing human curation can reduce alignment when cross-model interaction is strong;
- the A1-A6 settings produce different long-term reward outcomes because they use different cross-model synthetic data fractions.

### Qwen2.5-0.5B Experiment

The Qwen experiment should show that:

- cross-domain updates improve cross-domain rewards;
- target-domain rewards remain relatively stable;
- preference domain mismatch weakens the observable effect of cross-model interaction on the target evaluation tasks.

## Main Scripts

### `cifar10_experiment/cifar10_experiment.py`

Main script for CIFAR-10 training. It is used for both baseline training and A1-A6 iterative training.

### `cifar10_experiment/scripts/plot.ipynb`

Notebook for visualizing CIFAR-10 convergence curves and reward curves.

### `cifar10_experiment/scripts/utils.py`

Utility functions used by the CIFAR-10 experiment and plotting notebook.

### `guassian_experiment/gaussian_mechanism.py`

Self-contained script for the Gaussian mechanism experiment. It runs the simulation and saves the final figures.

### `qwen_experiment/prepare_data.py`

Script for preparing summarization and paraphrasing datasets.

### `qwen_experiment/train_lora.py`

Script for training task-specific LoRA adapters based on Qwen2.5-0.5B-Instruct.

### `qwen_experiment/iterative_loop.py`

Main script for the Qwen iterative self-consuming training loop.

### `qwen_experiment/plot.ipynb`

Notebook for visualizing reward curves from the Qwen experiment.

### `qwen_experiment/src/`

Utility modules for the Qwen experiment:

- `data.py`: dataset loading and preprocessing utilities;
- `modeling.py`: model and adapter loading utilities;
- `reward.py`: reward computation functions;
- `sft.py`: supervised fine-tuning utilities;
- `utils.py`: general helper functions.

<!-- ## Citation

If you use this code, please cite:

```bibtex
@inproceedings{zhang2026curationbackfires,
  title={When and How Human Curation Backfires: Preference Alignment under Multi-Model Self-Consuming Loop},
  author={Zhang, Yang and Wei, Xiukun and Zhang, Xueru},
  booktitle={Proceedings of the International Conference on Machine Learning},
  year={2026}
}
``` -->

## Contact

For questions about the experiments, please open an issue in this repository or contact the authors directly.
