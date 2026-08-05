# ID-VTG: Image-Disambiguated Video Temporal Grounding

[![Dataset](https://img.shields.io/badge/Dataset-Hugging%20Face-yellow)](https://huggingface.co/datasets/Chloe-UniU-oO/ID-VTG)
[![Conference](https://img.shields.io/badge/ACM%20MM-2026-blue)](#citation)

Official PyTorch implementation of **ID-VTG: Image-Disambiguated Video Temporal Grounding** (ACM MM 2026).

> **Paper:** coming soon  
> **Dataset:** [Chloe-UniU-oO/ID-VTG](https://huggingface.co/datasets/Chloe-UniU-oO/ID-VTG)

## 📖 Introduction

Video Temporal Grounding (VTG) aims to localize a temporal segment in an untrimmed video according to a natural-language query. However, text alone can be ambiguous when multiple visually similar subjects perform the same or similar actions.

We introduce **Image-Disambiguated Video Temporal Grounding (ID-VTG)**, where each query contains:

- a **text query** describing the target action; and
- a **reference image** identifying the target instance.

We also introduce two benchmarks:

- **IDVTG-Gym**, which focuses on fine-grained and compositionally ordered gymnastics actions involving visually similar athletes;
- **IDVTG-InternVid**, which covers open-world videos with diverse entities and strong temporal distractors.

Our model, **Visually-Guided Disambiguation Aggregation (VGD-Agg)**, adopts a dual-branch fast-slow architecture. The fast branch generates temporal proposals, while the slow branch performs fine-grained matching between video frames and the reference image. A learnable **Compare Token** and **Depress Value** are used to suppress visual distractors and improve instance-aware temporal grounding.

## ✨ Highlights

- A new multimodal temporal grounding task using **text + reference-image queries**.
- Two ID-VTG benchmarks covering fine-grained sports and open-world videos.
- A dual-branch **VGD-Agg** framework for fine-grained visual disambiguation.
- Strong performance on IDVTG-Gym, IDVTG-InternVid, and the out-of-domain Web test set.

## 📊 Main Results

| Dataset | R@1, IoU=0.5 | R@1, IoU=0.7 | mIoU |
|---|---:|---:|---:|
| IDVTG-Gym | 61.83 | 56.53 | 54.17 |
| IDVTG-InternVid | 51.21 | 41.24| 48.30 |
| IDVTG-Web | 21.99 | 13.25 | 22.34 |

## 📁 Repository Structure

```text
.
├── data
│   ├── crop.py
│   └── sample.py
├── eval
│   ├── eval_idvtg_gym.sh
│   ├── eval_idvtg_internvid.sh
│   ├── idvtg_gym_opt_action.yaml
│   ├── idvtg_gym_opt_all.yaml
│   ├── idvtg_internvid_opt_all.yaml
│   ├── idvtg_internvid_opt_amb.yaml
│   └── idvtg_web_opt_all.yaml
├── eval.py
├── input_data
│   ├── annotations
│   ├── image_feat
│   └── video_feat
├── libs
│   ├── core
│   ├── data
│   ├── modeling
│   ├── nms
│   ├── dist_utils.py
│   ├── train_utils.py
│   └── worker.py
├── opts
│   ├── idvtg_gym.yaml
│   └── idvtg_internvid.yaml
└── train.py
```

## 🛠️ Installation

### 1. Create the Conda Environment

Clone this repository and create the Conda environment:

```bash
git clone git@github.com:JingliWei-oO/ID-VTG.git
cd ID-VTG

conda env create -f environment.yml
conda activate idvtg
```



### 2. Compile the 1D NMS Extension

This repository is built upon [SnAG](https://github.com/fmu2/snag_release) and uses a PyTorch C++ extension for efficient 1D temporal NMS.

After installing PyTorch and the remaining Python dependencies, compile the extension in the same Conda environment used for training and evaluation:

```bash
cd ./libs/nms
python setup_nms.py install
cd ../..
```

> [!IMPORTANT]
> Recompile the NMS extension whenever the PyTorch version is changed or upgraded.

> [!NOTE]
> Compilation is recommended on Linux and requires a compatible C++ compiler. If OpenMP is enabled by the extension, ensure that the compiler and OpenMP runtime are available on your system.

## 📦 Dataset and Feature Preparation

### 1. Download ID-VTG

The dataset is available on Hugging Face:

- [https://huggingface.co/datasets/Chloe-UniU-oO/ID-VTG](https://huggingface.co/datasets/Chloe-UniU-oO/ID-VTG)

It can also be downloaded with the Hugging Face CLI:

```bash
python -m pip install -U huggingface_hub

hf download Chloe-UniU-oO/ID-VTG \
    --repo-type dataset \
    --local-dir ./input_data/raw
```

Please follow the dataset card for any access requirements and the organization of the original videos, query images, and annotations.

### 2. Extract CLIP Features

This codebase expects pre-extracted visual features.

Use the pre-trained **CLIP ViT-L/14** model to extract:

- **video features**, saved as `{video_name}.npy`;
- **query-image features**, saved as `{query_id}.npy`.

The feature dimension must be:

```text
D = 768
```

The final data directory should follow this structure:

```text
input_data
├── annotations
│   ├── idvtg_gym_train.json
│   ├── idvtg_gym_val_action.json
│   ├── idvtg_gym_val.json
│   ├── idvtg_internvid_train.json
│   ├── idvtg_internvid_val.json
│   ├── idvtg_internvid_val_amb.json
│   └── idvtg_web_test.json
├── image_feat
│   ├── idvtg_gym_img_feat
│   │   └── {query_id}.npy
│   ├── idvtg_internvid_train_img_feat
│   │   └── {query_id}.npy
│   ├── idvtg_internvid_val_img_feat
│   │   └── {query_id}.npy
│   └── idvtg_web_img_feat
│       └── {query_id}.npy
└── video_feat
    ├── idvtg_gym_video_feat
    │   └── {video_name}.npy
    ├── idvtg_internvid_video_feat
    │   └── {video_name}.npy
    └── idvtg_web_video_feat
        └── {video_name}.npy
```

Before training or evaluation, check that the data paths in the YAML configuration files match your local directory structure.

## 🚂 Training

Run all commands from the repository root:

```bash
cd /path/to/ID-VTG
```

### IDVTG-InternVid

```bash
python ./train.py \
    --opt idvtg_internvid.yaml \
    --name idvtg_internvid
```

### IDVTG-Gym

```bash
python ./train.py \
    --opt idvtg_gym.yaml \
    --name idvtg_gym
```

Training outputs are saved under:

```text
experiments/<experiment_name>/
```

## 📈 TensorBoard

Optionally monitor training with TensorBoard:

```bash
tensorboard --logdir=./experiments/idvtg_internvid/tensorboard
```

or:

```bash
tensorboard --logdir=./experiments/idvtg_gym/tensorboard
```

## 🧪 Evaluation

The evaluation configurations and scripts are provided in `eval/`:

```text
eval
├── eval_idvtg_gym.sh
├── eval_idvtg_internvid.sh
├── idvtg_gym_opt_action.yaml
├── idvtg_gym_opt_all.yaml
├── idvtg_internvid_opt_all.yaml
├── idvtg_internvid_opt_amb.yaml
└── idvtg_web_opt_all.yaml
```

### Standard Evaluation on IDVTG-Gym

Copy the evaluation script and configuration into the corresponding experiment directory:

```bash
cd /path/to/ID-VTG

cp eval/eval_idvtg_gym.sh \
   eval/idvtg_gym_opt_all.yaml \
   ./experiments/idvtg_gym/

cd ./experiments/idvtg_gym
bash eval_idvtg_gym.sh
```

The evaluation logs will be written to:

```text
./experiments/idvtg_gym/eval/
```

### Standard Evaluation on IDVTG-InternVid

```bash
cd /path/to/ID-VTG

cp eval/eval_idvtg_internvid.sh \
   eval/idvtg_internvid_opt_all.yaml \
   ./experiments/idvtg_internvid/

cd ./experiments/idvtg_internvid
bash eval_idvtg_internvid.sh
```

The evaluation logs will be written to:

```text
./experiments/idvtg_internvid/eval/
```

### Evaluation at Different Action Granularities

To evaluate IDVTG-Gym separately at different action granularities, such as holistic events and sub-actions:

```bash
cd /path/to/ID-VTG

cp eval/idvtg_gym_opt_action.yaml \
   ./experiments/idvtg_gym/

i=1
python eval.py \
    --name idvtg_gym \
    --opt idvtg_gym_opt_action \
    --ckpt best${i} \
    --vis True
```

Replace `i=1` with the index of the checkpoint to evaluate.

### Evaluation under Different Ambiguity Types

To evaluate IDVTG-InternVid under different ambiguity settings:

```bash
cd /path/to/ID-VTG

cp eval/idvtg_internvid_opt_amb.yaml \
   ./experiments/idvtg_internvid/

i=1
python eval.py \
    --name idvtg_internvid \
    --opt idvtg_internvid_opt_amb \
    --ckpt best${i} \
    --vis True
```

Replace `i=1` with the index of the checkpoint to evaluate.

## 📝 Citation

If you find this project useful, please cite our paper:

```bibtex
@inproceedings{idvtg2026,
  title     = {ID-VTG: Image-Disambiguated Video Temporal Grounding},
  author    = {Minghang Zheng and Jingli Wei and Hongyi Yang and Yang Liu},
  booktitle = {Proceedings of the ACM International Conference on Multimedia},
  year      = {2026}
}
```

Please add the official DOI and page numbers once the final publication metadata is available.

## 🤝 Acknowledgements

This codebase is built upon [SnAG: Scalable and Accurate Video Grounding](https://github.com/fmu2/snag_release). We sincerely thank the authors for releasing their code.

We also acknowledge the open-source implementations and pretrained models of ActionFormer and CLIP.

If you use this codebase, please also consider citing SnAG:

```bibtex
@inproceedings{mu2024snag,
  title     = {SnAG: Scalable and Accurate Video Grounding},
  author    = {Mu, Fangzhou and Mo, Sicheng and Li, Yin},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2024}
}
```
