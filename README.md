# FantasyPortrait: Enhancing Multi-Character Portrait Animation with Expression-Augmented Diffusion Transformers

[![Home Page](https://img.shields.io/badge/Project-FantasyPortrait-blue.svg)](https://fantasy-amap.github.io/fantasy-portrait/)
[![arXiv](https://img.shields.io/badge/Arxiv-2507.12956-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2507.12956)
[![hf_paper](https://img.shields.io/badge/🤗-FantasyPortrait-red.svg)](https://huggingface.co/papers/2507.12956)

## Abstract
Producing expressive facial animations from static images is a challenging task. Prior methods relying on explicit geometric priors (e.g., facial landmarks or 3DMM) often suffer from artifacts in cross reenactment and struggle to capture subtle emotions. Furthermore, existing approaches lack support for multi-character animation, as driving features from different individuals frequently interfere with one another, complicating the task. To address these challenges, we propose FantasyPortrait, a diffusion transformer based framework capable of generating high-fidelity and emotion-rich animations for both single- and multi-character scenarios. Our method introduces an expression-augmented learning strategy that utilizes implicit representations to capture identity-agnostic facial dynamics, enhancing the model's ability to render fine-grained emotions. For multi-character control, we design a masked cross-attention mechanism that ensures independent yet coordinated expression generation, effectively preventing feature interference. To advance research in this area, we propose the Multi-Expr dataset and ExprBench, which are specifically designed datasets and benchmarks for training and evaluating multi-character portrait animations. Extensive experiments demonstrate that FantasyPortrait significantly outperforms state-of-the-art methods in both quantitative metrics and qualitative evaluations, excelling particularly in challenging cross reenactment and multi-character contexts.
![Overview](https://github.com/Fantasy-AMAP/fantasy-portrait/raw/main/asserts/overview1_2.png)


## Code, Model and Datasets
The code, models, and dataset will be made publicly available soon.

## 🔗Citation
If you find this repository useful, please consider giving a star ⭐ and citation
```
@article{wang2025fantasyportrait,
  title={FantasyPortrait: Enhancing Multi-Character Portrait Animation with Expression-Augmented Diffusion Transformers},
  author={Wang, Qiang and Wang, Mengchao and Jiang, Fan and Fan, Yaqi and Qi, Yonggang and Xu, Mu},
  journal={arXiv preprint arXiv:2507.12956},
  year={2025}
}
```
