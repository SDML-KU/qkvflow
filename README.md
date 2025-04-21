# Neural ODE Transformers: Analyzing Internal Dynamics and Adaptive Fine-tuning

[![ICLR 2025](https://img.shields.io/badge/ICLR-2025-blue)](https://iclr.cc/)

This repository contains the official implementation of the paper **"Neural ODE Transformers: Analyzing Internal Dynamics and Adaptive Fine-tuning"**, accepted at the International Conference on Learning Representations (ICLR) 2025.

## Abstract

Recent advancements in large language models (LLMs) based on transformer architectures have sparked significant interest in understanding their inner workings. In this paper, we introduce a novel approach to modeling transformer architectures using highly flexible non-autonomous neural ordinary differential equations (ODEs). Our proposed model parameterizes all weights of attention and feed-forward blocks through neural networks, expressing these weights as functions of a continuous layer index. Through spectral analysis of the model's dynamics, we uncover an increase in eigenvalue magnitude that challenges the weight-sharing assumption prevalent in existing theoretical studies. We also leverage the Lyapunov exponent to examine token-level sensitivity, enhancing model interpretability. Our neural ODE transformer demonstrates performance comparable to or better than vanilla transformers across various configurations and datasets, while offering flexible fine-tuning capabilities that can adapt to different architectural constraints.

## Model Architecture

Our model formulates transformers as neural ODEs with highly flexible non-autonomous vector fields. Instead of shared weights across layers, we parameterize all weights through neural networks that express these weights as functions of a continuous layer index (time).
The model includes:

- Time-dependent weights for attention components (Q, K, V)
- Time-dependent weights for feed-forward networks
- Representation of weights using a time-dependent unit that embeds time information in the Fourier domain


## Experimental Results


### Language Modeling
Comparable or better performance than vanilla transformers across various configurations
![Perplexity](assets/llama_vs_our_llama_1b.jpg)

Significant improvements in downstream tasks, particularly in reading comprehension
![Downstream](assets/five_shot_llama.png)

### Adaptive finetune
Flexible fine-tuning capabilities that can adapt to different architectural constraints
<div style="display: flex; justify-content: space-between;">
  <img src="assets/lora_wiki.jpg" alt="OWT to Wikitext" width="24%" />
  <img src="assets/full_wiki.jpg" alt="OWT to Wikitext" width="24%" />
  <img src="assets/lora_owt.jpg" alt="Wikitext to OWT" width="24%" />
  <img src="assets/full_owt.jpg" alt="Wikitext to OWT" width="24%" />
</div>


## Implementation
The implementation is built on JAX, utilizing an ecosystem that includes [Equinox](https://github.com/patrick-kidger/equinox), [Haliax](https://github.com/stanford-crfm/haliax), and the [Levanter](https://github.com/stanford-crfm/levanter) framework.


## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{
tong2025neural,
title={Neural {ODE} Transformers: Analyzing Internal Dynamics and Adaptive Fine-tuning},
author={Anh Tong and Thanh Nguyen-Tang and Dongeun Lee and Duc Nguyen and Toan Tran and David Leo Wright Hall and Cheongwoong Kang and Jaesik Choi},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=XnDyddPcBT}
}
```

## License
[MIT License](LICENSE)

