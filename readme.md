# Anonymization-Guided Latent Perturbation for Source-Identity Reduction with Limited Edit Deviation in Instruction-Guided Image Editing

Official research repository for:

**Sanghyeok Seo, Jiwoo Kim, and Jong-Uk Hou**  
Hallym University, Chuncheon, Republic of Korea

> Manuscript submitted to *Sensors*.

## Overview

Instruction-guided image editing can modify semantic attributes while preserving the recognizable identity of the source subject. This creates a risk of unauthorized identity-preserving manipulation in which a recognizable person is placed in a fabricated semantic context.

This work studies an identity-oriented protection strategy that jointly considers:

- **source-identity reduction**, and
- **deviation from the corresponding undefended edit**.

Rather than aiming only for maximal disruption of the editing process, the proposed approach seeks an empirical operating point where source-identity similarity is reduced while the resulting edit remains comparatively close to the edit obtained from the original, unprotected image.

## Method

Given a source image \(x\), we first generate an anonymized counterpart \(x_{\mathrm{anon}}\). The anonymized image is used as a **source-conditioned latent reference**, not as the final protected output.

Using a VAE encoder \(E\),

\[
z_{\mathrm{anon}} = E(x_{\mathrm{anon}})
\]

and the latent representation of a perturbed source image is

\[
z_{\mathrm{perturbed}} = E(x+\delta).
\]

The default formulation optimizes an \(\ell_\infty\)-bounded pixel-space perturbation toward the anonymization-derived latent reference:

\[
\delta^*
=
\arg\min_{\delta}
\left\|E(x+\delta)-z_{\mathrm{anon}}\right\|_2^2
\quad
\mathrm{s.t.}
\quad
\|\delta\|_\infty \leq \epsilon.
\]

The protected image is then

\[
x_{\mathrm{protected}} = x+\delta^*.
\]

The anonymized counterpart is used only as an optimization reference. The method does **not** assume that Euclidean distance in the VAE latent space is identity-specific, and it does **not** guarantee that the edited output reproduces a particular anonymous identity.

## Evaluation

Experiments were conducted on **9,998 CelebA-Wild images**.

The primary evaluation uses **HIVE** as the instruction-guided image editor. Additional experiments and qualitative observations include **InstructPix2Pix (IP2P)** and **DreamBooth**.

Evaluated quantities include:

- Face-recognition similarity (**FR**) for source-identity similarity
- CLIP image similarity (**CLIP-I**)
- Prompt-related score (**CLIP-S**)
- **PSNR**
- **SSIM**
- **LPIPS**

PSNR, SSIM, and LPIPS are used to measure deviation between a defended edit and the corresponding undefended edit. They are not interpreted as direct measures of human perceptual quality.

## Main Results

In the primary HIVE setting:

- No Defense FR: **0.620**
- Ours FR: **0.567**
- Ours PSNR: **22.93**
- Ours SSIM: **0.747**
- Ours LPIPS: **0.278**

The proposed method does not achieve the lowest FR score among all evaluated defenses. Instead, it provides moderate source-identity reduction while obtaining the highest PSNR and SSIM and the lowest LPIPS among the evaluated defense methods in the primary setting.

On InstructPix2Pix, the proposed method obtains an FR score comparable to PhotoGuard while showing lower reference-edit deviation in LPIPS:

- Ours LPIPS: **0.262**
- PhotoGuard LPIPS: **0.428**

These results are interpreted as an **identity--edit-deviation trade-off**, rather than as universal superiority over stronger disruption-oriented or identity-oriented defenses.

## Repository Note

This repository contains research code developed during the experiments for this project. Some scripts may include experimental or ablation variants used during development.

For the manuscript's reported **default formulation**, the core objective is the anonymization proximity loss described above. Auxiliary disruption- or face-recognition-based objectives are treated as experimental/ablation variants rather than components of the default formulation.

Please refer to the manuscript for the exact experimental setting associated with each reported result.

## Dataset

The experiments use the publicly available **CelebA-Wild** dataset.

The initial evaluation subset contained 10,000 images. Two images were excluded because face detection failed, resulting in **9,998 evaluated images**.

The dataset itself is not redistributed in this repository.

## Limitations

The current results should be interpreted within the evaluated settings.

In particular:

- the \(\ell_\infty\) constraint bounds pixel-wise perturbation magnitude but does not independently establish perceptual imperceptibility;
- the VAE latent objective does not explicitly disentangle identity from pose, expression, background, or other content;
- the experiments do not establish that the anonymized target is causally superior to alternative target constructions;
- the final edited output is not guaranteed to match the identity of the anonymized reference;
- PSNR, SSIM, and LPIPS measure deviation from the corresponding undefended edit rather than human-perceived image quality;
- cross-editor robustness is not comprehensively established;
- robustness to JPEG compression, resizing, cropping, purification, and VAE re-encoding is not systematically evaluated.

## Citation

If you use this work, please cite the paper after its bibliographic information becomes available.

```bibtex
@article{seo2026anonymization,
  title   = {Anonymization-Guided Latent Perturbation for Source-Identity Reduction with Limited Edit Deviation in Instruction-Guided Image Editing},
  author  = {Seo, Sanghyeok and Kim, Jiwoo and Hou, Jong-Uk},
  year    = {2026},
  note    = {Manuscript submitted to Sensors}
}
```

## Authors

- **Sanghyeok Seo** — Division of AI Convergence, Hallym University
- **Jiwoo Kim** — Division of Software, Hallym University
- **Jong-Uk Hou** — Division of Software, Hallym University

Corresponding author: **Jong-Uk Hou**  
Email: **juhou@hallym.ac.kr**

## Funding

This research was supported by a **Hallym University Research Fund, 2025 (HRF-202501-013)**.

## License

Please refer to the repository license for usage terms.


### Example

To run the script using **GPU 0**, an **epsilon** of **$0.05$**, and enabling both **`loss_disrupt`** and **`loss_fr`**:

```bash
python main_defend.py --device "0" --eps 0.05 --disrupt True --fr True
```
