# Third-party notices

The root MIT license applies to original RTTS contributions. It does not replace licenses of incorporated code or external assets. Upstream copyright notices in vendored files are retained.

| Component | Location/use | Upstream license |
| :--- | :--- | :--- |
| [CLIP](https://github.com/openai/CLIP) | `ovss/clip/`, tokenizer vocabulary | MIT; `licenses/CLIP.txt` |
| [NaCLIP](https://github.com/sinahmr/NACLIP) | Attention/OVSS modifications | MIT; `licenses/NaCLIP.txt` |
| [MLMP](https://github.com/dosowiechi/MLMP) | Evaluation scaffolding and adaptation methods | MIT; `licenses/MLMP.txt` |
| [TENT](https://github.com/DequanWang/tent) | Entropy adaptation | MIT; `licenses/TENT.txt` |
| [WATT](https://github.com/Mehrdad-Noori/WATT) | Weight averaging adaptation | MIT; `licenses/WATT.txt` |
| [SAM2](https://github.com/facebookresearch/sam2) | Vendored optional `sam2/` | Apache-2.0; `licenses/SAM2.txt` |
| [imagecorruptions](https://github.com/bethgelab/imagecorruptions) | Modified corruption code and frost assets | Apache-2.0; `licenses/imagecorruptions.txt` |
| [PAMR](https://github.com/visinf/1-stage-wseg) | Archived `utils_local/pamr.py` | Apache-2.0; file retains TU Darmstadt notice; Apache terms in `licenses/SAM2.txt` |

SAM, MMSegmentation, MMCV, timm, and other dependencies are installed separately under their own licenses. Dataset and model-checkpoint use is governed by upstream terms. The archived imagecorruptions fork contains compatibility changes for newer scikit-image; SAM2 experimental integration and RTTS backbone changes derive from the anonymous RTTS snapshot.

The upstream MLMP and WATT license files contain anonymized copyright holders; their texts are preserved without inventing replacements. TPT here is a local adaptation inspired by its upstream implementation as recorded in the source docstring.
