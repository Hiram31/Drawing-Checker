# Drawing-Checker

**A vision RAG framework for automated comparison of engineering drawings**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Workflow](https://img.shields.io/badge/Pipeline-YOLO%20%2B%20CLIP%20%2B%20LLM-black)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-orange)

Drawing-Checker compares rasterized engineering drawings against reference drawings using YOLO view detection, CLIP retrieval and view matching, and a multimodal LLM through Azure OpenAI. It produces discrepancy reports and, when the LLM returns a valid structured summary, visual overlays for human review.

This repository contains the research prototype accompanying the published paper:

> Jiwei Zhou, Jorge D. Camba, Pedro Company, and Manuel Contero. **Drawing-Checker: A Vision RAG Framework for Automated Comparison of Engineering Drawings.** *Procedia CIRP*, **142**, 127–132, 2026. 36th CIRP Design Conference (CIRP Design 2026). [https://doi.org/10.1016/j.procir.2026.05.235](https://doi.org/10.1016/j.procir.2026.05.235).

[Read the paper on ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2212827126008085) · [Citation metadata](CITATION.cff)

## Overview

![Drawing-Checker pipeline: retrieval, view matching, discrepancy generation, and visual outputs](assets/drawing-checker-overview.png)

The main pipeline, [`pipeline/drawing_checker_yolo_GPT.py`](pipeline/drawing_checker_yolo_GPT.py), processes every PNG or JPEG target in the configured target directory:

1. Cache reference CLIP embeddings, detected view counts, and normalized view boxes as JSON files.
2. Retrieve the nearest reference using Euclidean distance between whole-image CLIP embeddings.
3. Detect and crop views in both drawings with YOLO (`conf=0.7`, `imgsz=640`).
4. Match views using mutual best CLIP matches with cosine similarity above `0.85` or bounding-box IoU above `0.25`, then an IoU fallback for remaining views.
5. Suppress unmatched reference views that overlap or lie near detected target views, then send matched crops and remaining unmatched views to Azure OpenAI for comparison.
6. Save the model's text report and parse its delimited JSON summary to draw annotation highlights and view outlines.

The bundled reference, target, and detector checkpoint support a starter run. The full training dataset, the paper's 500 customized evaluation cases, and evaluation automation are not included in the tracked repository. The current training scripts and configurable API deployment do not constitute an exact reproduction of the published experiments.

## Requirements

- Python 3.10 or newer and a virtual environment.
- An Azure OpenAI resource, API key, and deployed model that accepts image inputs through chat completions. The example configuration uses the deployment name `gpt-5-mini`; set it to your actual Azure deployment name.
- Internet access for Azure API requests and the first download of `openai/clip-vit-base-patch32` from Hugging Face. Cached CLIP files can be reused afterward.
- Poppler on `PATH` if using the optional PDF conversion utility.

API requests send drawing images or view crops to your configured Azure endpoint and incur usage charges. Use drawings you are permitted to process there. A GPU is not required by the comparison scripts; the training scripts select Apple MPS when available and otherwise use CPU.

## Installation and configuration

Run these commands from the repository root in a POSIX shell:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell and copy the template with `Copy-Item .env.example .env`.

Edit `.env` with your paths and Azure credentials. [`config.py`](config.py) loads it through `python-dotenv`; existing environment variables take precedence. All settings below are read from the environment without code-level defaults, so start with the complete template:

```dotenv
REFERENCE_DIR=images/ref_images
TARGET_DIR=images/target_images
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_KEY=<your-key>
AZURE_GPT_DEPLOYMENT=gpt-5-mini
API_VERSION=2024-12-01-preview
GOLD_STANDARD_DIR=gold_standards
OUTPUT_DIR=outputs
WITHOUT_SEGMENTATION_OUTPUT_DIR=without_segmentation_outputs
YOLO_MODEL_PATH=model/weights/drawing_checker.pt
YOLO_PREDICT_DIR=predictions
```

Run commands from the repository root so relative paths and the bundled font resolve correctly. Keep `.env` private; it is already excluded from version control by `.gitignore`.

The endpoint, key, deployment name, and API version must work together for your Azure resource. The deployment and API version in the template are example values, not provisioned services.

`REFERENCE_DIR` and `TARGET_DIR` must already exist. The scripts create their cache and output directories as needed. Dependencies in `requirements.txt` are unpinned, so installations can vary over time.

## Run the bundled example

The tracked starter assets are:

- Reference: [`images/ref_images/base_ref.png`](images/ref_images/base_ref.png)
- Target: [`images/target_images/base_target.png`](images/target_images/base_target.png)
- Detector: [`model/weights/drawing_checker.pt`](model/weights/drawing_checker.pt)

After configuring `.env`, run:

```bash
python pipeline/drawing_checker_yolo_GPT.py
```

The first run loads CLIP and generates reference metadata under `GOLD_STANDARD_DIR`. Subsequent runs reuse JSON files that contain `clip_embedding` and `view_count`; unreadable or incomplete files are regenerated. Image edits and detector changes do **not** automatically invalidate otherwise complete cache files. Remove the affected generated JSON files before rerunning when changing references or the detector. Use a separate cache directory for each reference collection, and remove stale JSON files when removing references.

**Automatic reference addition:** if the nearest reference has a CLIP distance greater than `5.5`, the script saves metadata for the target, copies its image into `REFERENCE_DIR` if that filename is absent, refreshes the reference embeddings, and skips comparison for that target. This changes the reference library and can affect later targets in the same run. It does not represent a human approval of that drawing; use a working copy of your reference collection when experimenting.

### Outputs

With the default configuration, results appear under `outputs/`:

| File or directory | Contents |
| --- | --- |
| `<timestamp>_<target>_<deployment>[_high].txt` | LLM comparison report, including the structured summary when returned |
| `<timestamp>_<target>_highlighted_view_<N>_<deployment>.png` | Matched view crops with missing annotations highlighted in red and extra annotations in orange |
| `<timestamp>_<target>_FULL_HIGHLIGHTED_<deployment>.png` | Target drawing with annotation highlights, green matched-view outlines, orange extra-view outlines, and red missing-view outlines |
| `debug_crops/` | Reference and target crops and detection visualizations; names are reused across runs |

Highlight images require a parseable JSON block between `STRUCTURED_SUMMARY_START` and `STRUCTURED_SUMMARY_END` with a `matched_views` field. If parsing fails, the text report remains available and the script logs `No structured annotations found in GPT output.` Malformed summary fields can also interrupt overlay generation. Targets automatically added as references do not receive comparison reports or discrepancy overlays.

## Full-image baseline

Run the baseline without YOLO detection or CLIP retrieval:

```bash
python pipeline/drawing_checker_full_img.py
```

It pairs reference and target files by their **independently sorted filenames**, using `zip`, rather than matching names or retrieving similar images. Arrange both directories so the sorted lists correspond. Any files beyond the shorter list are silently skipped. This script recognizes lowercase `.png`, `.jpg`, and `.jpeg` extensions.

The baseline sends each full-image pair to Azure OpenAI and writes only text reports to `WITHOUT_SEGMENTATION_OUTPUT_DIR`, named `<timestamp>_<target>_without_segmentation.txt`. It does not produce discrepancy overlays.

### Deployment-specific request settings

The main pipeline sends `reasoning_effort="high"` when the configured deployment name is exactly `o4-mini`, `gpt-5-mini`, or `gpt-5-chat`. For other deployment names it sends `temperature=0.0`. Both branches send `top_p=1.0`. The baseline always sends `reasoning_effort="high"` and `top_p=1.0`.

These branches depend on the **deployment name**, not automatic model detection. If you use a custom deployment name or a model that rejects these parameters, adapt the request settings in the corresponding script.

## Use your own drawings

Set `REFERENCE_DIR` to your reference collection and `TARGET_DIR` to the drawings to check. The main pipeline accepts `.png`, `.jpg`, and `.jpeg` extensions case-insensitively. Give each image a unique filename stem within its collection because reference cache filenames are based on stems. PDFs and CAD-native files are not direct pipeline inputs.

To rasterize a folder of PDFs, install Poppler separately and run:

```bash
python utils/pdf2png.py /path/to/pdfs --dpi 300 --output /path/to/output_pngs
```

The utility scans the folder without recursion and creates `<pdf_stem>_page<N>.png` for each page. Without `--output`, PNGs are saved beside their source PDFs.

## Detector prediction and training

To inspect detector predictions on `REFERENCE_DIR` without LLM comparison:

```bash
python model/predict.py
```

This script uses `YOLO_MODEL_PATH`, detects at confidence `0.6`, and writes annotated images to `YOLO_PREDICT_DIR`.

Training is optional. Supply your own YOLO-format dataset and edit [`model/dataset.yaml`](model/dataset.yaml) to point to it, preferably using an absolute dataset root. The configured class is `0: View`; training and validation labels must use the same class mapping. Then run one of the training scripts from `model/`:

```bash
cd model
python train_v8n.py
# Alternatively:
python train_v11n.py
```

Both scripts currently train for 100 epochs at image size 640, starting from `yolov8n.pt` or `yolo11n.pt`. They use different augmentation settings. The pretrained weights may be downloaded on first use. Point `YOLO_MODEL_PATH` to your resulting checkpoint and regenerate reference metadata before comparison.

The dataset and image utilities accept paths and settings as command-line arguments. Run them with `--help` for all options:

```bash
python utils/convert_voc_to_yolo.py /path/to/voc --labels-output /path/to/labels --visualization-output /path/to/previews --classes view
python utils/update_dataset_class_id.py /path/to/labels /path/to/remapped_labels --map 0:3 1:0
python utils/count_annotations.py /path/to/train/labels /path/to/val/labels
python utils/resize_an_img.py /path/to/input.png /path/to/resized.png --width 800 --height 600
```

For VOC conversion, `--classes` lists the exact XML class names in YOLO ID order; use names that match your annotations and training configuration. Conversion expects same-stem PNG images by default; use `--image-extension .jpg` for JPEGs. Default conversion outputs go to `outputs/voc/labels` and `outputs/voc/previews`, which are ignored by Git when run from the repository root. Class remapping requires a different output directory, preserves unmapped IDs, and skips malformed label lines. Annotation counting ignores blank lines. Resizing sets the exact dimensions and may change the aspect ratio.

| Script | Purpose and setup |
| --- | --- |
| `utils/convert_voc_to_yolo.py` | Convert VOC XML annotations and visualize boxes; supply paths and class names through CLI arguments |
| `utils/update_dataset_class_id.py` | Remap label class IDs; supply source/destination directories and `--map OLD:NEW ...` |
| `utils/count_annotations.py` | Count YOLO annotations; supply training and validation label directories |
| `utils/resize_an_img.py` | Resize one image; supply input/output paths and optional dimensions (default: 800 × 600) |
| `utils/plot_results.py` | Plot training metrics; edit the run CSV and output paths |
| `utils/convert_for_tensorboard.py` | Convert run CSVs to TensorBoard logs; expects `model/runs/detect` from the repository root |

## Repository layout

```text
.
├── README.md
├── CITATION.cff
├── .env.example                  # Configuration template; no credentials
├── config.py
├── requirements.txt
├── assets/drawing-checker-overview.png
├── pipeline/
│   ├── drawing_checker_yolo_GPT.py
│   └── drawing_checker_full_img.py
├── model/
│   ├── weights/drawing_checker.pt
│   ├── dataset.yaml
│   ├── predict.py
│   ├── train_v8n.py
│   └── train_v11n.py
├── utils/                        # PDF conversion, dataset and plotting helpers, fonts
└── images/
    ├── ref_images/base_ref.png
    └── target_images/base_target.png
```

`gold_standards/`, `outputs/`, `without_segmentation_outputs/`, and `predictions/` are generated locally and ignored by Git. Training data, training runs, additional drawings, and archived experiments are also ignored. The two starter PNGs and `model/weights/drawing_checker.pt` are explicit exceptions and are tracked.

## Limitations

This is a research prototype intended to assist human review. Missed or incomplete view detections can exclude annotations from the LLM input, and the current matcher can fail when either drawing has no detected views. Spatial fallbacks assume comparable image layouts and scales. LLM reports can miss differences, invent findings, or return inaccurate coordinates. The overlay schema handles missing and extra annotations, without a separate modified-annotation category. Review the source drawings and text reports alongside the highlights.

## Citation

If you use Drawing-Checker, adapt its code, or build on the workflow described in our paper, please cite Zhou et al. (2026), DOI: 10.1016/j.procir.2026.05.235.

```bibtex
@article{zhou2026drawingchecker,
  author  = {Zhou, Jiwei and Camba, Jorge D. and Company, Pedro and Contero, Manuel},
  title   = {{Drawing-Checker}: A Vision {RAG} Framework for Automated Comparison of Engineering Drawings},
  journal = {Procedia CIRP},
  volume  = {142},
  pages   = {127--132},
  year    = {2026},
  doi     = {10.1016/j.procir.2026.05.235},
  url     = {https://doi.org/10.1016/j.procir.2026.05.235}
}
```

[`CITATION.cff`](CITATION.cff) also identifies the paper as the preferred citation for GitHub's citation feature.

## License

No repository license is currently included. The paper's publication license is separate from the licensing of the code, checkpoint, drawings, and bundled fonts. A repository license and any applicable third-party notices should be added before distributing this as an open-source release.
