# LRIF-Net: Reproducibility Code

This repository contains the release code for **LRIF-Net (H32-CDP)**, the
baseline-preserving residual multimodal fusion model described in the submitted
SCI manuscript. It is intentionally limited to the final paper-facing method,
its locked protocol, and focused verification tests.

## Scope and data

The formal protocol uses the canonical CrisisMMD Task 2 split (7,314
image-text pairs: 5,119 train, 1,097 validation, and 1,098 test) with
`openai/clip-vit-large-patch14-336`. The dataset, tweet IDs, images, checkpoints,
and result archives are **not** included here. Obtain CrisisMMD from its official
source and follow its terms of use; configure local paths at run time rather than
hard-coding private machine paths.

## Layout

- `src/train_pdlf_clip.py`: frozen CLIP/B1 parent and dataset utilities.
- `src/h32_rlif_model.py`: recurrent latent interaction model.
- `src/train_h32_rlif.py`: shared H32 data, corruption, loss, and evaluation code.
- `src/train_h32_cdp.py`: locked clean-decision-preserved training entry point.
- `src/evaluate_h32_cdp_final_test.py`: one-time locked test evaluator; use only
  after the manuscript's test lock and authorization requirements are satisfied.
- `src/audit_h32_cdp_run.py`: read-only artifact audit.
- `configs/`: the locked H32-CDP protocol and implementation hash record.
- `docs/H32_CDP_PROTOCOL.md`: scientific protocol and stop rules.
- `tests/`: CPU-level model and loss tests; they do not load the CrisisMMD test split.

## Environment

Python 3.10+ with a CUDA-capable PyTorch installation is recommended for full
training. Install the pinned high-level dependencies with:

```powershell
python -m pip install -r requirements.txt
```

The CLIP checkpoint is downloaded by Hugging Face Transformers when first used.
For an offline or institutional environment, pre-cache the checkpoint and pass
the local model identifier instead.

## Verification

From the repository root:

```powershell
python -m py_compile src\train_pdlf_clip.py src\h32_rlif_model.py src\train_h32_rlif.py src\train_h32_cdp.py src\evaluate_h32_cdp_final_test.py src\audit_h32_cdp_run.py
python -m pytest -q tests\test_h32_rlif.py tests\test_h32_cdp.py
```

The tests use small dummy modules and are intended as implementation smoke
checks. They are not substitutes for the locked validation protocol.

## Reproduction boundaries

Do not tune hyperparameters or select checkpoints from the test set. Validation
results select development checkpoints; the test evaluator is deliberately
locked and requires the matching authorization configuration and artifact hashes.
The manuscript reports the final locked results and should be treated as the
source of truth for paper numbers.

## Release status

This is a local/private release repository prepared for author and reviewer
reproducibility. Before public release, replace any institution-specific access
instructions with a stable repository or archive URL.

## License and citation

The source code is released under the MIT License. This license does not apply
to CrisisMMD or any other third-party dataset, image, model checkpoint, or
publication content. Citation metadata are provided in `CITATION.cff`.
