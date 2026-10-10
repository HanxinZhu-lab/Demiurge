# Third-party notices

Demiurge combines a custom video pipeline with modified third-party implementations. It is not an upstream release of any dependency, and this document does not grant a new license or replace the included terms.

## Included source

| Component | Included terms | Important conditions |
| --- | --- | --- |
| DiffSynth-Studio | `code/DiffSynth-Studio/LICENSE` | Apache-2.0; retain the license, attribution, and notices of modification. |
| TrajectoryCrafter | `code/TrajectoryCrafter/License` | Academic, research, and education use only; commercial and production use is prohibited by the included terms. |
| TrajectoryCrafter models and CogVideoX terms | `code/TrajectoryCrafter/License_Model` | Separate model restrictions apply; the original bilingual terms are retained. |
| DepthCrafter | `code/TrajectoryCrafter/DepthCrafter/LICENSE` | Academic, research, and education use only; commercial and production use is prohibited by the included terms. |
| CogVideoX, Diffusers, and other derived code | Copyright and license headers in the relevant source files | Preserve file-level notices and review the applicable upstream terms. |

Copyright and attribution belong to the respective rights holders, including Tencent, the CogVideo model team, the Hugging Face team, Stability AI, and the DiffSynth-Studio contributors. Original legal texts are intentionally not translated, shortened, or removed.

## Modifications

The bundled source already contains custom RGB/depth conditioning, model layers, multi-source warping, alignment, and checkpoint-loading changes. These are not pristine upstream packages. The original upstream commit IDs and a complete pre-existing patch inventory have not yet been established.

The publication cleanup changes comments, documentation, diagnostics, prompt configuration, and local path handling, and removes inactive experiments and verbose debug dumps. Modified source files carry a short notice referring here. This notice does not assert ownership of upstream code or establish permission to distribute pre-existing local modifications.

## Weights, media, and release status

- Pretrained weights, fine-tuned checkpoints, input videos, generated outputs, caches, and machine-specific provenance are excluded from the source release.
- Obtain model weights from their authorized distributors and comply with each model's license. A code license does not automatically cover its weights, training data, or generated media.
- Confirm the ownership and redistribution rights of the three custom checkpoints and any sample video before publishing them separately.
- Record exact upstream revisions and review all required notices before release.
- Choose a license for independently authored Demiurge code only after confirming ownership and compatibility. No blanket MIT or Apache license is granted for this combined repository.

The included noncommercial restrictions mean the complete pipeline should not be described as unrestricted, commercially usable, or OSI-approved open-source software. Public source availability and unrestricted open-source licensing are different claims.
