# Third-party software and model notices

This project’s **original application source** is offered under the MIT License
(see `LICENSE`). This does **not** relicense third-party libraries, model
weights, operating-system packages, or services. Their own license terms apply.

## Principal components verified as of 2026-10-30

| Component | License confirmed | Upstream |
| --- | --- | --- |
| faster-whisper (recognition runtime) | MIT | https://github.com/SYSTRAN/faster-whisper/blob/master/LICENSE |
| OpenAI Whisper (original model/project) | MIT | https://github.com/openai/whisper/blob/main/LICENSE |
| Systran/faster-whisper-small (model currently bundled by default) | MIT (model card) | https://huggingface.co/Systran/faster-whisper-small |
| Vexa (architecture/algorithm design reference, not bundled as a dependency) | Apache-2.0 | https://github.com/Vexa-ai/vexa/blob/main/LICENSE |

### Model redistribution

The Docker build downloads and bundles a CTranslate2 Whisper model. Its
license is independent of this application's MIT License. The **default**
`Systran/faster-whisper-small` model is listed as MIT in its upstream card.
If `MODEL_SIZE` or `ASR_MODEL_PATH` is changed, **verify the exact model
repository, revision, model card, and redistribution conditions independently**
before shipping a new image.

### Libraries, native runtime, and containers

The Python dependency lists in `requirements.txt` and
`deploy/fargate/requirements.txt` include additional libraries and indirect
dependencies. Docker also includes a Python base image, Linux packages
(including `libgomp1`) and native packages pulled during installation.

**This document is not an exhaustive attribution or license audit.** Before
redistributing a Docker image or making a commercial package, collect licenses
and required notices for the exact resolved dependency versions, transitive
packages, base image, and downloaded model. Retain their license texts and
copyright notices as required. In particular, pay attention to native media
libraries and their build-time licensing options.

### Vexa design reference

The Fargate streaming pipeline was informed by concepts documented in the
Vexa project (e.g. LocalAgreement, time-based audio windows, quality gates).
The implementation here is a separate Python implementation and does not
vendor Vexa source files. Vexa is licensed under Apache-2.0; do not copy its
source code into this MIT repository without checking attribution, notices,
and modification conditions. This note is provided as a transparent design
reference and does not imply Vexa endorsement.

### Hosted services and input rights

AWS Bedrock is an optional hosted service, not software relicensed here.
Meeting audio, privacy, recording consent, trademarks, and applicable service
terms are outside the scope of the source-code license.
