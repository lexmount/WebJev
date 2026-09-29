# Notices

**Mapika/decider** (<https://github.com/Mapika/decider>, Apache License 2.0). `setup.sh` checks out commit
`c4daaac28af9fea95d627015cffa2dd5a5926ee6` into `third_party/decider`; it is not redistributed in this
repository. We use it unchanged for the model readout and loss (`decider.model`, `decider.train`), the prompt
format (`decider.prompt`, `decider.systemone`), the data recipe (`decider.data`, including its task loaders,
teacher prompts and mixture builder) and the Muon/AdamW update rules (`moe/optim.py`). The code in `src/` and
`data/` is ours: the 8 x A100 execution (rank scheduling, sharded CPU-resident optimizer state, bounded expert
blocks, checkpoint/resume), the export, the teacher client and its verification, and the builders of the
added data components. `src/export.py` copies the upstream `decider/` package into the exported model
directory, where the serving code imports it; that copy stays under the Apache License 2.0.

**Qwen/Qwen3.5-35B-A3B-Base** (<https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Base>, Apache License 2.0) is the base
model. Exported model directories include its license file.

**Open-Jev community checker** (<https://github.com/Zefan-Cai/Open-Jev>, MIT). `data/download_sources.py` fetches
`jev/community_diversity_v2.py` and `jev/data.py` at commit `3308a15` to re-derive the labels of the community
policy scenarios; they are not redistributed here.

**Datasets.** Every dataset used by the builders keeps its own license and terms of use: the Hugging Face
datasets listed in `data/general/source_revisions.json` and `data/download_sources.py`, Mind2Web, TREC, the
teacher files shipped with Mapika/decider, and our live-web decisions (see the dataset card at
<https://huggingface.co/datasets/Lexmount/WebJev>).
