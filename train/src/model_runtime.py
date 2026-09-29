"""The trainable model: upstream DecisionModel on Qwen3.5-35B-A3B-Base, routed experts frozen.

Everything except the routed-expert weights is trained (about 2.45B of 34.66B language-model parameters):
attention, shared experts, routers, embeddings, norms and the LM head. The loss reads the LM-head rows of the
option letters at each answer slot (upstream `decider.model`), so the model learns a distribution over the
listed options in one forward pass; nothing is generated.

A100 memory adaptation: the routed experts still run the grouped-MM kernels, but in checkpointed blocks of
at most `expert_chunk_tokens` flattened token rows. This bounds the expert activations; it is not a context
limit and does not change routing (routing is per token and already decided when the experts run).
"""
import json
import os
from pathlib import Path
import sys

UPSTREAM = Path(os.environ.get("WEBJEV_UPSTREAM", Path(__file__).resolve().parents[1] / "third_party" / "decider"))
sys.path.insert(0, str(UPSTREAM))

import torch  # noqa: E402
from torch.utils.checkpoint import checkpoint  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from decider.model import DecisionModel, collate  # noqa: E402,F401  (collate is re-exported for train.py)
from decider.prompt import letter_ids  # noqa: E402

EXPERT_CHUNK_TOKENS = int(os.environ.get("WEBJEV_EXPERT_CHUNK_TOKENS", "1024"))


def bounded_experts_forward(original, limit=EXPERT_CHUNK_TOKENS):
    """Run the selected expert forward over row blocks; hidden states, top-k indices and weights are sliced together."""
    def forward(hidden_states, top_k_index, top_k_weights):
        if hidden_states.shape[0] <= limit:
            return original(hidden_states, top_k_index, top_k_weights)
        outputs = []
        for start in range(0, hidden_states.shape[0], limit):
            values = (hidden_states[start:start + limit], top_k_index[start:start + limit], top_k_weights[start:start + limit])
            if torch.is_grad_enabled():
                outputs.append(checkpoint(original, *values, use_reentrant=False))
            else:
                outputs.append(original(*values))
        return torch.cat(outputs, dim=0)
    return forward


def is_routed_expert(name: str) -> bool:
    return ".experts." in name and "shared" not in name


class TrainingModel(DecisionModel):
    def __init__(self, path, grad_ckpt=True):
        torch.nn.Module.__init__(self)
        torch.backends.cuda.enable_cudnn_sdp(False)
        self.tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.lm, info = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16, experts_implementation="grouped_mm", attn_implementation="sdpa",
            local_files_only=True, output_loading_info=True)
        if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs"):
            raise RuntimeError("base model loading is incomplete: " + str(info)[:4000])
        unexpected = [k for k in info.get("unexpected_keys", []) if not k.startswith(("mtp.", "model.visual."))]
        if unexpected:
            raise RuntimeError("unexpected base tensors: " + str(unexpected[:20]))
        self.loading_info = json.loads(json.dumps(info, default=lambda v: sorted(v) if isinstance(v, set) else str(v)))
        self.lm.config.use_cache = False
        if grad_ckpt:
            self.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.register_buffer("letters", torch.tensor(letter_ids(self.tok)), persistent=False)
        for name, param in self.lm.named_parameters():
            if is_routed_expert(name):
                param.requires_grad_(False)
        for name, module in self.lm.named_modules():
            if name.endswith(".experts"):
                # frozen weights still pass gradients to their inputs and routing weights
                module.forward = bounded_experts_forward(module.forward)

    def trainable_named(self):
        return [(name, param) for name, param in self.lm.named_parameters() if param.requires_grad]

    def parameter_summary(self):
        return dict(trainable=sum(p.numel() for _, p in self.trainable_named()),
                    frozen=sum(p.numel() for p in self.lm.parameters() if not p.requires_grad),
                    total=sum(p.numel() for p in self.lm.parameters()),
                    model_class=type(self.lm).__name__, loading_info=self.loading_info)


def to_cuda(batch, device):
    return {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}
