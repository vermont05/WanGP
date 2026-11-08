# Gemma 3 12B text encoder for LTX-2.3 on the iGPU (llama.cpp Vulkan, ~/src/gemma-layers).
# Enabled by LTX_GEMMA_IGPU=<path to the Gemma 3 12B (QAT) GGUF>. The eGPU then never loads Gemma:
# the iGPU reads its weights straight from system RAM instead of streaming them over the eGPU link.
import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch

BIN = os.environ.get("LTX_GEMMA_IGPU_BIN", str(Path.home() / "src/gemma-layers/gemma-layers"))
SHM = Path(os.environ.get("LTX_GEMMA_IGPU_TMP", "/dev/shm"))
N_CTX = 1024  # LTX pads/truncates Gemma prompts to 1024 tokens
# Seconds of no requests before the Gemma process exits and frees its ~7 GB of shared RAM (the iGPU has no VRAM of its
# own), so it is gone before denoising pins the transformer. Negative = keep it loaded forever.
IDLE_EXIT_S = float(os.environ.get("LTX_GEMMA_IGPU_IDLE_EXIT", "3"))


def gguf_path() -> str | None:
    p = os.environ.get("LTX_GEMMA_IGPU", "").strip()
    return p or None


class GemmaIGPU:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get(cls, n_tokens: int = N_CTX) -> "GemmaIGPU":
        # context sized to the prompt (multiples of 256): its KV cache and compute buffers are ~1.5 GB smaller at 256 than 1024
        n_ctx = min(N_CTX, max(256, -(-n_tokens // 256) * 256))
        with cls._lock:
            inst = cls._instance
            if inst is not None and (inst.proc.poll() is not None or inst.n_ctx < n_ctx):
                inst._cancel_idle(); inst.close(); cls._instance = None
            if cls._instance is None:
                cls._instance = cls(gguf_path(), n_ctx)
            cls._instance._cancel_idle()
            return cls._instance

    @classmethod
    def release_soon(cls):
        """Called after an encode: stop the process if no other request comes within IDLE_EXIT_S."""
        with cls._lock:
            inst = cls._instance
            if inst is None or IDLE_EXIT_S < 0:
                return
            inst._cancel_idle()
            inst._timer = threading.Timer(IDLE_EXIT_S, cls._idle_exit, args=(inst,))
            inst._timer.daemon = True
            inst._timer.start()

    @classmethod
    def _idle_exit(cls, inst):
        with cls._lock:
            if cls._instance is not inst:
                return
            cls._instance = None
        with inst.req_lock:
            inst.close()
        print("[Gemma iGPU] idle: process stopped, RAM freed", flush=True)

    def _cancel_idle(self):
        t = getattr(self, "_timer", None)
        if t is not None:
            t.cancel()
            self._timer = None

    def close(self):
        if self.proc.poll() is None:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()

    def __init__(self, model: str, n_ctx: int = N_CTX):
        self.n_ctx = n_ctx
        env = dict(os.environ)
        env["GGML_VK_VISIBLE_DEVICES"] = os.environ.get("LTX_GEMMA_IGPU_VK_DEVICE", "0")  # Vulkan0 = 680M iGPU
        env.pop("HSA_OVERRIDE_GFX_VERSION", None)
        t0 = time.time()
        self.proc = subprocess.Popen([BIN, model, str(n_ctx)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, text=True, env=env, bufsize=1)
        ready = self.proc.stdout.readline().split()
        if not ready or ready[0] != "READY":
            raise RuntimeError(f"gemma-layers failed to start ({' '.join(ready) or 'no output'})")
        self.n_layer, self.n_embd = int(ready[1]), int(ready[2])
        self.req_lock = threading.Lock()
        print(f"[Gemma iGPU] {Path(model).name} ready in {time.time() - t0:.1f}s ({self.n_layer} layers, ctx {n_ctx})", flush=True)

    def hidden_states(self, token_ids: list[int]) -> torch.Tensor:
        """float32 [n_layer + 1, n_tokens, n_embd] on the CPU."""
        out = SHM / f"gemma-layers-{os.getpid()}-{threading.get_ident()}.f32"
        with self.req_lock:
            self.proc.stdin.write(str(out) + " " + " ".join(map(str, token_ids)) + "\n")
            self.proc.stdin.flush()
            reply = self.proc.stdout.readline().split()
        if not reply or reply[0] != "OK":
            raise RuntimeError("gemma-layers: " + (" ".join(reply) or "process exited"))
        n, d = int(reply[1]), int(reply[2])
        try:
            arr = np.fromfile(out, dtype=np.float32).reshape(self.n_layer + 1, n, d)
        finally:
            out.unlink(missing_ok=True)
        print(f"[Gemma iGPU] {n} tokens encoded in {float(reply[3]) / 1000:.1f}s", flush=True)
        return torch.from_numpy(arr)


class ExternalGemmaStub(torch.nn.Module):
    """Stands in for the transformers Gemma model so the LTX text-encoder wrapper and the offloader keep working."""

    def __init__(self):
        super().__init__()
        self.register_buffer("_anchor", torch.zeros(1, dtype=torch.bfloat16), persistent=False)

    @property
    def device(self):
        return torch.device("cuda")


def encode_raw_external(tokenizer, text: str, dtype=torch.bfloat16):
    """Same result layout as transformers' output_hidden_states with LTX's left padding to 1024 tokens."""
    pairs = tokenizer.tokenize_with_weights(text)["gemma"]
    ids = [int(t[0]) for t in pairs]
    mask = [int(w[1]) for w in pairs]
    real = [i for i, m in zip(ids, mask) if m]
    if mask[-len(real):] != [1] * len(real):
        raise RuntimeError("expected left padding")
    try:
        hs = GemmaIGPU.get(len(real)).hidden_states(real)  # [L, n, d]
    finally:
        GemmaIGPU.release_soon()
    L, n, d = hs.shape
    full = torch.zeros(L, 1, len(ids), d, dtype=dtype, device="cuda")
    full[:, 0, len(ids) - n:] = hs.to("cuda", non_blocking=False).to(dtype)
    attention_mask = torch.tensor([mask], device="cuda")
    return tuple(full[i] for i in range(L)), attention_mask
