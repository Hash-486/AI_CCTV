# ============================================================
# embedder.py -- Batched OSNet appearance embedding
#
# One forward pass per frame for all person crops, rather than one
# pass per crop. This is the single largest speed win over the base
# project, which called torchreid's FeatureExtractor per crop.
#
# FeatureExtractor is deliberately not used here. Its __call__ does
# numpy -> PIL -> T.Resize -> T.ToTensor -> T.Normalize for every
# crop before stacking, which is CPU-bound, single-threaded, and
# offers no fp16 path. Preprocessing with cv2 into a preallocated
# batch tensor avoids the PIL round trip entirely.
#
# The same embedding serves two consumers:
#   1. DeepSORT's cosine metric (short-term frame-to-frame association)
#   2. The identity gallery (long-term re-identification)
# Sharing one re-ID-trained representation means both timescales agree
# on what "the same person" means. In the base project they did not:
# MobileNetV2 decided short-term association and could silently swap
# two identities, after which the OSNet gallery had no mechanism to
# notice, and kept appending the wrong person's features to the entry.
# ============================================================
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from config import (
    TORCHREID_PATH,
    REID_MODEL_NAME,
    REID_MODEL_PATH,
    REID_IMAGE_SIZE,
    REID_HALF,
    REID_MIN_CROP_H,
    REID_MIN_CROP_W,
    REID_PIXEL_MEAN,
    REID_PIXEL_STD,
    USE_CUDA_GRAPH,
    GRAPH_BATCH,
)

# The vendored torchreid is not pip-installed; it lives in the parent project.
if TORCHREID_PATH not in sys.path and os.path.isdir(TORCHREID_PATH):
    sys.path.insert(0, TORCHREID_PATH)

from torchreid.models import build_model  # noqa: E402
from torchreid.utils import load_pretrained_weights  # noqa: E402


class OSNetEmbedder:
    """Extracts L2-normalised appearance embeddings for person crops."""

    def __init__(
        self,
        model_name=REID_MODEL_NAME,
        model_path=REID_MODEL_PATH,
        image_size=REID_IMAGE_SIZE,
        device=None,
        half=REID_HALF,
        use_cuda_graph=USE_CUDA_GRAPH,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.half = half and self.device.type == "cuda"
        self.h, self.w = image_size
        self.model_name = model_name

        # pretrained=True here means ImageNet, which is NOT what we want.
        # Build without it when a re-ID checkpoint is available, then load
        # the re-ID weights explicitly. Falling back to ImageNet is allowed
        # but loudly warned about, because it silently destroys re-ID quality.
        has_reid_weights = bool(model_path) and os.path.isfile(model_path)
        model = build_model(
            model_name,
            num_classes=1,
            pretrained=not has_reid_weights,
            use_gpu=self.device.type == "cuda",
        )
        if has_reid_weights:
            load_pretrained_weights(model, model_path)
            self.weights = os.path.basename(model_path)
        else:
            self.weights = "IMAGENET (no re-ID weights found)"
            print(
                f"[OSNetEmbedder] WARNING: no re-ID checkpoint at {model_path!r}. "
                "Falling back to ImageNet weights, which are not trained for "
                "person re-identification and will substantially degrade "
                "identity matching. Run tools/fetch_reid_weights.py."
            )

        model.eval()
        model.to(self.device)
        if self.half:
            model.half()
        self.model = model

        # Normalisation constants kept on-device as (1, 3, 1, 1) so the
        # whole batch is normalised in one broadcast op.
        dtype = torch.float16 if self.half else torch.float32
        self.mean = torch.tensor(
            REID_PIXEL_MEAN, device=self.device, dtype=dtype
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            REID_PIXEL_STD, device=self.device, dtype=dtype
        ).view(1, 3, 1, 1)

        self.dtype = torch.float16 if self.half else torch.float32
        self.dim = self._probe_dim()

        # CUDA graph state. See _build_graph() for why this exists.
        self.graph_batch = GRAPH_BATCH
        self._graph = None
        self._static_in = None
        self._static_out = None
        if use_cuda_graph and self.device.type == "cuda":
            self._build_graph()

    def _probe_dim(self):
        """Determine the embedding width by running one dummy crop through."""
        dummy = torch.zeros(
            1, 3, self.h, self.w, device=self.device, dtype=self.dtype
        )
        with torch.no_grad():
            out = self.model(dummy)
        return int(out.shape[1])

    def _build_graph(self):
        """Capture the forward pass as a replayable CUDA graph.

        OSNet is kernel-launch bound, not compute bound. Its omni-scale
        blocks are four parallel branches of depthwise-separable convs,
        repeated -- an enormous number of individually tiny operations.
        Measured on an RTX 5070 at 256x128:

            batch 1   12.47 ms      osnet_x1_0    13.61 ms
            batch 8   13.30 ms      osnet_x0_25   13.03 ms

        Eight times the work for six percent more time, and a model with
        sixteen times fewer parameters running at the same speed. Both say
        the GPU is idle waiting for launches, so neither a smaller model
        nor a smaller batch helps. Capturing the whole forward as one graph
        replaces thousands of launches with one: 12.91 ms -> 2.66 ms,
        bit-identical output.

        A fixed batch size is required because a graph records exact shapes.
        Padding to GRAPH_BATCH is nearly free for the same reason the
        speedup exists -- unused rows cost launch overhead we are already
        paying. Batches larger than GRAPH_BATCH are processed in chunks.
        """
        try:
            self._static_in = torch.zeros(
                self.graph_batch, 3, self.h, self.w,
                device=self.device, dtype=self.dtype,
            )
            # Warm up on a side stream first. Capturing straight away can
            # record lazily-initialised cuDNN workspace allocations into the
            # graph, which then replay against freed memory.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                with torch.no_grad():
                    for _ in range(3):
                        self.model(self._static_in)
            torch.cuda.current_stream().wait_stream(stream)

            graph = torch.cuda.CUDAGraph()
            with torch.no_grad():
                with torch.cuda.graph(graph):
                    self._static_out = self.model(self._static_in)
            self._graph = graph
        except Exception as exc:  # pragma: no cover - hardware dependent
            print(f"[OSNetEmbedder] CUDA graph capture failed ({exc}); "
                  "falling back to eager execution.")
            self._graph = None
            self._static_in = None
            self._static_out = None

    def _forward(self, tensor):
        """Run the model, via CUDA graph replay when possible."""
        if self._graph is None:
            with torch.no_grad():
                return self.model(tensor)

        n = tensor.shape[0]
        outs = []
        for start in range(0, n, self.graph_batch):
            chunk = tensor[start:start + self.graph_batch]
            k = chunk.shape[0]
            self._static_in[:k].copy_(chunk)
            if k < self.graph_batch:
                self._static_in[k:].zero_()
            self._graph.replay()
            outs.append(self._static_out[:k].clone())
        return torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]

    def _valid(self, box, frame_h, frame_w):
        """A crop is usable only if it is large enough to carry appearance."""
        x1, y1, x2, y2 = box
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(frame_w, int(x2)); y2 = min(frame_h, int(y2))
        if x2 - x1 < REID_MIN_CROP_W or y2 - y1 < REID_MIN_CROP_H:
            return None
        return x1, y1, x2, y2

    def embed(self, frame, boxes):
        """Embed every box in one batched forward pass.

        Args:
            frame: BGR uint8 image (H, W, 3)
            boxes: (N, 4) array of [x1, y1, x2, y2]

        Returns:
            feats: (N, D) float32, L2-normalised. Rows for rejected crops
                   are all zeros.
            valid: (N,) bool. False where the crop was too small to embed.
                   Callers must not admit invalid rows to the gallery.
        """
        n = len(boxes)
        if n == 0:
            return np.zeros((0, self.dim), np.float32), np.zeros((0,), bool)

        frame_h, frame_w = frame.shape[:2]
        feats = np.zeros((n, self.dim), np.float32)
        valid = np.zeros((n,), bool)

        crops = []
        keep_idx = []
        for i, box in enumerate(boxes):
            rect = self._valid(box, frame_h, frame_w)
            if rect is None:
                continue
            x1, y1, x2, y2 = rect
            crop = frame[y1:y2, x1:x2]
            # cv2.resize takes (width, height); OSNet wants 256 tall x 128 wide.
            crop = cv2.resize(crop, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            crops.append(crop)
            keep_idx.append(i)

        if not crops:
            return feats, valid

        out = self._embed_resized(crops)
        for row, i in enumerate(keep_idx):
            feats[i] = out[row]
            valid[i] = True

        return feats, valid

    def _embed_resized(self, crops):
        """Embed a list of crops already resized to (self.h, self.w)."""
        # (B, H, W, 3) BGR uint8 -> (B, 3, H, W) RGB normalised float
        batch = np.stack(crops, axis=0)[:, :, :, ::-1]      # BGR -> RGB
        batch = np.ascontiguousarray(batch.transpose(0, 3, 1, 2))
        tensor = torch.from_numpy(batch).to(self.device, non_blocking=True)
        tensor = tensor.half() if self.half else tensor.float()
        tensor = tensor.div_(255.0).sub_(self.mean).div_(self.std)

        with torch.no_grad():
            out = self._forward(tensor)
            out = F.normalize(out.float(), p=2, dim=1)
        return out.cpu().numpy().astype(np.float32)

    def embed_images(self, images):
        """Embed whole pre-cropped images in one batched pass.

        For datasets like Market-1501 the images ARE the crops, so going
        through embed() with a full-image box would run one forward pass
        per image and throw away the batching entirely.
        """
        if not images:
            return np.zeros((0, self.dim), np.float32)
        crops = [
            cv2.resize(im, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            for im in images
        ]
        return self._embed_resized(crops)
