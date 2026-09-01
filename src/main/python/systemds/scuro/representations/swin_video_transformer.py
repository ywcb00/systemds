# -------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# -------------------------------------------------------------
from torchvision.models.video.swin_transformer import swin3d_t

from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.representations.unimodal import UnimodalRepresentation
from systemds.scuro.representations.representation import RepresentationStats
from typing import Callable, Dict, Tuple, Any
import torch.utils.data
import torch
import torchvision.models as models
import numpy as np
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.drsearch.operator_registry import (
    register_representation,
    register_expensive_representation,
)
from systemds.scuro.dataloader.video_loader import VideoStats
from systemds.scuro.representations.utils import (
    LengthBucketBatchSampler,
    OwnerAccumulator,
    get_sequence_lengths,
    move_batch_to_device,
    pin_memory_for,
    transformer_inference_context,
)

from systemds.scuro.utils.torch_dataset import CustomDataset
from systemds.scuro.utils.static_variables import (
    get_device,
    get_device_for_model,
)


@register_representation([ModalityType.VIDEO])
@register_expensive_representation([ModalityType.VIDEO])
class SwinVideoTransformer(UnimodalRepresentation):
    _EMBED_DIM = 768
    cache_in_worker = True

    def __init__(self, layer_name="avgpool", batch_size=8, params=None):
        parameters = {
            "layer_name": [
                "features",
                "features.1",
                "features.2",
                "features.3",
                "features.4",
                "features.5",
                "features.6",
                "avgpool",
            ],
            # "batch_size": [1, 2, 4, 8, 16, 32],
        }
        self.data_type = torch.float32
        super().__init__("SwinVideoTransformer", ModalityType.EMBEDDING, parameters)
        if params is not None:
            layer_name = params.get("layer_name", layer_name)
            batch_size = int(params.get("batch_size", batch_size))
        self.layer_name = layer_name
        self.batch_size = batch_size
        self.model = swin3d_t(weights=models.video.Swin3D_T_Weights.KINETICS400_V1)
        self.device = get_device_for_model(self.model, memory_factor=1.5)
        self._gpu_id = self.device.index
        self._activation_hook = None
        self.model = self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @property
    def gpu_id(self):
        return self._gpu_id

    @gpu_id.setter
    def gpu_id(self, gpu_id):
        self._gpu_id = gpu_id
        self.device = get_device(gpu_id)

    def get_output_stats(self, input_stats) -> RepresentationStats:
        num_instances = getattr(input_stats, "num_instances", 0)
        return RepresentationStats(num_instances, (self._EMBED_DIM,))

    def estimate_output_memory_bytes(self, input_stats: VideoStats) -> int:
        dt = int(torch.tensor([], dtype=self.data_type).element_size())
        return input_stats.num_instances * self._EMBED_DIM * dt

    def estimate_peak_memory_bytes(self, input_stats: VideoStats) -> dict:
        dt = int(torch.tensor([], dtype=self.data_type).element_size())
        temporal = max(input_stats.max_length, 1)
        input_bytes = (
            dt
            * input_stats.max_channels
            * temporal
            * input_stats.max_height
            * input_stats.max_width
        )
        output_bytes = self.estimate_output_memory_bytes(input_stats)
        n = max(input_stats.num_instances, 1)
        output_bytes_batch = output_bytes / n

        batch_peak_bytes = (input_bytes + self._EMBED_DIM * dt) * 2

        safety_margin_bytes = 100 * 1024 * 1024

        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_bytes = param_size + buffer_size

        cpu_peak = (
            size_all_bytes * 2 * dt
            + output_bytes_batch
            + output_bytes
            + input_bytes
            + safety_margin_bytes
        )
        gpu_peak = (size_all_bytes * dt + batch_peak_bytes) * 6
        return {"cpu_peak_bytes": int(cpu_peak), "gpu_peak_bytes": int(gpu_peak)}

    def transform(self, modality, aggregation=None):
        self.model = self.model.to(self.device)
        self.model.eval()
        self.swin_output = None

        def get_features(name_):
            def hook(
                _module: torch.nn.Module, input_: Tuple[torch.Tensor], output: Any
            ):
                self.swin_output = output

            return hook

        if self.layer_name and self._activation_hook is None:
            for name, layer in self.model.named_modules():
                if name == self.layer_name:
                    self._activation_hook = layer.register_forward_hook(
                        get_features(name)
                    )
                    break

        dataset = CustomDataset(modality.data, self.data_type, "cpu")
        lengths = get_sequence_lengths(modality.data, modality.metadata)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=LengthBucketBatchSampler(
                lengths, self.batch_size, exact=True
            ),
            pin_memory=pin_memory_for(self.device),
        )
        accumulator = OwnerAccumulator(len(dataset), len(dataset), aggregation)

        with transformer_inference_context(self.device):
            for batch in dataloader:
                batch = move_batch_to_device(batch, self.device)
                video_ids = batch["id"].long()
                frames = batch["data"].permute(0, 2, 1, 3, 4)
                _ = self.model(frames)
                values = self.swin_output
                if isinstance(values, tuple):
                    values = values[0]
                if values.ndim == 2:
                    pooled = values
                elif self.layer_name.startswith("features"):
                    pooled = values.mean(dim=tuple(range(1, values.ndim - 1)))
                else:
                    pooled = values.mean(dim=tuple(range(2, values.ndim)))
                accumulator.update(pooled, video_ids, video_ids)

        transformed_modality = TransformedModality(
            modality, self, self.output_modality_type
        )
        transformed_modality.data = accumulator.finalize()
        transformed_modality.data_type = np.float32
        return transformed_modality
