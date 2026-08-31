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
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from systemds.scuro.modality.type import ModalityType
from systemds.scuro.representations.aggregated_representation import (
    AggregatedRepresentation,
)
from systemds.scuro.representations.clip import CLIPVisual
from systemds.scuro.representations.resnet import ResNet
from systemds.scuro.representations.vgg import VGG19


class _ResNetModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.avgpool = torch.nn.Identity()

    def forward(self, images):
        values = images.flatten(2).mean(dim=2)[:, :2] * self.scale
        return self.avgpool(values[:, :, None, None])


class _VGGModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.classifier = torch.nn.Sequential(torch.nn.Identity())

    def forward(self, images):
        values = images.flatten(2).mean(dim=2)[:, :2] * self.scale
        return self.classifier[0](values)


def _video_modality(videos):
    height, width, channels = videos[0][0].shape
    return SimpleNamespace(
        modality_type=ModalityType.VIDEO,
        modality_id=0,
        metadata=[
            ModalityType.VIDEO.create_metadata(30, len(video), width, height, channels)
            for video in videos
        ],
        data_type=np.float32,
        transform_time=0,
        data=videos,
    )


def _representation(representation_class, batch_size):
    representation = object.__new__(representation_class)
    representation.data_type = torch.float32
    representation.device = torch.device("cpu")
    representation.batch_size = batch_size
    representation._activation_hook = None
    representation.activation = None
    representation.output_modality_type = ModalityType.EMBEDDING
    if representation_class is ResNet:
        representation.model = _ResNetModel()
        representation.layer_name = "avgpool"
    else:
        representation.model = _VGGModel()
        representation.layer_name = "classifier.0"
    return representation


class TestNeuralEncoderBatching(unittest.TestCase):
    def setUp(self):
        self.videos = [
            [
                np.full((8, 8, 3), 32, dtype=np.uint8),
                np.full((8, 8, 3), 64, dtype=np.uint8),
            ],
            [
                np.full((8, 8, 3), 96, dtype=np.uint8),
                np.full((8, 8, 3), 128, dtype=np.uint8),
                np.full((8, 8, 3), 160, dtype=np.uint8),
            ],
        ]
        self.aggregation = AggregatedRepresentation("mean")

    def test_global_frame_batching_is_invariant_to_batch_size(self):
        for representation_class in (ResNet, VGG19):
            with self.subTest(representation=representation_class.__name__):
                results = [
                    _representation(representation_class, batch_size)
                    .transform(_video_modality(self.videos), self.aggregation)
                    .data
                    for batch_size in (1, 4)
                ]
                self.assertEqual(results[0].shape, (len(self.videos), 2))
                np.testing.assert_allclose(results[1], results[0])

    def test_global_frame_batching_matches_patient_by_patient(self):
        for representation_class in (ResNet, VGG19):
            with self.subTest(representation=representation_class.__name__):
                global_result = (
                    _representation(representation_class, 3)
                    .transform(_video_modality(self.videos), self.aggregation)
                    .data
                )
                per_patient = np.concatenate(
                    [
                        _representation(representation_class, 3)
                        .transform(_video_modality([video]), self.aggregation)
                        .data
                        for video in self.videos
                    ],
                    axis=0,
                )
                np.testing.assert_allclose(global_result, per_patient)

    def test_frame_encoders_support_aggregation_pushdown(self):
        input_stats = SimpleNamespace(num_instances=3)
        for representation_class, hidden_dim in (
            (ResNet, 512),
            (VGG19, 4096),
            (CLIPVisual, 512),
        ):
            with self.subTest(representation=representation_class.__name__):
                representation = object.__new__(representation_class)
                representation.params = {"_pushdown_aggregation": {}}
                representation.data_type = torch.float32

                self.assertTrue(representation.supports_aggregation_pushdown)
                stats = representation.get_output_stats(input_stats)
                self.assertEqual(stats.num_instances, input_stats.num_instances)
                self.assertEqual(stats.output_shape, (hidden_dim,))
                self.assertIsNone(stats.aggregate_dim)


if __name__ == "__main__":
    unittest.main()
