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

import shutil
import unittest
from unittest.mock import patch

import numpy as np
import torch

from systemds.scuro.dataloader.base_loader import LazyFileSequence
from systemds.scuro.dataloader.image_loader import ImageLoader
from systemds.scuro.dataloader.video_loader import VideoLoader
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.modality.unimodal_modality import UnimodalModality
from systemds.scuro.representations.color_histogram import ColorHistogram
from systemds.scuro.representations.optical_flow import OpticalFlow
from systemds.scuro.representations.utils import flatten_owned_sequences
from systemds.scuro.utils.torch_dataset import CustomDataset
from tests.scuro.data_generator import setup_data


class TestLazyVisualLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_file_path = "test_lazy_visual_data"
        cls.data_generator = setup_data(
            [ModalityType.IMAGE, ModalityType.VIDEO],
            2,
            cls.test_file_path,
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_file_path, ignore_errors=True)

    def _loader(self, loader_type, indices=None, **kwargs):
        modality_type = (
            ModalityType.IMAGE if loader_type is ImageLoader else ModalityType.VIDEO
        )
        return loader_type(
            self.data_generator.get_modality_path(modality_type),
            indices or self.data_generator.indices,
            **kwargs,
        )

    def test_unchunked_load_keeps_only_file_references(self):
        for loader_type in (ImageLoader, VideoLoader):
            with self.subTest(loader=loader_type.__name__):
                loader = self._loader(loader_type)
                with patch.object(
                    loader, "_decode_file", wraps=loader._decode_file
                ) as decode:
                    data, metadata = loader.load()

                    self.assertIsInstance(data, LazyFileSequence)
                    self.assertEqual(len(data), 2)
                    self.assertEqual(len(metadata), 2)
                    decode.assert_not_called()

                    self.assertIsInstance(data[0], np.ndarray)
                    decode.assert_called_once()

    def test_custom_dataset_decodes_only_the_requested_batch(self):
        loader = self._loader(ImageLoader)
        with patch.object(loader, "_decode_file", wraps=loader._decode_file) as decode:
            data, _ = loader.load()
            dataset = CustomDataset(data, torch.float32, "cpu")
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

            first_batch = next(iter(dataloader))

            self.assertEqual(first_batch["data"].shape[0], 1)
            decode.assert_called_once()

    def test_modality_subset_preserves_lazy_file_references(self):
        loader = self._loader(ImageLoader)
        modality = UnimodalModality(loader)
        with patch.object(loader, "_decode_file", wraps=loader._decode_file) as decode:
            modality.extract_raw_data()
            subset = modality.subset([1])

            self.assertIsInstance(subset.data, LazyFileSequence)
            self.assertEqual(len(subset.data), 1)
            decode.assert_not_called()
            self.assertIsInstance(subset.data[0], np.ndarray)
            decode.assert_called_once()

    def test_unchunked_peak_memory_is_not_the_full_corpus(self):
        for loader_type in (ImageLoader, VideoLoader):
            loader = self._loader(loader_type)
            modality = UnimodalModality(loader)
            peak = modality.estimate_peak_memory_bytes()["cpu_peak_bytes"]
            total = modality.estimate_memory_bytes()
            self.assertLess(peak, total)

    def test_chunked_loading_still_returns_decoded_chunks(self):
        loader = self._loader(ImageLoader, chunk_size=1)

        first_data, first_metadata = loader.load()
        second_data, second_metadata = loader.load()

        self.assertIsInstance(first_data, list)
        self.assertNotIsInstance(first_data, LazyFileSequence)
        self.assertEqual(len(first_data), 1)
        self.assertEqual(len(first_metadata), 1)
        self.assertEqual(len(second_data), 1)
        self.assertEqual(len(second_metadata), 1)

    def test_histogram_streams_lazy_images_and_videos(self):
        for loader_type in (ImageLoader, VideoLoader):
            with self.subTest(loader=loader_type.__name__):
                loader = self._loader(loader_type)
                modality = UnimodalModality(loader)
                with patch.object(
                    loader, "_decode_file", wraps=loader._decode_file
                ) as decode:
                    transformed = modality.apply_representation(
                        ColorHistogram(bins=4, normalize=True)
                    )

                self.assertEqual(len(transformed.data), 2)
                self.assertEqual(decode.call_count, 2)

    def test_optical_flow_streams_one_lazy_video_at_a_time(self):
        loader = self._loader(VideoLoader, indices=self.data_generator.indices[:1])
        modality = UnimodalModality(loader)
        with patch.object(loader, "_decode_file", wraps=loader._decode_file) as decode:
            transformed = modality.apply_representation(OpticalFlow())

        self.assertEqual(len(transformed.data), 1)
        self.assertEqual(len(transformed.data[0]), loader.stats.max_length - 1)
        decode.assert_called_once()

    def test_flattened_frame_view_caches_only_the_current_owner(self):
        class CountingSequences:
            def __init__(self):
                self.values = [
                    np.arange(2).reshape(2, 1),
                    np.arange(3).reshape(3, 1),
                ]
                self.reads = []

            def __getitem__(self, index):
                self.reads.append(index)
                return self.values[index]

            def __len__(self):
                return len(self.values)

        sequences = CountingSequences()
        frames, owner_ids = flatten_owned_sequences(sequences, [2, 3])

        np.testing.assert_array_equal(frames[0], [0])
        np.testing.assert_array_equal(frames[1], [1])
        self.assertEqual(sequences.reads, [0])
        np.testing.assert_array_equal(frames[2], [0])
        self.assertEqual(sequences.reads, [0, 1])
        self.assertEqual(owner_ids, [0, 0, 1, 1, 1])
