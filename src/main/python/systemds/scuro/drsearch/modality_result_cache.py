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
from typing import Any, Dict, List, Optional

from systemds.scuro.drsearch.modality_shared_memory import unlink_shm
from systemds.scuro.utils.static_variables import DEBUG


class RefCountResultCache:
    def __init__(self):
        self.cache: Dict[str, Any] = {}
        self.ref_count: Dict[str, int] = {}
        self.memory_usage_per_node: Dict[str, int] = {}
        self.shared_memory_names: Dict[str, List[str]] = {}
        self._shm_retain_count: Dict[str, int] = {}

    def get(self, node_id: str) -> Any:
        return self.cache[node_id]

    def add_result(
        self,
        node_id: str,
        result: Any,
        shm_name: Optional[str] = None,
        resident_bytes: Optional[int] = None,
        shm_bytes: int = 0,
    ):
        if shm_name is not None:
            self.shared_memory_names[node_id] = [shm_name]
        self.cache[node_id] = result
        self.memory_usage_per_node[node_id] = int(resident_bytes or 0) + int(
            shm_bytes or 0
        )
        if DEBUG:
            print(
                f"Node {node_id} has a CPU memory usage of "
                f"{self.memory_usage_per_node[node_id]/1024**3:.5f} GB"
                + (
                    f" ({int(shm_bytes or 0)/1024**3:.5f} GB of it shared memory)"
                    if shm_name is not None
                    else ""
                )
            )

    def inc_ref(self, node_id: str):
        self.ref_count[node_id] = self.ref_count.get(node_id, 0) + 1

    def dec_ref(self, node_id: str):
        if node_id not in self.ref_count:
            return
        self.ref_count[node_id] -= 1
        if self.ref_count[node_id] <= 0:
            self.ref_count[node_id] = 0
            self._try_cleanup_node(node_id)

    def clear(self, node_id: str):
        self.ref_count[node_id] = 0
        self._try_cleanup_node(node_id)

    def retain_shm_names(self, shm_names: List[str]) -> List[str]:
        retained: List[str] = []
        for shm_name in shm_names:
            if not shm_name:
                continue
            self._shm_retain_count[shm_name] = (
                self._shm_retain_count.get(shm_name, 0) + 1
            )
            retained.append(shm_name)
        return retained

    def release_shm_names(self, shm_names: List[str]) -> None:
        nodes_to_recheck: List[str] = []
        for shm_name in shm_names:
            if not shm_name:
                continue
            count = self._shm_retain_count.get(shm_name, 0) - 1
            if count <= 0:
                self._shm_retain_count.pop(shm_name, None)
            else:
                self._shm_retain_count[shm_name] = count
            for node_id, node_names in self.shared_memory_names.items():
                if shm_name in node_names and node_id not in nodes_to_recheck:
                    nodes_to_recheck.append(node_id)
        for node_id in nodes_to_recheck:
            self._try_cleanup_node(node_id)

    def __len__(self):
        return len(self.cache)

    def get_memory_total_memory_usage(self):
        return sum(self.memory_usage_per_node.values())

    def _shm_names_in_use(self, shm_names: List[str]) -> bool:
        return any(self._shm_retain_count.get(name, 0) > 0 for name in shm_names)

    def _try_cleanup_node(self, node_id: str) -> None:
        if self.ref_count.get(node_id, 0) > 0:
            return
        shm_names = self.shared_memory_names.get(node_id, [])
        if shm_names and self._shm_names_in_use(shm_names):
            return
        self.cache.pop(node_id, None)
        self.ref_count.pop(node_id, None)
        self.memory_usage_per_node.pop(node_id, None)
        self._cleanup_shared_memory(node_id)

    def _cleanup_shared_memory(self, node_id: str):
        names = self.shared_memory_names.pop(node_id, [])
        for shm_name in names:
            unlink_shm(shm_name)

    def cleanup_all(self):
        self._shm_retain_count.clear()
        for node_id in list(self.shared_memory_names.keys()):
            self.ref_count.pop(node_id, None)
            self.cache.pop(node_id, None)
            self.memory_usage_per_node.pop(node_id, None)
            self._cleanup_shared_memory(node_id)
