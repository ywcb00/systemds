/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.runtime.ooc.util;

import org.apache.sysds.runtime.instructions.ooc.OOCStream;
import org.apache.sysds.runtime.ooc.cache.OOCFuture;
import org.apache.sysds.runtime.ooc.cache.io.SpillableObject;
import org.apache.sysds.runtime.ooc.memory.InMemoryQueueCallback;
import org.apache.sysds.runtime.ooc.memory.ManagedPayload;
import org.apache.sysds.runtime.ooc.memory.MemoryAllowance;
import org.apache.sysds.runtime.ooc.store.MaterializedCallback;
import org.apache.sysds.runtime.ooc.store.StateTable;
import org.apache.sysds.runtime.ooc.store.StoreLease;

public final class StateTableUtils {
	public static <T extends SpillableObject> OOCFuture<OOCStream.QueueCallback<T>> take(StateTable<T> table, int slot,
		MemoryAllowance allowance) {
		OOCFuture<StoreLease<T>> future = table.take(slot, allowance);
		OOCFuture<OOCStream.QueueCallback<T>> toReturn = new OOCFuture<>();
		future.whenComplete((l, err) -> {
			if(err != null)
				toReturn.completeExceptionally(err);
			else
				toReturn.complete(new MaterializedCallback<>(l));
		});
		return toReturn;
	}

	public static <T extends SpillableObject> void put(StateTable<T> table, int slot, OOCStream.QueueCallback<T> tile,
		MemoryAllowance allowance) {
		if(tile instanceof MaterializedCallback<T> pinned && pinned.pinnedEntry() != null) {
			table.putReference(slot, pinned.pinnedEntry());
			return;
		}
		ManagedPayload<T> payload;
		if(tile instanceof InMemoryQueueCallback<T> managed && managed.getManagedBytes() > 0)
			payload = managed.extractManagedPayload();
		else {
			T value = tile.get();
			long bytes = value.size();
			allowance.reserveBlocking(bytes);
			payload = new ManagedPayload<>(value, bytes, allowance);
		}
		try {
			table.put(slot, payload);
		}
		catch(RuntimeException error) {
			payload.release();
			throw error;
		}
	}

	public static <T extends SpillableObject> OOCFuture<Match<T>> putOrTake(StateTable<T> table, int slot,
		OOCStream.QueueCallback<T> tile, MemoryAllowance allowance) {
		if(tile instanceof MaterializedCallback<T> pinned && pinned.pinnedEntry() != null) {
			MaterializedCallback<T> retained = (MaterializedCallback<T>) pinned.keepOpen();
			pinned.close();
			return putReferenceOrTake(table, slot, retained, allowance);
		}
		ManagedPayload<T> payload;
		if(tile instanceof InMemoryQueueCallback<T> managed && managed.getManagedBytes() > 0) {
			payload = managed.extractManagedPayload();
			managed.close();
		}
		else {
			T value = tile.get();
			long bytes = value.size();
			allowance.reserveBlocking(bytes);
			payload = new ManagedPayload<>(value, bytes, allowance);
			tile.close();
		}
		OOCFuture<Match<T>> result = new OOCFuture<>();
		OOCFuture<StoreLease<T>> matched;
		try {
			matched = table.putOrTake(slot, payload, allowance);
		}
		catch(RuntimeException ex) {
			payload.release();
			return OOCFuture.failed(ex);
		}
		matched.whenComplete((lease, error) -> {
			if(error != null) {
				payload.release();
				result.completeExceptionally(error);
			}
			else if(lease == null)
				result.complete(null);
			else
				result.complete(
					new Match<>(new MaterializedCallback<>(StoreLease.create(payload.value(), payload::release)),
						new MaterializedCallback<>(lease)));
		});
		return result;
	}

	private static <T extends SpillableObject> OOCFuture<Match<T>> putReferenceOrTake(StateTable<T> table, int slot,
		MaterializedCallback<T> pinned, MemoryAllowance allowance) {
		OOCFuture<Match<T>> result = new OOCFuture<>();
		OOCFuture<StoreLease<T>> matched;
		try {
			matched = table.putReferenceOrTake(slot, pinned.pinnedEntry(), allowance);
		}
		catch(RuntimeException ex) {
			pinned.close();
			return OOCFuture.failed(ex);
		}
		matched.whenComplete((lease, error) -> {
			if(error != null) {
				pinned.close();
				result.completeExceptionally(error);
			}
			else if(lease == null) {
				pinned.close();
				result.complete(null);
			}
			else
				result.complete(new Match<>(pinned, new MaterializedCallback<>(lease)));
		});
		return result;
	}

	public record Match<T extends SpillableObject>(OOCStream.QueueCallback<T> left, OOCStream.QueueCallback<T> right) {
	}
}
