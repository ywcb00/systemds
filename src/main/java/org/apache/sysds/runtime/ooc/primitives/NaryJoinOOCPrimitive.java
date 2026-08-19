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

package org.apache.sysds.runtime.ooc.primitives;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.function.ToIntFunction;
import java.util.function.ToLongFunction;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.ooc.OOCStream;
import org.apache.sysds.runtime.instructions.ooc.OOCStreamable;
import org.apache.sysds.runtime.instructions.ooc.SubscribableTaskQueue;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.ooc.cache.OOCFuture;
import org.apache.sysds.runtime.ooc.memory.InMemoryQueueCallback;
import org.apache.sysds.runtime.ooc.memory.ReservationBudget;
import org.apache.sysds.runtime.ooc.planning.OOCAccessPattern;
import org.apache.sysds.runtime.ooc.store.StateTable;
import org.apache.sysds.runtime.ooc.store.StoreLease;
import org.apache.sysds.runtime.ooc.stream.StreamContext;
import org.apache.sysds.runtime.ooc.util.OOCInstructionUtils;
import org.apache.sysds.runtime.ooc.util.OOCUtils;
import org.apache.sysds.runtime.ooc.util.StateTableUtils;

public final class NaryJoinOOCPrimitive extends OOCPrimitive {
	private final List<OOCStreamable<IndexedMatrixValue>> _inputs;
	private final OOCStreamable<IndexedMatrixValue> _output;
	private final ToIntFunction<IndexedMatrixValue> _key;
	private final ToLongFunction<IndexedMatrixValue> _size;
	private final Function<List<IndexedMatrixValue>, IndexedMatrixValue> _operation;
	private final long _storeTaskBytes;
	private final long _joinTaskBytes;
	private final AtomicInteger _active = new AtomicInteger(1);
	private final CompletableFuture<Void> _activeCompletion = new CompletableFuture<>();
	private StateTable<IndexedMatrixValue> _table;
	private OOCStream<JoinWork> _ready;
	private OOCStream<IndexedMatrixValue> _outputStream;

	public NaryJoinOOCPrimitive(List<OOCStreamable<IndexedMatrixValue>> inputs,
		OOCStreamable<IndexedMatrixValue> output, ToIntFunction<IndexedMatrixValue> key,
		ToLongFunction<IndexedMatrixValue> size, Function<List<IndexedMatrixValue>, IndexedMatrixValue> operation,
		long storeTaskBytes, long joinTaskBytes, StreamContext context) {
		super(context, inputs.toArray(OOCStreamable[]::new));
		if(inputs.size() < 2)
			throw new IllegalArgumentException("N-ary join requires at least two inputs.");
		_inputs = inputs;
		_output = output;
		_key = key;
		_size = size;
		_operation = operation;
		_storeTaskBytes = storeTaskBytes;
		_joinTaskBytes = joinTaskBytes;
	}

	@Override
	protected void inferPatternsInternal() {
		_pattern = OOCAccessPattern.ANY;
		for(OOCPrimitive child : getChildren())
			_pattern = _pattern.fused(child.getAccessPattern());
		if(_pattern.isPlannable() && _pattern != OOCAccessPattern.ANY)
			for(OOCPrimitive child : getChildren())
				child.requestPattern(_pattern);
		inferParentPatterns();
	}

	@Override
	protected void requestPatternInternal(OOCAccessPattern accessPattern) {
		_pattern = accessPattern;
		for(OOCPrimitive child : getChildren())
			child.requestPattern(accessPattern);
	}

	@Override
	protected void startExecution() {
		List<OOCStream<IndexedMatrixValue>> inputs = new ArrayList<>(_inputs.size());
		for(int i = 0; i < _inputs.size(); i++)
			inputs.add(getInputReadStream(i));

		int groups = (int) OOCUtils.getNumBlocks(inputs.get(0).getDataCharacteristics());
		_table = new StateTable<>(groups * inputs.size());
		_outputStream = _output.getWriteStream();
		_ready = new SubscribableTaskQueue<>();
		getContext().addOutStream(_outputStream, _ready);
		CompletableFuture<Void> processing = OOCInstructionUtils.submitCloseableOOCTasks(_ready, this::process,
			getContext());
		CompletableFuture.allOf(processing, _activeCompletion).whenComplete((ignored, error) -> {
			if(error != null)
				fail(error);
			try {
				_outputStream.closeInput();
			}
			catch(Throwable failure) {
				fail(failure);
			}
			finally {
				onComplete();
			}
		});
		OOCInstructionUtils.submitOOCTask(() -> drive(inputs), new StreamContext().addOutStream(_outputStream));
	}

	private void drive(List<OOCStream<IndexedMatrixValue>> inputs) {
		try {
			byte[] groupCtr = new byte[(int) OOCUtils.getNumBlocks(inputs.get(0).getDataCharacteristics())];
			int n = inputs.size();
			int unmatchedGroups = 0;
			while(true) {
				List<OOCStream.QueueCallback<IndexedMatrixValue>> callbacks = new ArrayList<>(n);
				try {
					int eos = 0;
					for(OOCStream<IndexedMatrixValue> input : inputs) {
						OOCStream.QueueCallback<IndexedMatrixValue> callback = input.dequeueCB();
						callbacks.add(callback);
						if(callback == null || callback.isEos())
							eos++;
					}
					if(eos != 0) {
						if(eos != n)
							throw new DMLRuntimeException("Join inputs contain a different number of blocks");
						if(unmatchedGroups != 0)
							throw new DMLRuntimeException("Join inputs contain unmatched blocks");
						break;
					}

					for(int i = 0; i < n; i++) {
						OOCStream.QueueCallback<IndexedMatrixValue> callback = callbacks.get(i);
						int group = _key.applyAsInt(callback.get());
						int count = ++groupCtr[group];
						if(count == 1)
							unmatchedGroups++;
						if(count == n) {
							unmatchedGroups--;
							onJoinGroupAvailable(group, callback, i, n);
						}
						else {
							ReservationBudget budget = OOCUtils.reserveBudget(_allowance, _storeTaskBytes);
							try {
								StateTableUtils.put(_table, group * n + i, callback, budget);
							}
							finally {
								budget.close();
							}
						}
					}
				}
				finally {
					for(OOCStream.QueueCallback<IndexedMatrixValue> callback : callbacks)
						if(callback != null)
							callback.close();
				}
			}
		}
		catch(Throwable failure) {
			fail(failure);
			throw DMLRuntimeException.of(failure);
		}
		finally {
			completeActive();
		}
	}

	private void onJoinGroupAvailable(int group, OOCStream.QueueCallback<IndexedMatrixValue> callback,
		int callbackIndex, int n) {
		ReservationBudget budget = null;
		OOCStream.QueueCallback<IndexedMatrixValue> anchor = null;
		boolean active = false;
		try {
			budget = OOCUtils.reserveBudget(_allowance, _joinTaskBytes);
			anchor = callback.keepOpen();
			_active.incrementAndGet();
			active = true;
			List<OOCFuture<StoreLease<IndexedMatrixValue>>> futures = new ArrayList<>(n - 1);
			try {
				for(int i = 0; i < n; i++)
					if(i != callbackIndex)
						futures.add(_table.take(group * n + i, budget).map(lease -> {
							if(lease == null)
								throw new DMLRuntimeException("Join input block is missing");
							return lease;
						}));
			}
			catch(Throwable failure) {
				futures.add(OOCFuture.failed(failure));
			}
			OOCFuture<List<StoreLease<IndexedMatrixValue>>> leases = OOCFuture.allOf(futures, StoreLease::close);
			OOCStream.QueueCallback<IndexedMatrixValue> pendingAnchor = anchor;
			ReservationBudget pendingBudget = budget;
			anchor = null;
			budget = null;
			active = false;
			leases.whenComplete(
				(values, error) -> onJoinReady(pendingAnchor, callbackIndex, values, pendingBudget, error));
		}
		finally {
			if(anchor != null)
				anchor.close();
			if(budget != null)
				budget.close();
			if(active)
				completeActive();
		}
	}

	private void onJoinReady(OOCStream.QueueCallback<IndexedMatrixValue> anchor, int anchorIndex,
		List<StoreLease<IndexedMatrixValue>> leases, ReservationBudget budget, Throwable error) {
		JoinWork work = null;
		try {
			if(error != null)
				throw DMLRuntimeException.of(error);
			work = new JoinWork(anchor, anchorIndex, leases, budget);
			_ready.enqueue(work);
			work = null;
		}
		catch(Throwable failure) {
			fail(failure);
		}
		finally {
			if(work != null)
				work.close();
			if(error != null) {
				anchor.close();
				budget.close();
			}
			completeActive();
		}
	}

	private void process(JoinWork work) {
		List<IndexedMatrixValue> values = new ArrayList<>(work._leases.size() + 1);
		int lease = 0;
		for(int i = 0; i <= work._leases.size(); i++)
			values.add(i == work._anchorIndex ? work._anchor.get() : work._leases.get(lease++).value());
		IndexedMatrixValue output = _operation.apply(values);
		long bytes = _size.applyAsLong(output);
		work._budget.reserveBlocking(bytes);
		OOCStream.QueueCallback<IndexedMatrixValue> callback = new InMemoryQueueCallback<>(output, null, work._budget,
			bytes);
		try {
			_outputStream.enqueue(callback);
			callback = null;
		}
		finally {
			if(callback != null)
				callback.close();
		}
	}

	private void completeActive() {
		if(_active.decrementAndGet() != 0)
			return;
		try {
			_table.close();
			try {
				_ready.closeInput();
			}
			catch(IllegalStateException ignored) {
			}
		}
		finally {
			_activeCompletion.complete(null);
		}
	}

	private static final class JoinWork implements AutoCloseable {
		private final OOCStream.QueueCallback<IndexedMatrixValue> _anchor;
		private final int _anchorIndex;
		private final List<StoreLease<IndexedMatrixValue>> _leases;
		private final ReservationBudget _budget;

		private JoinWork(OOCStream.QueueCallback<IndexedMatrixValue> anchor, int anchorIndex,
			List<StoreLease<IndexedMatrixValue>> leases, ReservationBudget budget) {
			_anchor = anchor;
			_anchorIndex = anchorIndex;
			_leases = leases;
			_budget = budget;
		}

		@Override
		public void close() {
			_anchor.close();
			for(StoreLease<IndexedMatrixValue> lease : _leases)
				lease.close();
			_budget.close();
		}
	}
}
