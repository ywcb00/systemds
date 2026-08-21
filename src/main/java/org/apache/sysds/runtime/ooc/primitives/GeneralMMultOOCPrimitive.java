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

import java.util.List;
import java.util.concurrent.CompletionException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.sysds.runtime.instructions.ooc.OOCStream;
import org.apache.sysds.runtime.instructions.ooc.OOCStreamable;
import org.apache.sysds.runtime.instructions.ooc.SubscribableTaskQueue;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.operators.AggregateBinaryOperator;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.ooc.cache.OOCCacheManager;
import org.apache.sysds.runtime.ooc.cache.OOCFuture;
import org.apache.sysds.runtime.ooc.memory.ManagedPayload;
import org.apache.sysds.runtime.ooc.memory.ReservationBudget;
import org.apache.sysds.runtime.ooc.planning.OOCAccessPattern;
import org.apache.sysds.runtime.ooc.planning.OOCStoreLayout;
import org.apache.sysds.runtime.ooc.store.CountingLiveness;
import org.apache.sysds.runtime.ooc.store.IndexedMaterializedStoreReader;
import org.apache.sysds.runtime.ooc.store.MaterializedStore;
import org.apache.sysds.runtime.ooc.store.StateTable;
import org.apache.sysds.runtime.ooc.store.StoreLease;
import org.apache.sysds.runtime.ooc.stream.StreamContext;
import org.apache.sysds.runtime.ooc.util.OOCInstructionUtils;
import org.apache.sysds.runtime.ooc.util.OOCUtils;

public final class GeneralMMultOOCPrimitive extends OOCPrimitive {
	private final OOCStreamable<IndexedMatrixValue> _output;
	private final AggregateBinaryOperator _multiply;
	private final BinaryOperator _plus;
	private final AtomicBoolean _sourceComplete = new AtomicBoolean();
	private final AtomicInteger _active = new AtomicInteger(1);
	private MaterializedStore<IndexedMatrixValue> _leftStore;
	private MaterializedStore<IndexedMatrixValue> _rightStore;
	private IndexedMaterializedStoreReader<IndexedMatrixValue> _leftReader;
	private IndexedMaterializedStoreReader<IndexedMatrixValue> _rightReader;
	private StateTable<IndexedMatrixValue> _accumulators;
	private OOCStream<AutoCloseable> _ready;
	private OOCStream<IndexedMatrixValue> _outputStream;
	private int _rowBlocks;
	private int _innerBlocks;
	private int _colBlocks;
	private int _nextTask;
	private int _numTasks;
	private long _taskBytes;

	public GeneralMMultOOCPrimitive(OOCStreamable<IndexedMatrixValue> left, OOCStreamable<IndexedMatrixValue> right,
		OOCStreamable<IndexedMatrixValue> output, AggregateBinaryOperator multiply, BinaryOperator plus,
		StreamContext context) {
		super(context, left, right);
		_output = output;
		_multiply = multiply;
		_plus = plus;
	}

	@Override
	public List<OOCMaterializedInputRequest> requiredMaterializedInputs() {
		return List.of(new OOCMaterializedInputRequest(0, OOCStoreLayout.ROW_MAJOR, 1),
			new OOCMaterializedInputRequest(1, OOCStoreLayout.COL_MAJOR, 1));
	}

	@Override
	protected void inferPatternsInternal() {
		_pattern = OOCAccessPattern.ANY;
		inferParentPatterns();
	}

	@Override
	protected void requestPatternInternal(OOCAccessPattern accessPattern) {
		_pattern = _pattern.preferred(accessPattern);
		OOCPrimitive left = getInputDependency(0);
		OOCPrimitive right = getInputDependency(1);
		if(left != null)
			left.requestPattern(OOCAccessPattern.ROW_MAJOR);
		if(right != null)
			right.requestPattern(OOCAccessPattern.COL_MAJOR);
	}

	@Override
	protected void startExecution() {
		DataCharacteristics left = getInput(0).getDataCharacteristics();
		DataCharacteristics right = getInput(1).getDataCharacteristics();
		_rowBlocks = Math.toIntExact(OOCUtils.getNumRowBlocks(left));
		_innerBlocks = Math.toIntExact(OOCUtils.getNumColBlocks(left));
		_colBlocks = Math.toIntExact(OOCUtils.getNumColBlocks(right));
		_numTasks = _rowBlocks * _innerBlocks * _colBlocks;
		long leftBytes = OOCUtils.estimateFullTileBytes(left);
		long rightBytes = OOCUtils.estimateFullTileBytes(right);
		long outputBytes = OOCUtils.estimateFullTileBytes(_output.getDataCharacteristics());
		_taskBytes = OOCCacheManager.getGlobalCache().maxPhysicalPinBytes(leftBytes) +
			OOCCacheManager.getGlobalCache().maxPhysicalPinBytes(rightBytes) +
			OOCCacheManager.getGlobalCache().maxPhysicalPinBytes(outputBytes) + outputBytes * 3;

		_outputStream = _output.getWriteStream();
		_ready = new SubscribableTaskQueue<>();
		_accumulators = new StateTable<>();
		getContext().addOutStream(_outputStream, _ready);
		OOCInstructionUtils.submitCloseableOOCTasks(_ready, this::process, getContext())
			.whenComplete((ignored, error) -> {
				try {
					if(error != null)
						fail(error);
					_outputStream.closeInput();
				}
				catch(Throwable failure) {
					fail(failure);
				}
				finally {
					cleanup();
				}
			});

		OOCFuture.allOf(List.of(getMaterializedInput(0), getMaterializedInput(1)), MaterializedStore::close)
			.whenComplete(this::storesReady);
	}

	private void storesReady(List<MaterializedStore<IndexedMatrixValue>> stores, Throwable error) {
		if(error != null) {
			fail(error);
			finishSource();
			return;
		}
		_leftStore = stores.get(0);
		_rightStore = stores.get(1);
		OOCFuture.allOf(List.of(_leftStore.completion(), _rightStore.completion()))
			.whenComplete((ignored, completionError) -> {
				if(completionError != null) {
					fail(completionError);
					finishSource();
					return;
				}
				_leftReader = _leftStore.openIndexedReader(new CountingLiveness(_leftStore.size(), _colBlocks));
				_rightReader = _rightStore.openIndexedReader(new CountingLiveness(_rightStore.size(), _rowBlocks));
				scheduleNext();
			});
	}

	private void scheduleNext() {
		while(true) {
			if(hasFailed() || _nextTask == _numTasks) {
				finishSource();
				return;
			}

			OOCFuture<?> reservation = _allowance.reserveAsync(_taskBytes);

			if(!reservation.isDone()) {
				// _nextTask ctr cannot be stale due to OOCFuture synchronization barrier
				reservation.whenComplete((ignored, error) -> {
					if(error != null) {
						fail(error);
						finishSource();
						return;
					}

					startTask();
					scheduleNext();
				});
				return;
			}

			try {
				reservation.getNow(null);
			}
			catch(CompletionException ex) {
				fail(ex.getCause());
				finishSource();
				return;
			}

			startTask();
		}
	}

	private void startTask() {
		ReservationBudget budget = new ReservationBudget(_allowance, _taskBytes).enableReuse();

		int task = _nextTask++;
		_active.incrementAndGet();
		requestInputs(task, budget);
	}

	private void requestInputs(int task, ReservationBudget budget) {
		int inner = task % _innerBlocks;
		int row;
		int col;
		if(_pattern == OOCAccessPattern.COL_MAJOR) {
			row = task / _innerBlocks % _rowBlocks;
			col = task / (_innerBlocks * _rowBlocks);
		}
		else {
			col = task / _innerBlocks % _colBlocks;
			row = task / (_innerBlocks * _colBlocks);
		}
		int leftIndex = row * _innerBlocks + inner;
		int rightIndex = col * _innerBlocks + inner;
		int outputSlot = row * _colBlocks + col;
		try {
			OOCFuture.allOf(List.of(_leftReader.request(leftIndex, budget), _rightReader.request(rightIndex, budget)),
				StoreLease::close).whenComplete((inputs, error) -> {
					if(error != null) {
						budget.close();
						fail(error);
						completeOne();
						return;
					}
					try {
						_ready.enqueue(new MultiplyWork(inputs.get(0), inputs.get(1), outputSlot, budget));
					}
					catch(Throwable failure) {
						inputs.forEach(StoreLease::close);
						budget.close();
						fail(failure);
						completeOne();
					}
				});
		}
		catch(Throwable failure) {
			budget.close();
			fail(failure);
			completeOne();
		}
	}

	private void process(AutoCloseable work) {
		if(work instanceof MultiplyWork multiply)
			multiply(multiply);
		else
			merge((MergeWork) work);
	}

	private void multiply(MultiplyWork work) {
		ReservationBudget budget = work.takeBudget();
		ManagedPayload<IndexedMatrixValue> partial = null;
		try {
			MatrixBlock left = (MatrixBlock) work._left.value().getValue();
			MatrixBlock right = (MatrixBlock) work._right.value().getValue();
			MatrixBlock block = left.aggregateBinaryOperations(left, right, new MatrixBlock(), _multiply);
			partial = payload(work._outputSlot, 1, block, budget);
			OOCFuture<List<Void>> released = work.releaseInputsAsync();
			ManagedPayload<IndexedMatrixValue> result = partial;
			partial = null;
			// wait for closure to not exceed reserved budget
			released.whenComplete((ignored, error) -> {
				if(error != null) {
					result.release();
					budget.close();
					fail(error);
					completeOne();
				}
				else
					reduce(work._outputSlot, result, budget);
			});
		}
		catch(Throwable failure) {
			if(partial != null)
				partial.release();
			budget.close();
			fail(failure);
			completeOne();
		}
	}

	private void reduce(int slot, ManagedPayload<IndexedMatrixValue> incoming, ReservationBudget budget) {
		if(count(incoming.value()) == _innerBlocks) {
			finalizeOutput(slot, incoming, budget);
			return;
		}
		OOCFuture<StoreLease<IndexedMatrixValue>> match;
		try {
			match = _accumulators.putOrTake(slot, incoming, budget);
		}
		catch(Throwable failure) {
			incoming.release();
			budget.close();
			fail(failure);
			completeOne();
			return;
		}
		match.whenComplete((existing, error) -> {
			if(error != null) {
				incoming.release();
				budget.close();
				fail(error);
				completeOne();
			}
			else if(existing == null) {
				budget.close();
				completeOne();
			}
			else {
				try {
					_ready.enqueue(new MergeWork(slot, incoming, existing, budget));
				}
				catch(Throwable failure) {
					incoming.release();
					existing.close();
					budget.close();
					fail(failure);
					completeOne();
				}
			}
		});
	}

	private void merge(MergeWork work) {
		ReservationBudget budget = work.takeBudget();
		ManagedPayload<IndexedMatrixValue> merged = null;
		try {
			IndexedMatrixValue existing = work._existing.value();
			IndexedMatrixValue incoming = work._incoming.value();
			MatrixBlock block = ((MatrixBlock) existing.getValue()).binaryOperations(_plus, incoming.getValue(),
				new MatrixBlock());
			merged = payload(work._slot, count(existing) + count(incoming), block, budget);
			work.releaseIncoming();
			OOCFuture<Void> released = work.closeExistingAsync();
			ManagedPayload<IndexedMatrixValue> result = merged;
			merged = null;
			released.whenComplete((ignored, error) -> {
				if(error != null) {
					result.release();
					budget.close();
					fail(error);
					completeOne();
				}
				else
					reduce(work._slot, result, budget);
			});
		}
		catch(Throwable failure) {
			if(merged != null)
				merged.release();
			budget.close();
			fail(failure);
			completeOne();
		}
	}

	private void finalizeOutput(int slot, ManagedPayload<IndexedMatrixValue> payload, ReservationBudget budget) {
		try {
			MatrixBlock block = (MatrixBlock) payload.value().getValue();
			payload.release();
			OOCUtils.enqueueExact(_outputStream,
				new IndexedMatrixValue(new MatrixIndexes(slot / _colBlocks + 1L, slot % _colBlocks + 1L), block),
				budget);
		}
		catch(Throwable failure) {
			payload.release();
			budget.close();
			fail(failure);
		}
		completeOne();
	}

	private static ManagedPayload<IndexedMatrixValue> payload(int slot, int count, MatrixBlock block,
		ReservationBudget budget) {
		long bytes = block.getExactSerializedSize();
		budget.reserveBlocking(bytes);
		return new ManagedPayload<>(new IndexedMatrixValue(new MatrixIndexes(slot + 1L, count), block), bytes, budget);
	}

	private static int count(IndexedMatrixValue value) {
		return Math.toIntExact(value.getIndexes().getColumnIndex());
	}

	private void finishSource() {
		if(_sourceComplete.compareAndSet(false, true))
			completeOne();
	}

	private void completeOne() {
		if(_active.decrementAndGet() != 0)
			return;
		try {
			_ready.closeInput();
		}
		catch(IllegalStateException ignored) {
		}
	}

	private void cleanup() {
		if(_accumulators != null)
			_accumulators.close();
		if(_leftReader != null)
			_leftReader.close();
		if(_rightReader != null)
			_rightReader.close();
		if(_leftStore != null)
			_leftStore.close();
		if(_rightStore != null)
			_rightStore.close();
		onComplete();
	}

	private static final class MultiplyWork implements AutoCloseable {
		private final int _outputSlot;
		private StoreLease<IndexedMatrixValue> _left;
		private StoreLease<IndexedMatrixValue> _right;
		private ReservationBudget _budget;

		private MultiplyWork(StoreLease<IndexedMatrixValue> left, StoreLease<IndexedMatrixValue> right, int outputSlot,
			ReservationBudget budget) {
			_left = left;
			_right = right;
			_outputSlot = outputSlot;
			_budget = budget;
		}

		private ReservationBudget takeBudget() {
			ReservationBudget budget = _budget;
			_budget = null;
			return budget;
		}

		private OOCFuture<List<Void>> releaseInputsAsync() {
			OOCFuture<Void> left = _left.closeAsync();
			OOCFuture<Void> right = _right.closeAsync();
			_left = null;
			_right = null;
			return OOCFuture.allOf(List.of(left, right));
		}

		@Override
		public void close() {
			if(_left != null)
				_left.close();
			if(_right != null)
				_right.close();
			if(_budget != null)
				_budget.close();
		}
	}

	private static final class MergeWork implements AutoCloseable {
		private final int _slot;
		private ManagedPayload<IndexedMatrixValue> _incoming;
		private StoreLease<IndexedMatrixValue> _existing;
		private ReservationBudget _budget;

		private MergeWork(int slot, ManagedPayload<IndexedMatrixValue> incoming,
			StoreLease<IndexedMatrixValue> existing, ReservationBudget budget) {
			_slot = slot;
			_incoming = incoming;
			_existing = existing;
			_budget = budget;
		}

		private ReservationBudget takeBudget() {
			ReservationBudget budget = _budget;
			_budget = null;
			return budget;
		}

		private void releaseIncoming() {
			_incoming.release();
			_incoming = null;
		}

		private OOCFuture<Void> closeExistingAsync() {
			OOCFuture<Void> released = _existing.closeAsync();
			_existing = null;
			return released;
		}

		@Override
		public void close() {
			if(_incoming != null)
				_incoming.release();
			if(_existing != null)
				_existing.close();
			if(_budget != null)
				_budget.close();
		}
	}
}
