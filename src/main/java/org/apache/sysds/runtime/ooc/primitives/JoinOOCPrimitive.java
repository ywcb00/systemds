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

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiFunction;
import java.util.function.ToIntFunction;
import java.util.function.ToLongFunction;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.ooc.CachingStream;
import org.apache.sysds.runtime.instructions.ooc.OOCStream;
import org.apache.sysds.runtime.instructions.ooc.OOCStreamable;
import org.apache.sysds.runtime.instructions.ooc.SubscribableTaskQueue;
import org.apache.sysds.runtime.ooc.cache.OOCCacheManager;
import org.apache.sysds.runtime.ooc.cache.OOCFuture;
import org.apache.sysds.runtime.ooc.cache.io.SpillableObject;
import org.apache.sysds.runtime.ooc.memory.InMemoryQueueCallback;
import org.apache.sysds.runtime.ooc.memory.ReservationBudget;
import org.apache.sysds.runtime.ooc.planning.OOCAccessPattern;
import org.apache.sysds.runtime.ooc.store.StateTable;
import org.apache.sysds.runtime.ooc.stream.StreamContext;
import org.apache.sysds.runtime.ooc.util.OOCInstructionUtils;
import org.apache.sysds.runtime.ooc.util.OOCUtils;
import org.apache.sysds.runtime.ooc.util.StateTableUtils;

public class JoinOOCPrimitive<L extends SpillableObject, R extends SpillableObject, O> extends OOCPrimitive {
	private final OOCStreamable<O> _output;
	private final ToIntFunction<L> _leftKey;
	private final ToIntFunction<R> _rightKey;
	private final ToLongFunction<O> _outputSize;
	private final BiFunction<L, R, O> _operation;
	private final long _taskBytes;
	private final AtomicInteger _pending = new AtomicInteger(1);
	private final AtomicInteger _unmatched = new AtomicInteger();
	private final CompletableFuture<Void> _pendingCompletion = new CompletableFuture<>();
	private StateTable<SpillableObject> _table;
	private OOCStream<O> _outputStream;

	public JoinOOCPrimitive(OOCStreamable<L> left, OOCStreamable<R> right, OOCStreamable<O> output,
		ToIntFunction<L> leftKey, ToIntFunction<R> rightKey, ToLongFunction<O> outputSize,
		BiFunction<L, R, O> operation, long taskBytes, StreamContext context) {
		super(context, left, right);
		_output = output;
		_leftKey = leftKey;
		_rightKey = rightKey;
		_outputSize = outputSize;
		_operation = operation;
		_taskBytes = taskBytes;
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
		OOCStream<L> left = getInputReadStream(0);
		OOCStream<R> right = getInputReadStream(1);
		_table = new StateTable<>(OOCCacheManager.getGlobalCache(), CachingStream._streamSeq.getNextID());
		_outputStream = _output.getWriteStream();
		OOCStream<JoinWork> matches = new SubscribableTaskQueue<>();

		getContext().addOutStream(_outputStream);
		CompletableFuture<Void> processing = OOCInstructionUtils.submitCloseableOOCTasks(matches, this::process,
			getContext());
		CompletableFuture.allOf(processing, _pendingCompletion).thenRun(() -> {
			try {
				_table.close();
				onComplete();
			}
			finally {
				_outputStream.closeInput();
			}
		});

		OOCInstructionUtils.submitOOCTask(() -> drive(left, right, matches),
			new StreamContext().addOutStream(_outputStream));
	}

	private void drive(OOCStream<L> leftInput, OOCStream<R> rightInput, OOCStream<JoinWork> matches) {
		try {
			while(true) {
				OOCStream.QueueCallback<L> left = leftInput.dequeueCB();
				OOCStream.QueueCallback<R> right = rightInput.dequeueCB();
				boolean leftEos = left == null || left.isEos();
				boolean rightEos = right == null || right.isEos();
				if(leftEos || rightEos) {
					if(left != null)
						left.close();
					if(right != null)
						right.close();
					if(leftEos != rightEos)
						throw new DMLRuntimeException("Join inputs contain a different number of blocks");
					break;
				}
				accept(left, true, _leftKey.applyAsInt(left.get()), matches);
				accept(right, false, _rightKey.applyAsInt(right.get()), matches);
			}
		}
		catch(Throwable failure) {
			fail(failure);
			throw DMLRuntimeException.of(failure);
		}
		finally {
			completePending(matches);
		}
	}

	@SuppressWarnings("unchecked")
	private void accept(OOCStream.QueueCallback<? extends SpillableObject> callback, boolean left, int key,
		OOCStream<JoinWork> matches) {
		ReservationBudget budget = null;
		boolean pending = false;
		boolean handedOff = false;
		try {
			budget = OOCUtils.reserveBudget(_allowance, _taskBytes);
			_pending.incrementAndGet();
			pending = true;
			OOCFuture<StateTableUtils.Match<SpillableObject>> future = StateTableUtils.putOrTake(_table, key,
				(OOCStream.QueueCallback<SpillableObject>) callback, budget);
			handedOff = true;
			ReservationBudget pendingBudget = budget;
			budget = null;
			future.whenComplete((match, error) -> matchReady(match, left, pendingBudget, error, matches));
			pending = false;
		}
		finally {
			if(!handedOff)
				callback.close();
			if(pending)
				completePending(matches);
			if(budget != null)
				budget.close();
		}
	}

	private void matchReady(StateTableUtils.Match<SpillableObject> match, boolean incomingLeft,
		ReservationBudget budget, Throwable error, OOCStream<JoinWork> matches) {
		JoinWork work = null;
		try {
			if(error != null)
				throw DMLRuntimeException.of(error);
			if(match == null) {
				_unmatched.incrementAndGet();
				return;
			}
			_unmatched.decrementAndGet();
			work = new JoinWork(match.left(), match.right(), incomingLeft, budget);
			match = null;
			budget = null;
			matches.enqueue(work);
			work = null;
		}
		catch(Throwable failure) {
			fail(failure);
		}
		finally {
			if(work != null)
				work.close();
			if(match != null) {
				match.left().close();
				match.right().close();
			}
			if(budget != null)
				budget.close();
			completePending(matches);
		}
	}

	@SuppressWarnings("unchecked")
	private void process(JoinWork work) {
		SpillableObject incoming = work._incoming.get();
		SpillableObject existing = work._existing.get();
		L left = (L) (work._incomingLeft ? incoming : existing);
		R right = (R) (work._incomingLeft ? existing : incoming);
		O value = _operation.apply(left, right);
		long bytes = _outputSize.applyAsLong(value);
		work._budget.reserveBlocking(bytes);
		OOCStream.QueueCallback<O> callback = new InMemoryQueueCallback<>(value, null, work._budget, bytes);
		try {
			_outputStream.enqueue(callback);
			callback = null;
		}
		finally {
			if(callback != null)
				callback.close();
		}
	}

	private void completePending(OOCStream<JoinWork> matches) {
		if(_pending.decrementAndGet() != 0)
			return;
		try {
			int unmatched = _unmatched.get();
			if(unmatched != 0)
				fail(new DMLRuntimeException("Join inputs contain " + unmatched + " unmatched blocks"));
			else {
				try {
					matches.closeInput();
				}
				catch(Exception ignored) {
				}
			}
		}
		finally {
			_pendingCompletion.complete(null);
		}
	}

	private final class JoinWork implements AutoCloseable {
		private final OOCStream.QueueCallback<SpillableObject> _incoming;
		private final OOCStream.QueueCallback<SpillableObject> _existing;
		private final boolean _incomingLeft;
		private final ReservationBudget _budget;

		private JoinWork(OOCStream.QueueCallback<SpillableObject> incoming,
			OOCStream.QueueCallback<SpillableObject> existing, boolean incomingLeft, ReservationBudget budget) {
			_incoming = incoming;
			_existing = existing;
			_incomingLeft = incomingLeft;
			_budget = budget;
		}

		@Override
		public void close() {
			try {
				_incoming.close();
				_existing.close();
			}
			finally {
				_budget.close();
			}
		}
	}
}
