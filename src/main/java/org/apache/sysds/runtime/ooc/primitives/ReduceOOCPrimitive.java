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

import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.ToLongFunction;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.ooc.OOCStream;
import org.apache.sysds.runtime.instructions.ooc.OOCStreamable;
import org.apache.sysds.runtime.ooc.cache.OOCCacheManager;
import org.apache.sysds.runtime.ooc.memory.InMemoryQueueCallback;
import org.apache.sysds.runtime.ooc.memory.ManagedPayload;
import org.apache.sysds.runtime.ooc.memory.ReservationBudget;
import org.apache.sysds.runtime.ooc.planning.OOCAccessPattern;
import org.apache.sysds.runtime.ooc.stream.AllocatedOOCStream;
import org.apache.sysds.runtime.ooc.stream.StreamContext;
import org.apache.sysds.runtime.ooc.util.OOCInstructionUtils;
import org.apache.sysds.runtime.ooc.util.OOCUtils;

public final class ReduceOOCPrimitive<I, O> extends OOCPrimitive {
	private final OOCStreamable<I> _input;
	private final OOCStreamable<O> _output;
	private final Function<I, O> _partial;
	private final BiFunction<O, O, O> _merge;
	private final ToLongFunction<O> _size;
	private ManagedPayload<O> _accumulator;

	public ReduceOOCPrimitive(OOCStreamable<I> input, OOCStreamable<O> output, Function<I, O> partial,
		BiFunction<O, O, O> merge, ToLongFunction<O> size, StreamContext context) {
		super(context, input);
		_input = input;
		_output = output;
		_partial = partial;
		_merge = merge;
		_size = size;
	}

	@Override
	protected void inferPatternsInternal() {
		_pattern = OOCAccessPattern.ANY;
		inferParentPatterns();
	}

	@Override
	protected void requestPatternInternal(OOCAccessPattern accessPattern) {
		_pattern = OOCAccessPattern.ANY;
	}

	@Override
	protected void startExecution() {
		OOCStream<I> input = getInputReadStream(0);
		OOCStream<O> output = _output.getWriteStream();
		long inputBytes = OOCUtils.estimateOutputTileBytes(_input.getDataCharacteristics());
		long outputBytes = OOCUtils.estimateOutputTileBytes(_output.getDataCharacteristics());
		long taskBytes = OOCCacheManager.getGlobalCache().maxPhysicalPinBytes(inputBytes) + 2 * outputBytes;
		AllocatedOOCStream<I> admitted = new AllocatedOOCStream<>(input, _allowance, ignored -> taskBytes);
		getContext().addOutStream(output);
		OOCInstructionUtils.submitOOCTasks(admitted, callback -> {
			ReservationBudget budget = null;
			ManagedPayload<O> partial = null;
			try {
				budget = AllocatedOOCStream.detachBudget(callback).enableReuse();
				O value = _partial.apply(callback.get());
				long bytes = _size.applyAsLong(value);
				budget.reserveBlocking(bytes);
				partial = new ManagedPayload<>(value, bytes, budget);
				synchronized(this) {
					if(_accumulator != null) {
						value = _merge.apply(_accumulator.value(), partial.value());
						bytes = _size.applyAsLong(value);
						budget.reserveBlocking(bytes);
						_accumulator.release();
						partial.release();
						partial = new ManagedPayload<>(value, bytes, budget);
					}
					_accumulator = partial;
					partial = null;
					budget.close();
					budget = null;
				}
			}
			catch(Throwable error) {
				fail(error);
				throw DMLRuntimeException.of(error);
			}
			finally {
				if(partial != null)
					partial.release();
				if(budget != null)
					budget.close();
			}
		}, getContext()).thenRun(() -> {
			ManagedPayload<O> result;
			synchronized(this) {
				result = _accumulator;
				_accumulator = null;
			}
			try {
				if(hasFailed()) {
					if(result != null)
						result.release();
					return;
				}
				if(result == null)
					throw new DMLRuntimeException("Cannot reduce an empty OOC stream");
				OOCStream.QueueCallback<O> callback = new InMemoryQueueCallback<>(result);
				try {
					output.enqueue(callback);
					callback = null;
				}
				finally {
					if(callback != null)
						callback.close();
				}
				output.closeInput();
			}
			catch(Throwable error) {
				if(result != null)
					result.release();
				fail(error);
			}
			finally {
				onComplete();
			}
		});
	}
}
