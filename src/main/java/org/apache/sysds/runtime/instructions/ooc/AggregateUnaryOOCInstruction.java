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

package org.apache.sysds.runtime.instructions.ooc;

import org.apache.sysds.common.Types.CorrectionLocationType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.parfor.LocalTaskQueue;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.DoubleObject;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.data.OperationsOnMatrixValues;
import org.apache.sysds.runtime.matrix.operators.AggregateOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.ooc.primitives.GroupedReduceOOCPrimitive;
import org.apache.sysds.runtime.ooc.util.OOCInstructionUtils;
import org.apache.sysds.runtime.ooc.util.OOCUtils;

import java.util.HashMap;

public class AggregateUnaryOOCInstruction extends ComputationOOCInstruction {
	private AggregateOperator _aop = null;

	protected AggregateUnaryOOCInstruction(OOCType type, AggregateUnaryOperator auop, AggregateOperator aop, 
			CPOperand in, CPOperand out, String opcode, String istr) {
		super(type, auop, in, out, opcode, istr);
		_aop = aop;
	}

	protected AggregateUnaryOOCInstruction(OOCType type, Operator op, CPOperand in1, CPOperand in2, CPOperand in3,
		CPOperand out, String opcode, String istr) {
		super(type, op, in1, in2, in3, out, opcode, istr);
		_aop = null;
	}

	public static AggregateUnaryOOCInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		InstructionUtils.checkNumFields(parts, 2);
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand out = new CPOperand(parts[2]);
		
		String aopcode = InstructionUtils.deriveAggregateOperatorOpcode(opcode);
		CorrectionLocationType corrLoc = InstructionUtils.deriveAggregateOperatorCorrectionLocation(opcode);
		AggregateUnaryOperator aggun = InstructionUtils.parseBasicAggregateUnaryOperator(opcode);
		AggregateOperator aop = InstructionUtils.parseAggregateOperator(aopcode, corrLoc.toString());
		return new AggregateUnaryOOCInstruction(
			OOCType.AggregateUnary, aggun, aop, in1, out, opcode, str);
	}
	
	@Override
	public void processInstruction( ExecutionContext ec ) {
		//TODO support all types of aggregations, currently only full aggregation, row aggregation and column aggregation
		
		AggregateUnaryOperator aggun = (AggregateUnaryOperator) getOperator(); 
		MatrixObject min = ec.getMatrixObject(input1);
		DataCharacteristics chars = ec.getDataCharacteristics(input1.getName());
		int blen = chars != null && chars.getBlocksize() > 0 ? chars.getBlocksize() : ConfigurationManager
			.getBlocksize();

		if(!aggun.isRowAggregate() && !aggun.isColAggregate()) {
			processScalarAggregate(ec, min, aggun, blen);
			return;
		}
		if(OOCUtils.getNumBlocks(chars) > 0) {
			processPlannerMatrixAggregate(ec, min, aggun, blen);
			return;
		}

		OOCStream<IndexedMatrixValue> qIn = min.getStreamHandle();
		long emitThreshold = aggun.isRowAggregate() ? chars.getNumColBlocks() : chars.getNumRowBlocks();
		OOCMatrixBlockTracker aggTracker = new OOCMatrixBlockTracker(emitThreshold);
		HashMap<Long, MatrixBlock> corrs = new HashMap<>();
		OOCStream<IndexedMatrixValue> qOut = createWritableStream();
		OOCStream<IndexedMatrixValue> qLocal = createWritableStream();
		ec.getMatrixObject(output).setStreamHandle(qOut);

		mapOOC(qIn, qLocal, tmp -> {
			MatrixIndexes midx = aggun.isRowAggregate() ? new MatrixIndexes(tmp.getIndexes().getRowIndex(),
				1) : new MatrixIndexes(1, tmp.getIndexes().getColumnIndex());
			MatrixBlock ltmp = (MatrixBlock) ((MatrixBlock) tmp.getValue()).aggregateUnaryOperations(aggun,
				new MatrixBlock(), blen, tmp.getIndexes());
			return new IndexedMatrixValue(midx, ltmp);
		});

		addOutStream(qOut);
		submitOOCTasks(qLocal, callback -> {
			IndexedMatrixValue partial = callback.get();
			synchronized(aggTracker) {
				long idx = aggun.isRowAggregate() ? partial.getIndexes().getRowIndex() : partial.getIndexes()
					.getColumnIndex();
				MatrixBlock ret = aggTracker.get(idx);
				boolean ready;
				if(ret != null) {
					MatrixBlock corr = corrs.get(idx);
					OperationsOnMatrixValues.incrementalAggregation(ret, _aop.existsCorrection() ? corr : null,
						(MatrixBlock) partial.getValue(), _aop, true);
					ready = aggTracker.incrementCount(idx);
				}
				else {
					ret = (MatrixBlock) partial.getValue();
					MatrixBlock corr = _aop.existsCorrection() ? new MatrixBlock(ret.getNumRows(), ret.getNumColumns(),
						false) : null;
					ready = aggTracker.putAndIncrementCount(idx, ret);
					if(!ready && _aop.existsCorrection())
						corrs.put(idx, corr);
				}
				if(ready) {
					ret.dropLastRowsOrColumns(_aop.correction);
					qOut.enqueue(new IndexedMatrixValue(partial.getIndexes(), ret));
					aggTracker.remove(idx);
					corrs.remove(idx);
				}
			}
		}).thenRun(qOut::closeInput);
	}

	private void processPlannerMatrixAggregate(ExecutionContext ec, MatrixObject input, AggregateUnaryOperator operator,
		int blocksize) {
		OOCStream<IndexedMatrixValue> outputStream = createWritableStream();
		ec.getMatrixObject(output).setStreamHandle(outputStream);
		GroupedReduceOOCPrimitive.Grouping grouping = operator
			.isRowAggregate() ? GroupedReduceOOCPrimitive.Grouping.ROW_BLOCKS : GroupedReduceOOCPrimitive.Grouping.COL_BLOCKS;
		OOCInstructionUtils.groupedReduceIndexed(input.getStreamable(), outputStream, grouping,
			value -> aggregatePartial(value, operator, blocksize), this::mergeAggregate, this::finalizeAggregate,
			getContext());
	}

	private void processScalarAggregate(ExecutionContext ec, MatrixObject input, AggregateUnaryOperator operator,
		int blocksize) {
		OOCStream<MatrixBlock> partials = createWritableStream();
		mapOOC(input.getStreamHandle(), partials, value -> aggregatePartial(value, operator, blocksize));

		int extra = _aop.correction.getNumRemovedRowsColumns();
		MatrixBlock result = new MatrixBlock(1, 1 + extra, _aop.initialValue);
		MatrixBlock correction = new MatrixBlock(1, 1 + extra, false);
		MatrixBlock partial;
		while((partial = partials.dequeue()) != LocalTaskQueue.NO_MORE_TASKS)
			OperationsOnMatrixValues.incrementalAggregation(result, _aop.existsCorrection() ? correction : null,
				partial, _aop, true);
		ec.setScalarOutput(output.getName(), new DoubleObject(result.get(0, 0)));
	}

	private static MatrixBlock aggregatePartial(IndexedMatrixValue value, AggregateUnaryOperator operator,
		int blocksize) {
		return (MatrixBlock) value.getValue().aggregateUnaryOperations(operator, new MatrixBlock(), blocksize,
			value.getIndexes());
	}

	private MatrixBlock mergeAggregate(MatrixBlock left, MatrixBlock right) {
		OperationsOnMatrixValues.incrementalAggregation(left, null, right, _aop, true);
		return left;
	}

	private MatrixBlock finalizeAggregate(MatrixBlock block) {
		block.dropLastRowsOrColumns(_aop.correction);
		return block;
	}
}
