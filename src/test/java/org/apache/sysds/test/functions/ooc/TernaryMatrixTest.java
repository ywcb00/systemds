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

package org.apache.sysds.test.functions.ooc;

import java.io.IOException;

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.io.MatrixWriter;
import org.apache.sysds.runtime.io.MatrixWriterFactory;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

public class TernaryMatrixTest extends AutomatedTestBase {
	private static final String TEST_NAME = "TernaryMatrix";
	private static final String TEST_DIR = "functions/ooc/";
	private static final String TEST_CLASS_DIR = TEST_DIR + TernaryMatrixTest.class.getSimpleName() + "/";
	private static final int ROWS = 1200;
	private static final int COLS = 1100;
	private static final int BLOCK_SIZE = 1000;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME));
	}

	@Test
	public void testTernaryOperations() throws IOException {
		Types.ExecMode oldPlatform = setExecMode(Types.ExecMode.SINGLE_NODE);
		try {
			getAndLoadTestConfiguration(TEST_NAME);
			fullDMLScriptName = SCRIPT_DIR + TEST_DIR + TEST_NAME + ".dml";
			writeInput("A", MatrixBlock.randOperations(ROWS, COLS, 1, -1, 1, "uniform", 7));
			writeInput("B", MatrixBlock.randOperations(ROWS, COLS, 0.7, -2, 2, "uniform", 8));
			writeInput("C", MatrixBlock.randOperations(ROWS, COLS, 0.2, -3, 3, "uniform", 9));

			String[] outputs = {"plus", "minus", "ifelse"};
			Opcodes[] opcodes = {Opcodes.PM, Opcodes.MINUSMULT, Opcodes.IFELSE};
			for(int i = 0; i < outputs.length; i++) {
				programArgs = arguments(true, i + 1, outputs[i]);
				runTest(true, false, null, -1);
				Assert.assertTrue(heavyHittersContainsString(Instruction.OOC_INST_PREFIX + opcodes[i]));

				programArgs = arguments(false, i + 1, outputs[i] + "_target");
				runTest(true, false, null, -1);
				MatrixBlock actual = DataConverter.readMatrixFromHDFS(output(outputs[i]), Types.FileFormat.BINARY, ROWS,
					COLS, BLOCK_SIZE);
				MatrixBlock expected = DataConverter.readMatrixFromHDFS(output(outputs[i] + "_target"),
					Types.FileFormat.BINARY, ROWS, COLS, BLOCK_SIZE);
				TestUtils.compareMatrices(actual, expected, 1e-8);
			}
		}
		finally {
			resetExecMode(oldPlatform);
		}
	}

	private String[] arguments(boolean ooc, int operation, String result) {
		String[] args = new String[ooc ? 8 : 7];
		int offset = 0;
		args[offset++] = "-stats";
		if(ooc)
			args[offset++] = "-ooc";
		args[offset++] = "-args";
		args[offset++] = input("A");
		args[offset++] = input("B");
		args[offset++] = input("C");
		args[offset++] = Integer.toString(operation);
		args[offset] = output(result);
		return args;
	}

	private void writeInput(String name, MatrixBlock value) throws IOException {
		MatrixWriter writer = MatrixWriterFactory.createMatrixWriter(Types.FileFormat.BINARY);
		writer.writeMatrixToHDFS(value, input(name), ROWS, COLS, BLOCK_SIZE, value.getNonZeros());
		HDFSTool.writeMetaDataFile(input(name + ".mtd"), Types.ValueType.FP64,
			new MatrixCharacteristics(ROWS, COLS, BLOCK_SIZE, value.getNonZeros()), Types.FileFormat.BINARY);
	}
}
