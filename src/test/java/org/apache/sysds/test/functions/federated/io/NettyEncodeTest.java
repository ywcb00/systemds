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
package org.apache.sysds.test.functions.federated.io;

import io.netty.buffer.ByteBuf;
import io.netty.buffer.ByteBufOutputStream;
import io.netty.buffer.PooledByteBufAllocator;

import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.Arrays;
import java.util.Collection;

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.FastBufferedDataOutputStream;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class NettyEncodeTest extends AutomatedTestBase {

	// private static final Log LOG = LogFactory.getLog(NettyEncodeTest.class.getName());
	private final static String TEST_DIR = "functions/federated/";
	private final static String TEST_NAME = "NettyEncodeTest";
	private final static String TEST_CLASS_DIR = TEST_DIR + NettyEncodeTest.class.getSimpleName() + "/";
	private final static int blocksize = 1024;

	@Parameterized.Parameter()
	public int rows;
	@Parameterized.Parameter(1)
	public int cols;
	@Parameterized.Parameter(2)
	public boolean rowPartitioned;
	@Parameterized.Parameter(3)
	public int fedCount;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME));
	}

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		// number of rows or cols has to be >= number of federated locations.
		return Arrays.asList(new Object[][] {{10, 13, true, 2}});
	}

	@Test
	public void nettyEncode() {
		PooledByteBufAllocator pbba = new PooledByteBufAllocator(false);
		ByteBuf bb = pbba.ioBuffer(256 * 1024 * 1024, 2147483647);
		double[][] X = getRandomMatrix(600000, 200, 0, 1, 1, 42);
		MatrixBlock mb = DataConverter.convertToMatrixBlock(X);
		ByteBufOutputStream bout = new ByteBufOutputStream(bb);
		try {
			ObjectOutputStream oout = new ObjectOutputStream(bout);
			FastBufferedDataOutputStream fbdos = new FastBufferedDataOutputStream(oout);
			long t0 = System.nanoTime();
			mb.write(fbdos);
			long t1 = System.nanoTime();
			System.out.println("NettyEncodeTest.java:87 - matrixblock write time: " + (((double)t1 - t0) / 1000000000) + "secs");
		} catch(IOException ioe) {
			ioe.printStackTrace();
		}
	}

}
