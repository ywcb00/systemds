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

package org.apache.sysds.test.functions.federated.quaternary;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.Collection;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class FederatedWeightedSquaredLossTest extends AutomatedTestBase
{
  private final static Log LOG = LogFactory.getLog(FederatedWeightedSquaredLossTest.class.getName());

  private final static String STD_TEST_NAME = "FederatedWSLossTest";
  private final static String PRE_TEST_NAME = "FederatedWSLossPreTest";
  private final static String POST_TEST_NAME = "FederatedWSLossPostTest";
  private final static String TEST_DIR = "functions/federated/quaternary/";
  private final static String TEST_CLASS_DIR = TEST_DIR + FederatedWeightedSquaredLossTest.class.getSimpleName() + "/";

  private final static String OUTPUT_NAME = "Z";

  private final static double TEST_TOLERANCE = 1e-9;

  private final static int blocksize = 1024;

  private String test_name;
  @Parameterized.Parameter()
  public int rows;
  @Parameterized.Parameter(1)
  public int cols;
  @Parameterized.Parameter(2)
  public int rank;
  @Parameterized.Parameter(3)
  public double sparsity;


  @Override
  public void setUp()
  {
    addTestConfiguration(STD_TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, STD_TEST_NAME, new String[]{OUTPUT_NAME}));
    addTestConfiguration(PRE_TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, PRE_TEST_NAME, new String[]{OUTPUT_NAME}));
    addTestConfiguration(POST_TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, POST_TEST_NAME, new String[]{OUTPUT_NAME}));
  }

  @Parameterized.Parameters
  public static Collection<Object[]> data()
  {
    // rows must be even
    return Arrays.asList(new Object[][] {
      // {rows, cols, rank, sparsity}
      {2000, 50, 10, 0.01},
      {2000, 50, 10, 0.9},
      {2000, 50, 10, 0.01},
      {2000, 50, 10, 0.9}
    });
  }

  @BeforeClass
  public static void init()
  {
    TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
  }

  @Test
  public void federatedWeightedSquaredLossSingleNode()
  {
    test_name = STD_TEST_NAME;
    federatedWeightedSquaredLoss(ExecMode.SINGLE_NODE);
  }

  @Test
  public void federatedWeightedSquaredLossSpark()
  {
    test_name = STD_TEST_NAME;
    federatedWeightedSquaredLoss(ExecMode.SPARK);
  }

  @Test
  public void federatedWeightedSquaredLossPreSingleNode()
  {
    test_name = PRE_TEST_NAME;
    federatedWeightedSquaredLoss(ExecMode.SINGLE_NODE);
  }

  @Test
  public void federatedWeightedSquaredLossPreSpark()
  {
    test_name = PRE_TEST_NAME;
    federatedWeightedSquaredLoss(ExecMode.SPARK);
  }

  @Test
  public void federatedWeightedSquaredLossPostSingleNode()
  {
    test_name = PRE_TEST_NAME;
    federatedWeightedSquaredLoss(ExecMode.SINGLE_NODE);
  }

  @Test
  public void federatedWeightedSquaredLossPostSpark()
  {
    test_name = PRE_TEST_NAME;
    federatedWeightedSquaredLoss(ExecMode.SPARK);
  }

// -----------------------------------------------------------------------------

  public void federatedWeightedSquaredLoss(ExecMode exec_mode)
  {
    // store the previous platform config to restore it after the test
    ExecMode platform_old = getExecMode();

    getAndLoadTestConfiguration(test_name);
    String HOME = SCRIPT_DIR + TEST_DIR;

    int fed_rows = rows / 2;
    int fed_cols = cols;

    // generate dataset
    // matrix handled by two federated workers
    double[][] X1 = getRandomMatrix(fed_rows, fed_cols, 0, 1, sparsity, 3);
    double[][] X2 = getRandomMatrix(fed_rows, fed_cols, 0, 1, sparsity, 7);

    double[][] U = getRandomMatrix(rows, rank, 0, 1, 1, 512);
    double[][] V = getRandomMatrix(cols, rank, 0, 1, 1, 5040);

    writeInputMatrixWithMTD("X1", X1, false, new MatrixCharacteristics(fed_rows, fed_cols, blocksize, fed_rows * fed_cols));
    writeInputMatrixWithMTD("X2", X2, false, new MatrixCharacteristics(fed_rows, fed_cols, blocksize, fed_rows * fed_cols));

    writeInputMatrixWithMTD("U", U, true);
    writeInputMatrixWithMTD("V", V, true);

    if(!test_name.equals(STD_TEST_NAME))
    {
      double[][] W = getRandomMatrix(rows, cols, 0, 1, sparsity, 54);
      writeInputMatrixWithMTD("W", W, true);
    }

    // empty script name because we don't execute any script, just start the worker
    fullDMLScriptName = "";
    int port1 = getRandomAvailablePort();
    int port2 = getRandomAvailablePort();
    Thread thread1 = startLocalFedWorkerThread(port1, FED_WORKER_WAIT_S);
    Thread thread2 = startLocalFedWorkerThread(port2);

    getAndLoadTestConfiguration(test_name);

    // we need the reference file to not be written to hdfs, so we get the correct format
		setExecMode(ExecMode.SINGLE_NODE);
    // Run reference dml script with normal matrix
    fullDMLScriptName = HOME + test_name + "Reference.dml";
    programArgs = new String[] {"-nvargs", "in_X1=" + input("X1"), "in_X2=" + input("X2"),
      "in_U=" + input("U"), "in_V=" + input("V"), "in_W=" + input("W"),
      "out_Z=" + expected(OUTPUT_NAME)};
    LOG.debug(runTest(true, false, null, -1));

    // set the execution mode for the actual test
    setExecMode(exec_mode);
    // Run actual dml script with federated matrix
    fullDMLScriptName = HOME + test_name + ".dml";
    programArgs = new String[] {"-stats", "-nvargs",
      "in_X1=" + TestUtils.federatedAddress(port1, input("X1")),
      "in_X2=" + TestUtils.federatedAddress(port2, input("X2")),
      "in_U=" + input("U"),
      "in_V=" + input("V"),
      "in_W=" + input("W"),
      "rows=" + fed_rows, "cols=" + fed_cols, "out_Z=" + output(OUTPUT_NAME)};
    LOG.debug(runTest(true, false, null, -1));

    // compare the results via files
    compareResults(TEST_TOLERANCE);

    TestUtils.shutdownThreads(thread1, thread2);

    // check for federated operations
    Assert.assertTrue(heavyHittersContainsString("fed_wsloss"));

    // check that federated input files are still existing
    Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X1")));
    Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X2")));

    resetExecMode(platform_old);
  }

}
