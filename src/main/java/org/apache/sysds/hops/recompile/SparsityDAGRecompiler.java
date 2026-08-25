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

package org.apache.sysds.hops.recompile;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.lang3.mutable.MutableInt;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.hops.DataOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.HopsException;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.estim.EstimationUtils.EstimatorType;
import org.apache.sysds.hops.estim.SparsityEstimator.OpCode;
import org.apache.sysds.hops.estim.MMNode;
import org.apache.sysds.hops.estim.SparsityEstimator;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.util.CollectionUtils;
import org.apache.sysds.utils.Explain;

public class SparsityDAGRecompiler {
	private static final Log LOG = LogFactory.getLog(SparsityDAGRecompiler.class.getName());

	private final ExecutionContext _ec;

	public SparsityDAGRecompiler(ExecutionContext ec) {
		this._ec = ec;
	}

	protected static void clearLinksWithinChain(Hop hop, List<Hop> operators) {
		for(int i=0; i < operators.size(); i++) {
			Hop op = operators.get(i);
			if(op.getInput().size() != 2 || (i != 0 && op.getParent().size() > 1 )) {
				throw new HopsException(hop.printErrorLocation() +
					"Unexpected error while applying sparsity-based recompilation on matrix-mult chain. \n");
			}
			Hop input1 = op.getInput().get(0);
			Hop input2 = op.getInput().get(1);

			op.getInput().clear();
			input1.getParent().remove(op);
			input2.getParent().remove(op);
		}
	}

	/**
	 * NOTE: Copied from RewriteMatrixMultChainOptimizationSparse.java
	 * Obtains all dimension information of the chain and constructs the dimArray.
	 * If all dimensions are known it returns true; othrewise the mmchain rewrite
	 * should be ended without modifications.
	 *
	 * @param hop high-level operator
	 * @param chain list of high-level operators
	 * @param dimsArray dimension array
	 * @return true if all dimensions known
	 */
	protected static boolean getDimsArray(Hop hop, List<Hop> chain, double[] dimsArray) {
		boolean dimsKnown = true;

		// Build the array containing dimensions from all matrices in the chain
		// check the dimensions in the matrix chain to insure all dimensions are known
		for( int i=0; i< chain.size(); i++ )
			if( chain.get(i).getDim1() <= 0 || chain.get(i).getDim2() <= 0 )
				dimsKnown = false;

		if(dimsKnown) { // populate dims array if all dims known
			for( int i = 0; i < chain.size(); i++ ) {
				if (i == 0) {
					dimsArray[i] = chain.get(i).getDim1();
					if (dimsArray[i] <= 0) {
						throw new HopsException(hop.printErrorLocation() +
								"Hops::optimizeMMChain() : Invalid Matrix Dimension: "+ dimsArray[i]);
					}
				}
				else if (chain.get(i - 1).getDim2() != chain.get(i).getDim1()) {
					throw new HopsException(hop.printErrorLocation() +
						"Hops::optimizeMMChain() : Matrix Dimension Mismatch: " +
						chain.get(i - 1).getDim2()+" != "+chain.get(i).getDim1());
				}

				dimsArray[i + 1] = chain.get(i).getDim2();
				if( dimsArray[i + 1] <= 0 ) {
					throw new HopsException(hop.printErrorLocation() +
							"Hops::optimizeMMChain() : Invalid Matrix Dimension: " + dimsArray[i + 1]);
				}
			}
		}

		return dimsKnown;
	}

	/**
	 * NOTE: Copied from RewriteMatrixMultChainOptimizationSparse.java
	 * mmChainRelinkHops(): This method gets invoked after finding the optimal
	 * order (split[][]) from dynamic programming. It relinks the Hops that are
	 * part of the mmChain.
	 * @param mmChain : basic operands in the entire matrix multiplication chain.
	 * @param mmOperators : Hops that store the intermediate results in the chain.
	 *                      For example: A = B %*% (C %*% D) there will be three
	 *                      Hops in mmChain (B,C,D), and two Hops in mmOperators
	 *                     (one for each * %*%).
	 * @param h high level operator
	 * @param i array index i
	 * @param j array index j
	 * @param opIndex operator index
	 * @param split optimal order
	 * @param level log level
	 */
	protected final void mmChainRelinkHops(Hop h, int i, int j, List<Hop> mmChain,
		List<Hop> mmOperators, MutableInt opIndex, int[][] split, int level) {
		// NOTE: the opIndex is a MutableInt in order to get the correct positions
		// in ragged chains like ((((a, b), c), (D, E), f), e) that might be given
		// like that by the original scripts variable assignments
		// single matrix - end of recursion
		if(i == j) {
			logTraceHop(h, level);
			return;
		}

		if(LOG.isTraceEnabled()){
			String offset = Explain.getIdentation(level);
			LOG.trace(offset + "(");
		}

		// Set Input1 for current Hop h
		if(i == split[i][j]) {
			h.getInput().add(mmChain.get(i));
			mmChain.get(i).getParent().add(h);
		}
		else {
			int ix = opIndex.getValue();
			opIndex.increment();
			h.getInput().add(mmOperators.get(ix));
			mmOperators.get(ix).getParent().add(h);
		}

		// Set Input2 for current Hop h
		if(split[i][j] + 1 == j) {
			h.getInput().add(mmChain.get(j));
			mmChain.get(j).getParent().add(h);
		}
		else {
			int ix = opIndex.getValue();
			opIndex.increment();
			h.getInput().add(mmOperators.get(ix));
			mmOperators.get(ix).getParent().add(h);
		}

		// Find children for both the inputs
		mmChainRelinkHops(h.getInput().get(0), i, split[i][j],
			mmChain, mmOperators, opIndex, split, level+1);
		mmChainRelinkHops(h.getInput().get(1), split[i][j] + 1, j,
			mmChain, mmOperators, opIndex, split, level+1);

		// Propagate properties of input hops to current hop h
		h.refreshSizeInformation();

		if(LOG.isTraceEnabled()){
			String offset = Explain.getIdentation(level);
			LOG.trace(offset + ")");
		}
	}

	protected void optimizeMMChain(Hop hop, List<Hop> mmChain, List<Hop> mmOperators) {
		// Step 2: construct dims array and input matrices
		double[] dimsArray = new double[mmChain.size() + 1];
		boolean dimsKnown = getDimsArray(hop, mmChain, dimsArray);
		MMNode[] sketchArray = new MMNode[mmChain.size() + 1];
		boolean inputMetaAvail = getInputMatrixCharacteristics(hop, mmChain, sketchArray);
		if(dimsKnown && inputMetaAvail) {
			// Step 3: clear the links among Hops within the identified chain
			clearLinksWithinChain ( hop, mmOperators );

			// Step 4: Find the optimal ordering via dynamic programming.

			// Invoke Dynamic Programming
			int size = mmChain.size();
			int[][] split = mmChainDPSparse(dimsArray, sketchArray, mmChain.size());

			 // Step 5: Relink the hops using the optimal ordering (split[][]) found from DP.
			LOG.trace("Optimal Sparse MM Chain:");
			mmChainRelinkHops(mmOperators.get(0), 0, size - 1, mmChain, mmOperators,
				new MutableInt(1), split, 1);
		}
	}

	/**
	 * NOTE: Copied from RewriteMatrixMultChainOptimizationSparse.java
	 * mmChainDP(): Core method to perform dynamic programming on a given array
	 * of matrix dimensions.
	 *
	 * Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, Clifford Stein
	 * Introduction to Algorithms, Third Edition, MIT Press, page 395.
	 */
	private static int[][] mmChainDPSparse(double[] dimArray, MMNode[] sketchArray, int size) {
		double[][] dpMatrix = new double[size][size]; //min cost table
		MMNode[][] dpMatrixS = new MMNode[size][size]; //min sketch table
		int[][] split = new int[size][size]; //min cost index table

		//init minimum costs for chains of length 1
		for( int i = 0; i < size; i++ ) {
			Arrays.fill(dpMatrix[i], 0);
			Arrays.fill(split[i], -1);
			dpMatrixS[i][i] = sketchArray[i];
		}

		//compute cost-optimal chains for increasing chain sizes
		SparsityEstimator estim = EstimatorType.valueOf(ConfigurationManager.getDMLConfig()
			.getTextValue(DMLConfig.SPARSITY_ESTIMATOR)).getEstimator();
		for(int l = 2; l <= size; l++) { // chain length
			for(int i = 0; i < size - l + 1; i++) {
				int j = i + l - 1;
				// find cost of (i,j)
				dpMatrix[i][j] = Double.MAX_VALUE;
				for(int k = i; k <= j - 1; k++) {
					// construct estimation nodes (w/ lazy propagation and memoization)
					MMNode tmp = new MMNode(dpMatrixS[i][k], dpMatrixS[k+1][j], OpCode.MM);
					estim.estim(tmp);

					// recursive cost computation
					double cost = dpMatrix[i][k] + dpMatrix[k + 1][j] +
						OptimizerUtils.getSparsity(tmp.getLeft().getDataCharacteristics()) *
							OptimizerUtils.getSparsity(tmp.getRight().getDataCharacteristics()) *
							tmp.getLeft().getRows() * tmp.getLeft().getCols() * tmp.getRight().getCols();

					// prune suboptimal
					if( cost < dpMatrix[i][j] ) {
						dpMatrix[i][j] = cost;
						dpMatrixS[i][j] = tmp;
						split[i][j] = k;
					}
				}

				if(LOG.isTraceEnabled())
					LOG.trace("mmchainoptsparse [i=" + (i + 1) + ",j=" + (j + 1) + "]: costs = " + dpMatrix[i][j]
						+ ", split = " + (split[i][j] + 1));
			}
		}

		return split;
	}

	private static boolean getInputMatrixCharacteristics(Hop hop, List<Hop> chain, MMNode[] sketchArray) {
		boolean inputMetaAvail = true;

		for(int counter = 0; counter < chain.size(); counter++) {
			Hop currentHop = chain.get(counter);
			inputMetaAvail &= currentHop.isMatrix();
			inputMetaAvail &= !currentHop.isFederated();
			inputMetaAvail &= (currentHop.getNnz() != -1);
			if(inputMetaAvail) {
				sketchArray[counter] = new MMNode(currentHop.getDataCharacteristics());
			}
			else
				break;
		}

		return inputMetaAvail;
	}

	private static int inputCount(Hop p, Hop h) {
		return CollectionUtils.cardinality(h, p.getInput());
	}

	private static void logTraceHop(Hop hop, int level) {
		if(LOG.isTraceEnabled()) {
			String offset = Explain.getIdentation(level);
			LOG.trace(offset + "Hop " + hop.getName() + "(" + hop.getClass().getSimpleName() +
				", " + hop.getHopID() + ")" + " " + hop.getDim1() + "x" + hop.getDim2() +
				"[nnz" + hop.getNnz() + "]");
		}
	}

	/**
	 * NOTE: Copied from RewriteMatrixMultChainOptimizationSparse.java
	 * optimizeMMChain(): It optimizes the matrix multiplication chain in which
	 * the last Hop is "hop". Step-1) Identify the chain (mmChain). (Step-2) clear all
	 * links among the Hops that are involved in mmChain. (Step-3) Find the
	 * optimal ordering (dynamic programming) (Step-4) Relink the hops in
	 * mmChain.
	 *
	 * @param hop high-level operator
	 */
	private void prepAndOptimizeMMChain(Hop hop) {
		if(LOG.isTraceEnabled()) {
			LOG.trace("Sparsity-based MM Chain Recompilation for HOP: (" +
				hop.getClass().getSimpleName() + ", " + hop.getHopID() +
				", " + hop.getName() + ")");
		}

		List<Hop> mmChain = new ArrayList<>();
		List<Hop> mmOperators = new ArrayList<>();
		List<Hop> tempList;

		// Step 1: Identify the chain (mmChain) & clear all links among the Hops
		// that are involved in mmChain.

		// Initialize mmChain with my inputs
		mmOperators.add(hop);
		for(Hop hi : hop.getInput())
			mmChain.add(hi);

		// expand each Hop in mmChain to find the entire matrix multiplication
		// chain
		int i = 0;
		while(i < mmChain.size()) {
			boolean expandable = false;

			Hop h = mmChain.get(i);
			/*
			 * Check if mmChain[i] is expandable:
			 * 1) It must be MATMULT
			 * 2) It must not have been visited already
			 *    (one MATMULT should get expanded only in one chain)
			 * 3) Its output should not be used in multiple places
			 *    (either within chain or outside the chain)
			 */

			if(HopRewriteUtils.isMatrixMultiply(h) &&
				!((AggBinaryOp)h).hasLeftPMInput() && !h.isVisited()) {
				// check if the output of "h" is used at multiple places. If yes, it can
				// not be expanded.
				expandable = !(h.getParent().size() > 1 ||
					inputCount(h.getParent().get(0), h) > 1);
				if(!expandable) {
					optimizeHopDAG(h);
					break;
				}
			}
			else {
				i = i + 1;
			}

			if(expandable) {
				h.setVisited();
				tempList = mmChain.get(i).getInput();
				if(tempList.size() != 2) {
					throw new HopsException(hop.printErrorLocation() +
						"Hops::rule_OptimizeMMChain(): AggBinary must have exactly two inputs.");
				}

				// add current operator to mmOperators, and its input nodes to mmChain
				mmOperators.add(mmChain.get(i));
				mmChain.set(i, tempList.get(0));
				mmChain.add(i + 1, tempList.get(1));
			}
			else {
				optimizeHopDAG(h);
			}
		}

		// print the MMChain
		if(LOG.isTraceEnabled()) {
			LOG.trace("Identified MM Chain: ");
			for(Hop h : mmChain) {
				logTraceHop(h, 1);
			}
		}

		// core mmchain optimization
		if(mmChain.size() == 2)
			return; // nothing to optimize
		else
			optimizeMMChain(hop, mmChain, mmOperators);
	}

	private void prepHop(Hop hop) {
		hop.setVisited();

		// optimize the inputs
		for(Hop hi : hop.getInput())
			optimizeHopDAG(hi); // recursion

		// TODO: Optimize the hops that are not matrix multiplications. Its inputs are already optimized.
		// (i.e., pre-fetching and pre-computations)
		if(hop instanceof DataOp && ((DataOp)hop).isRead() && ((DataOp)hop).getNnz() < 0) {
			// TODO: pre-fetch
		}
		// TODO: pre-compute
	}

	private void optimizeHopDAG(Hop hop) {
		if(!hop.isMatrix() || hop.isFederated() || hop.isVisited())
			return;

		// TODO: Also rewrite chains with additional operations (e.g,. cbind, etc.)
		if(HopRewriteUtils.isMatrixMultiply(hop) && !((AggBinaryOp)hop).hasLeftPMInput()) {
			// Try to find and optimize the chain in which current Hop is the
			// last operator
			prepAndOptimizeMMChain(hop);
		}

		if(!hop.isVisited()) {
			prepHop(hop);
		}
		return;
	}

	private ArrayList<Hop> optimizeHopDAGs(ArrayList<Hop> roots) {
		for(Hop r : roots) {
			optimizeHopDAG(r);
		}
		return roots;
	}

	public static ArrayList<Hop> optimize(ArrayList<Hop> roots, ExecutionContext ec) {
		SparsityDAGRecompiler spRecomp = new SparsityDAGRecompiler(ec);
		Hop.resetVisitStatus(roots);
		ArrayList<Hop> ret = spRecomp.optimizeHopDAGs(roots);
		return ret;
	}
}
