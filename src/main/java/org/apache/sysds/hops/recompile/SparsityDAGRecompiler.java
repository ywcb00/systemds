package org.apache.sysds.hops.recompile;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.hops.DataOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.rewrite.RewriteMatrixMultChainOptimizationSparse;

public class SparsityDAGRecompiler {
	private static final Log LOG = LogFactory.getLog(SparsityDAGRecompiler.class);

	private static void rObtainInputMatrixCharacteristics(Hop hop) {
		if(hop.isMatrix() && !hop.isFederated()) {
			System.out.println("Hop: " + hop.toString() + " requires recompile " + hop.requiresRecompile());
			if(hop instanceof DataOp && ((DataOp) hop).isRead()) {
				DataOp dop = (DataOp) hop;
				// TODO: pre-fetch this read operation
				return;
			}
			List<Hop> inputs = hop.getInput();
			for(Hop in : inputs) {
				rObtainInputMatrixCharacteristics(in);
			}
		}
	}

	public static ArrayList<Hop> optimize(ArrayList<Hop> roots) {
		for(Hop r : roots) {
			if(r.isMatrix() && !r.isFederated()) {
				// TODO: pre-fetching of input data and deducing their metadata (nnz)
				// rObtainInputMatrixCharacteristics(r);
				RewriteMatrixMultChainOptimizationSparse rewriter = new RewriteMatrixMultChainOptimizationSparse();
				rewriter.rewriteHopDAGs(roots, null);
			}
		}
		return roots;
	}
}
