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

package org.apache.sysds.hops.estim;

import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

import org.apache.hadoop.shaded.org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.hops.estim.EstimationUtils.EstimatorType;
import org.apache.sysds.hops.estim.EstimatorMatrixHistogram.MatrixHistogram;
import org.apache.sysds.runtime.DMLRuntimeException;

// class representing and store any type of sparsity estimate
public class SparsityEstimate {
    Object _estimate;
    EstimatorType _type;

    public SparsityEstimate(Object estimate, EstimatorType type) {
        _estimate = estimate;
        _type = type;
    }

    public Object getEstimate() {
        return _estimate;
    }

    public EstimatorType getType() {
        return _type;
    }

    public boolean isType(EstimatorType checkType) {
        return _type == checkType;
    }

    public boolean isConvertible(EstimatorType type) {
        return SparsityEstimate.isConvertible(this.getType(), type);
    }

    public static boolean isConvertible(EstimatorType from, EstimatorType to) {
        boolean convertible = false;
        convertible |= from == to;
        switch (to) {
            case BASIC_AVG:
                convertible |= from == EstimatorType.RS;
            case RS:
                convertible |= from == EstimatorType.MNC;
                convertible |= from == EstimatorType.MNC_LIM;
                convertible |= from == EstimatorType.MNC_EXT;
                break;
            default:
                throw new NotImplementedException();
        }
        return convertible;
    }

    public SparsityEstimate convertToBasicAvg() {
        switch (_type) {
            case BASIC_AVG:
                return this;
            case RS:
                return new SparsityEstimate(Double.valueOf(
                    DoubleStream.of((double[])_estimate).average().getAsDouble()), EstimatorType.BASIC_AVG);
            case MNC, MNC_LIM, MNC_EXT:
                return new SparsityEstimate(Double.valueOf(
                    ((MatrixHistogram)_estimate).getSparsity()), EstimatorType.BASIC_AVG);
            default:
                throw new DMLRuntimeException("Cannot convert sparsity estimate from type " +
                    _type.name() + " to type " + EstimatorType.BASIC_AVG.name());
        }
    }

    public SparsityEstimate convertToRowwiseSparsity() {
        switch (_type) {
            case RS:
                return this;
            case MNC, MNC_LIM, MNC_EXT:
                return new SparsityEstimate(
                    IntStream.of(((MatrixHistogram)_estimate).getRowCounts()).mapToDouble(
                        nnz -> ((double)nnz) / ((MatrixHistogram)_estimate).getCols()).toArray(),
                    EstimatorType.RS);
            default:
                throw new DMLRuntimeException("Cannot convert sparsity estimate from type " +
                    _type.name() + " to type " + EstimatorType.BASIC_AVG.name());
        }
    }
}
