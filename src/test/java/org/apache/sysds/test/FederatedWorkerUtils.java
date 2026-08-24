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

package org.apache.sysds.test;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.util.function.BooleanSupplier;

/**
 * Test helpers that block until a federated worker is accepting TCP connections on its port.
 *
 * The federated worker opens its TCP port after Netty's {@code bind().sync()} returns. The methods here poll for a
 * successful TCP connection and throw {@link RuntimeException} on timeout or if the underlying
 * {@code Process}/{@code Thread} exits before the port becomes ready.
 */
public final class FederatedWorkerUtils {

	/** Sleep between successive poll rounds, in milliseconds. */
	private static final int POLL_INTERVAL_MS = 25;

	/**
	 * Per-attempt {@link Socket#connect} timeout in milliseconds, covering the full TCP handshake (round trip).
	 */
	private static final int CONNECT_TIMEOUT_MS = 2000;

	/**
	 * Minimum value applied to the caller-supplied {@code timeoutMs}, returns as soon as the worker accepts a
	 * connection.
	 */
	private static final int MIN_TIMEOUT_MS = 60_000;

	private FederatedWorkerUtils() {
		// utility class
	}

	/**
	 * Block until a federated worker is accepting TCP connections on {@code port}, or throw a
	 * {@link RuntimeException} after the effective timeout elapses.
	 *
	 * @param port      port the federated worker is expected to bind
	 * @param timeoutMs upper bound on the wait, in ms; raised to {@link #MIN_TIMEOUT_MS} if smaller
	 */
	public static void waitForWorker(int port, int timeoutMs) {
		waitForWorker(port, timeoutMs, () -> true, "worker");
	}

	/**
	 * Block until a federated worker is accepting TCP connections on {@code port}. Returns early with
	 * a {@link RuntimeException} if {@code aliveCheck} reports the worker is no longer alive.
	 */
	public static void waitForWorker(int port, int timeoutMs, BooleanSupplier aliveCheck, String workerKind) {
		final int effectiveTimeout = Math.max(timeoutMs, MIN_TIMEOUT_MS);
		final long deadline = System.currentTimeMillis() + effectiveTimeout;
		while(System.currentTimeMillis() < deadline) {
			if(!aliveCheck.getAsBoolean()) {
				throw new RuntimeException(
					"Federated " + workerKind + " on port " + port + " died before becoming ready.");
			}
			if(tryConnect(port, deadline)) {
				return;
			}
			sleepQuietly();
		}
		throw new RuntimeException("Federated " + workerKind + " on port " + port
			+ " did not become ready within " + effectiveTimeout + "ms.");
	}

	/** Overload that also returns early if the given worker process exits before the port is ready. */
	public static void waitForWorker(Process process, int port, int timeoutMs) {
		waitForWorker(port, timeoutMs, process::isAlive, "worker process");
	}

	/** Overload that also returns early if the given worker thread exits before the port is ready. */
	public static void waitForWorker(Thread thread, int port, int timeoutMs) {
		waitForWorker(port, timeoutMs, thread::isAlive, "worker thread");
	}

	/**
	 * Block until every listed federated worker is accepting TCP connections. All ports are polled in one shared loop,
	 * so the wall-clock wait is bounded by the slowest worker.
	 *
	 * @param ports     ports the workers are expected to bind
	 * @param timeoutMs upper bound on the wait, in ms; raised to {@link #MIN_TIMEOUT_MS} if smaller
	 */
	public static void waitForWorkers(int[] ports, int timeoutMs) {
		waitForWorkers(ports, timeoutMs, i -> true, "workers");
	}

	/**
	 * Overload that also returns early if any of the worker processes exits before its port is ready.
	 *
	 * @throws IllegalArgumentException if {@code processes.length != ports.length}
	 */
	public static void waitForWorkers(Process[] processes, int[] ports, int timeoutMs) {
		if(processes.length != ports.length) {
			throw new IllegalArgumentException(
				"processes/ports length mismatch: " + processes.length + " vs " + ports.length);
		}
		waitForWorkers(ports, timeoutMs, i -> processes[i].isAlive(), "worker processes");
	}

	/**
	 * Overload that also returns early if any of the worker threads exits before its port is ready.
	 *
	 * @throws IllegalArgumentException if {@code threads.length != ports.length}
	 */
	public static void waitForWorkers(Thread[] threads, int[] ports, int timeoutMs) {
		if(threads.length != ports.length) {
			throw new IllegalArgumentException(
				"threads/ports length mismatch: " + threads.length + " vs " + ports.length);
		}
		waitForWorkers(ports, timeoutMs, i -> threads[i].isAlive(), "worker threads");
	}

	/**
	 * Bulk variant taking a per-index liveness predicate so callers can plug in either {@code Process} or
	 * {@code Thread} liveness. Each port flips to ready as soon as it accepts a connection.
	 */
	public static void waitForWorkers(int[] ports, int timeoutMs, java.util.function.IntPredicate aliveCheck,
		String workerKind) {
		final int effectiveTimeout = Math.max(timeoutMs, MIN_TIMEOUT_MS);
		final long deadline = System.currentTimeMillis() + effectiveTimeout;
		final boolean[] ready = new boolean[ports.length];
		int remaining = ports.length;
		while(remaining > 0 && System.currentTimeMillis() < deadline) {
			// recheck the deadline per port, a sweep can spend up to CONNECT_TIMEOUT_MS on each of them
			for(int i = 0; i < ports.length && System.currentTimeMillis() < deadline; i++) {
				if(ready[i]) {
					continue;
				}
				if(!aliveCheck.test(i)) {
					throw new RuntimeException("Federated " + workerKind + " on port " + ports[i]
						+ " died before becoming ready.");
				}
				if(tryConnect(ports[i], deadline)) {
					ready[i] = true;
					remaining--;
				}
			}
			if(remaining > 0) {
				sleepQuietly();
			}
		}
		if(remaining > 0) {
			StringBuilder sb = new StringBuilder("Federated ").append(workerKind)
				.append(" did not all become ready within ").append(effectiveTimeout).append("ms. Pending ports:");
			for(int i = 0; i < ports.length; i++) {
				if(!ready[i]) {
					sb.append(' ').append(ports[i]);
				}
			}
			throw new RuntimeException(sb.toString());
		}
	}

	private static boolean tryConnect(int port, long deadline) {
		final long remaining = deadline - System.currentTimeMillis();
		if(remaining <= 0) // out of time => connect reads a timeout of 0 as "infinite"
			return false;
		try(Socket s = new Socket()) {
			s.connect(new InetSocketAddress("localhost", port), (int) Math.min(CONNECT_TIMEOUT_MS, remaining));
			return true;
		}
		catch(IOException e) { // closed port, or a handshake that outlasted the budget
			return false;
		}
	}

	private static void sleepQuietly() {
		try {
			Thread.sleep(POLL_INTERVAL_MS);
		}
		catch(InterruptedException ie) {
			Thread.currentThread().interrupt();
			throw new RuntimeException("Interrupted while waiting for federated worker", ie);
		}
	}
}
