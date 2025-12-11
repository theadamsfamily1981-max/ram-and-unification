#!/usr/bin/env python3
"""
pgu_constraints.py
Proof-Guided Updater (PGU) with SMT Topological Constraints

Implements formal safety verification for structural changes using:
- β₀ (Betti-0): Connected components - ensures network connectivity
- β₁ (Betti-1): Independent cycles - ensures redundancy/loops for fault tolerance
- Spectral constraints: Ensures Jacobian stability after changes

The PGU acts as an ACTIVE SAFETY GATE between AEPO proposals and model updates.

Architecture:
    AEPO Proposal → SMT Constraint Check → Topological Verification → Accept/Reject

Usage:
    pgu = ProofGuidedUpdater(config)
    result = pgu.verify_proposal(proposal, current_graph)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)


# =============================================================================
# Topological Invariants (Betti Numbers)
# =============================================================================

@dataclass
class TopologicalInvariants:
    """
    Betti numbers characterizing network topology.

    β₀: Number of connected components (should be 1 for a connected network)
    β₁: Number of independent cycles/loops (redundancy measure)
    β₂: Number of voids (typically 0 for neural networks)

    For antifragility:
    - β₀ = 1 (fully connected)
    - β₁ ≥ min_loops (redundant pathways for fault tolerance)
    """
    beta_0: int = 1  # Connected components
    beta_1: int = 0  # Independent cycles
    beta_2: int = 0  # Voids (usually 0)

    # Derived properties
    euler_characteristic: int = 0  # χ = β₀ - β₁ + β₂

    def __post_init__(self):
        self.euler_characteristic = self.beta_0 - self.beta_1 + self.beta_2

    def satisfies(self, constraints: 'TopologicalConstraints') -> Tuple[bool, List[str]]:
        """Check if invariants satisfy constraints."""
        violations = []

        if self.beta_0 > constraints.max_components:
            violations.append(f"β₀={self.beta_0} > max_components={constraints.max_components}")

        if self.beta_1 < constraints.min_loops:
            violations.append(f"β₁={self.beta_1} < min_loops={constraints.min_loops}")

        if constraints.require_connected and self.beta_0 != 1:
            violations.append(f"Network disconnected: β₀={self.beta_0} ≠ 1")

        return len(violations) == 0, violations


@dataclass
class TopologicalConstraints:
    """
    SMT-style constraints for network topology.

    These are the formal requirements that must be satisfied
    for any structural change to be accepted.
    """
    # Connectivity constraints
    require_connected: bool = True  # β₀ must equal 1
    max_components: int = 1         # Maximum allowed connected components

    # Redundancy constraints
    min_loops: int = 0              # Minimum β₁ (cycles) for fault tolerance
    max_loops: int = 100            # Maximum to prevent over-complexity

    # Stability constraints
    max_spectral_radius: float = 1.0  # ||J||_* < 1 for stability
    min_spectral_radius: float = 0.5  # Avoid over-damped dynamics

    # Capacity constraints
    max_neurons: int = 10000
    max_connections: int = 100000
    min_sparsity: float = 0.9       # At least 90% sparse


# =============================================================================
# Graph Representation for Topology Analysis
# =============================================================================

@dataclass
class NetworkGraph:
    """
    Simplified graph representation for topological analysis.

    Uses adjacency list for efficiency.
    """
    num_nodes: int
    edges: Set[Tuple[int, int]]  # (source, target) pairs

    @classmethod
    def from_weight_matrix(cls, W: Any, threshold: float = 1e-6) -> 'NetworkGraph':
        """Create graph from weight matrix."""
        import numpy as np

        if hasattr(W, 'numpy'):
            W = W.numpy()

        num_nodes = W.shape[0]
        edges = set()

        for i in range(num_nodes):
            for j in range(num_nodes):
                if abs(W[i, j]) > threshold:
                    edges.add((i, j))

        return cls(num_nodes=num_nodes, edges=edges)

    def add_edge(self, source: int, target: int) -> 'NetworkGraph':
        """Return new graph with edge added."""
        new_edges = self.edges.copy()
        new_edges.add((source, target))
        return NetworkGraph(num_nodes=self.num_nodes, edges=new_edges)

    def remove_edge(self, source: int, target: int) -> 'NetworkGraph':
        """Return new graph with edge removed."""
        new_edges = self.edges.copy()
        new_edges.discard((source, target))
        return NetworkGraph(num_nodes=self.num_nodes, edges=new_edges)

    def add_node(self) -> 'NetworkGraph':
        """Return new graph with node added."""
        return NetworkGraph(num_nodes=self.num_nodes + 1, edges=self.edges.copy())

    def remove_node(self, node: int) -> 'NetworkGraph':
        """Return new graph with node removed."""
        new_edges = {(s, t) for s, t in self.edges if s != node and t != node}
        # Renumber nodes after removal
        def renumber(n):
            return n if n < node else n - 1
        new_edges = {(renumber(s), renumber(t)) for s, t in new_edges}
        return NetworkGraph(num_nodes=self.num_nodes - 1, edges=new_edges)

    def get_adjacency_list(self) -> Dict[int, List[int]]:
        """Get adjacency list (undirected)."""
        adj = {i: [] for i in range(self.num_nodes)}
        for s, t in self.edges:
            adj[s].append(t)
            adj[t].append(s)  # Treat as undirected for topology
        return adj

    def sparsity(self) -> float:
        """Compute sparsity (fraction of zero entries)."""
        max_edges = self.num_nodes * self.num_nodes
        if max_edges == 0:
            return 1.0
        return 1.0 - len(self.edges) / max_edges


# =============================================================================
# Topological Analysis (Betti Number Computation)
# =============================================================================

class TopologyAnalyzer:
    """
    Compute topological invariants of network graphs.

    Uses Union-Find for β₀ and cycle detection for β₁.
    """

    def __init__(self):
        pass

    def compute_beta_0(self, graph: NetworkGraph) -> int:
        """
        Compute β₀ (number of connected components) using Union-Find.

        O(n α(n)) where α is inverse Ackermann function.
        """
        parent = list(range(graph.num_nodes))
        rank = [0] * graph.num_nodes

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return
            # Union by rank
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1

        # Process all edges (undirected)
        for s, t in graph.edges:
            union(s, t)

        # Count unique components
        components = len(set(find(i) for i in range(graph.num_nodes)))
        return components

    def compute_beta_1(self, graph: NetworkGraph) -> int:
        """
        Compute β₁ (number of independent cycles).

        For a connected graph: β₁ = |E| - |V| + 1
        For general graph: β₁ = |E| - |V| + β₀

        This is the cyclomatic complexity / circuit rank.
        """
        num_edges = len(graph.edges) // 2  # Undirected edges
        # Actually count unique undirected edges
        undirected_edges = set()
        for s, t in graph.edges:
            undirected_edges.add((min(s, t), max(s, t)))
        num_edges = len(undirected_edges)

        beta_0 = self.compute_beta_0(graph)

        # β₁ = |E| - |V| + β₀ (for undirected graph)
        beta_1 = num_edges - graph.num_nodes + beta_0
        return max(0, beta_1)  # Can't be negative

    def compute_invariants(self, graph: NetworkGraph) -> TopologicalInvariants:
        """Compute all topological invariants."""
        beta_0 = self.compute_beta_0(graph)
        beta_1 = self.compute_beta_1(graph)

        return TopologicalInvariants(
            beta_0=beta_0,
            beta_1=beta_1,
            beta_2=0,  # Always 0 for 1D simplicial complex (graph)
        )


# =============================================================================
# SMT Constraint Solver (Simplified)
# =============================================================================

class SMTConstraintResult(Enum):
    """Result of SMT constraint check."""
    SAT = "satisfiable"
    UNSAT = "unsatisfiable"
    UNKNOWN = "unknown"
    TIMEOUT = "timeout"


@dataclass
class SMTCheckResult:
    """Result of SMT constraint verification."""
    result: SMTConstraintResult
    satisfied_constraints: List[str]
    violated_constraints: List[str]
    model: Optional[Dict[str, Any]] = None  # Satisfying assignment if SAT
    proof: Optional[str] = None  # Proof if UNSAT


class SMTConstraintChecker:
    """
    SMT-style constraint checker for structural changes.

    In production, this would use Z3 or CVC5.
    Currently implements direct constraint evaluation.
    """

    def __init__(self, constraints: TopologicalConstraints):
        self.constraints = constraints
        self.topology = TopologyAnalyzer()

    def check_connectivity(self, graph: NetworkGraph) -> Tuple[bool, str]:
        """Check connectivity constraint (β₀ = 1)."""
        beta_0 = self.topology.compute_beta_0(graph)

        if self.constraints.require_connected and beta_0 != 1:
            return False, f"UNSAT: Network disconnected (β₀={beta_0}, required=1)"

        if beta_0 > self.constraints.max_components:
            return False, f"UNSAT: Too many components (β₀={beta_0} > max={self.constraints.max_components})"

        return True, f"SAT: Connectivity (β₀={beta_0})"

    def check_redundancy(self, graph: NetworkGraph) -> Tuple[bool, str]:
        """Check redundancy constraint (β₁ ≥ min_loops)."""
        beta_1 = self.topology.compute_beta_1(graph)

        if beta_1 < self.constraints.min_loops:
            return False, f"UNSAT: Insufficient redundancy (β₁={beta_1} < min={self.constraints.min_loops})"

        if beta_1 > self.constraints.max_loops:
            return False, f"UNSAT: Over-complex topology (β₁={beta_1} > max={self.constraints.max_loops})"

        return True, f"SAT: Redundancy (β₁={beta_1})"

    def check_capacity(self, graph: NetworkGraph) -> Tuple[bool, str]:
        """Check capacity constraints."""
        if graph.num_nodes > self.constraints.max_neurons:
            return False, f"UNSAT: Too many neurons ({graph.num_nodes} > max={self.constraints.max_neurons})"

        if len(graph.edges) > self.constraints.max_connections:
            return False, f"UNSAT: Too many connections ({len(graph.edges)} > max={self.constraints.max_connections})"

        sparsity = graph.sparsity()
        if sparsity < self.constraints.min_sparsity:
            return False, f"UNSAT: Insufficient sparsity ({sparsity:.2%} < min={self.constraints.min_sparsity:.2%})"

        return True, f"SAT: Capacity (n={graph.num_nodes}, e={len(graph.edges)}, s={sparsity:.2%})"

    def check_all(self, graph: NetworkGraph) -> SMTCheckResult:
        """Run all constraint checks."""
        satisfied = []
        violated = []

        # Check each constraint category
        checks = [
            self.check_connectivity(graph),
            self.check_redundancy(graph),
            self.check_capacity(graph),
        ]

        for ok, msg in checks:
            if ok:
                satisfied.append(msg)
            else:
                violated.append(msg)

        if violated:
            return SMTCheckResult(
                result=SMTConstraintResult.UNSAT,
                satisfied_constraints=satisfied,
                violated_constraints=violated,
                proof="; ".join(violated),
            )

        return SMTCheckResult(
            result=SMTConstraintResult.SAT,
            satisfied_constraints=satisfied,
            violated_constraints=[],
            model={
                'beta_0': self.topology.compute_beta_0(graph),
                'beta_1': self.topology.compute_beta_1(graph),
                'num_nodes': graph.num_nodes,
                'num_edges': len(graph.edges),
            },
        )


# =============================================================================
# Proof-Guided Updater (PGU)
# =============================================================================

class VerificationResult(Enum):
    """Result of PGU verification."""
    VERIFIED = "verified"
    REJECTED = "rejected"
    PENDING = "pending"


@dataclass
class PGUVerificationResult:
    """Complete verification result from PGU."""
    result: VerificationResult
    proposal_id: str
    before_invariants: TopologicalInvariants
    after_invariants: TopologicalInvariants
    smt_result: SMTCheckResult
    spectral_check: Optional[Dict[str, float]] = None
    verification_time_ms: float = 0.0


@dataclass
class PGUConfig:
    """Configuration for Proof-Guided Updater."""
    # Topological constraints
    constraints: TopologicalConstraints = field(default_factory=TopologicalConstraints)

    # Spectral constraints
    check_spectral: bool = True
    max_spectral_change: float = 0.2  # Max change in ||J||_*

    # Caching
    enable_cache: bool = True
    cache_size: int = 1000

    # Safety
    strict_mode: bool = True  # Reject on any violation
    allow_degradation: bool = False  # Allow temporary β₁ decrease


class ProofGuidedUpdater:
    """
    The PGU: Active Safety Gate for Structural Changes.

    Verifies that proposed structural changes maintain:
    1. Topological invariants (β₀, β₁)
    2. SMT constraints (connectivity, redundancy, capacity)
    3. Spectral stability (||J||_* bounds)

    Only PGU-verified changes can be applied to the model.
    """

    def __init__(self, config: Optional[PGUConfig] = None):
        if config is None:
            config = PGUConfig()

        self.config = config
        self.topology = TopologyAnalyzer()
        self.smt = SMTConstraintChecker(config.constraints)

        # Statistics
        self._verified_count = 0
        self._rejected_count = 0
        self._cache: Dict[str, PGUVerificationResult] = {}

    def _compute_proposal_hash(self, proposal: Dict[str, Any], graph: NetworkGraph) -> str:
        """Compute cache key for proposal."""
        import hashlib
        key = f"{proposal.get('type', '')}:{proposal.get('params', '')}:{graph.num_nodes}:{len(graph.edges)}"
        return hashlib.md5(key.encode()).hexdigest()[:16]

    def _apply_proposal_to_graph(
        self,
        proposal: Dict[str, Any],
        graph: NetworkGraph,
    ) -> NetworkGraph:
        """
        Simulate applying proposal to graph.

        Returns new graph without modifying original.
        """
        change_type = proposal.get('type', '')
        params = proposal.get('params', {})

        if change_type == 'add_connection':
            source = params.get('source', 0)
            target = params.get('target', 0)
            return graph.add_edge(source, target)

        elif change_type == 'remove_connection':
            source = params.get('source', 0)
            target = params.get('target', 0)
            return graph.remove_edge(source, target)

        elif change_type == 'add_neuron':
            return graph.add_node()

        elif change_type == 'remove_neuron':
            node = params.get('node', 0)
            return graph.remove_node(node)

        else:
            # Unknown change type - return unchanged
            logger.warning(f"PGU: Unknown change type '{change_type}'")
            return graph

    def verify_proposal(
        self,
        proposal: Dict[str, Any],
        current_graph: NetworkGraph,
        current_spectral_norm: Optional[float] = None,
    ) -> PGUVerificationResult:
        """
        Verify a structural change proposal.

        This is the MAIN ENTRY POINT for the safety gate.

        Args:
            proposal: Dict with 'type' and 'params' keys
            current_graph: Current network topology
            current_spectral_norm: Current ||J||_* (optional)

        Returns:
            PGUVerificationResult with verification status
        """
        import time
        t_start = time.time()

        proposal_id = proposal.get('id', self._compute_proposal_hash(proposal, current_graph))

        # Check cache
        if self.config.enable_cache and proposal_id in self._cache:
            cached = self._cache[proposal_id]
            logger.debug(f"PGU: Cache hit for proposal {proposal_id}")
            return cached

        # Compute before invariants
        before_invariants = self.topology.compute_invariants(current_graph)

        # Apply proposal to get new graph
        new_graph = self._apply_proposal_to_graph(proposal, current_graph)

        # Compute after invariants
        after_invariants = self.topology.compute_invariants(new_graph)

        # Run SMT constraint check
        smt_result = self.smt.check_all(new_graph)

        # Check spectral constraints (if enabled)
        spectral_check = None
        if self.config.check_spectral and current_spectral_norm is not None:
            # Estimate spectral change (simplified - would use actual Jacobian in production)
            edge_delta = len(new_graph.edges) - len(current_graph.edges)
            estimated_spectral_change = abs(edge_delta) * 0.01  # Rough estimate

            spectral_check = {
                'current': current_spectral_norm,
                'estimated_change': estimated_spectral_change,
                'max_allowed': self.config.max_spectral_change,
                'within_bounds': estimated_spectral_change <= self.config.max_spectral_change,
            }

        # Determine final result
        if smt_result.result == SMTConstraintResult.UNSAT:
            result = VerificationResult.REJECTED
            self._rejected_count += 1
        elif spectral_check and not spectral_check['within_bounds']:
            result = VerificationResult.REJECTED
            self._rejected_count += 1
            smt_result.violated_constraints.append(
                f"Spectral: change {spectral_check['estimated_change']:.4f} > max {spectral_check['max_allowed']}"
            )
        elif not self.config.allow_degradation and after_invariants.beta_1 < before_invariants.beta_1:
            result = VerificationResult.REJECTED
            self._rejected_count += 1
            smt_result.violated_constraints.append(
                f"Degradation: β₁ decreased from {before_invariants.beta_1} to {after_invariants.beta_1}"
            )
        else:
            result = VerificationResult.VERIFIED
            self._verified_count += 1

        t_elapsed = (time.time() - t_start) * 1000

        verification_result = PGUVerificationResult(
            result=result,
            proposal_id=proposal_id,
            before_invariants=before_invariants,
            after_invariants=after_invariants,
            smt_result=smt_result,
            spectral_check=spectral_check,
            verification_time_ms=t_elapsed,
        )

        # Cache result
        if self.config.enable_cache:
            if len(self._cache) >= self.config.cache_size:
                # Simple FIFO eviction
                oldest = next(iter(self._cache))
                del self._cache[oldest]
            self._cache[proposal_id] = verification_result

        # Log result
        if result == VerificationResult.VERIFIED:
            logger.info(f"PGU VERIFIED: {proposal.get('type')} (β₀={after_invariants.beta_0}, β₁={after_invariants.beta_1})")
        else:
            logger.warning(f"PGU REJECTED: {proposal.get('type')} - {smt_result.violated_constraints}")

        return verification_result

    def get_stats(self) -> Dict[str, Any]:
        """Get PGU statistics."""
        return {
            'verified_count': self._verified_count,
            'rejected_count': self._rejected_count,
            'cache_size': len(self._cache),
            'acceptance_rate': self._verified_count / max(1, self._verified_count + self._rejected_count),
        }


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate PGU with SMT constraints."""
    print("=" * 70)
    print("Proof-Guided Updater (PGU) Demo")
    print("SMT Topological Constraints for Structural Safety")
    print("=" * 70)

    # Create PGU with constraints
    constraints = TopologicalConstraints(
        require_connected=True,
        min_loops=2,  # Require at least 2 redundant paths
        max_loops=10,
        max_neurons=1000,
        min_sparsity=0.8,
    )
    config = PGUConfig(constraints=constraints)
    pgu = ProofGuidedUpdater(config)

    print(f"\nConstraints configured:")
    print(f"  - require_connected: {constraints.require_connected}")
    print(f"  - min_loops (β₁): {constraints.min_loops}")
    print(f"  - max_loops: {constraints.max_loops}")
    print(f"  - min_sparsity: {constraints.min_sparsity:.0%}")

    # Create initial graph with some structure
    # Small network: 10 nodes, ring topology + some extra edges for loops
    edges = set()
    num_nodes = 10

    # Ring (creates 1 loop)
    for i in range(num_nodes):
        edges.add((i, (i + 1) % num_nodes))
        edges.add(((i + 1) % num_nodes, i))

    # Add cross edges (creates more loops)
    edges.add((0, 5))
    edges.add((5, 0))
    edges.add((2, 7))
    edges.add((7, 2))
    edges.add((3, 8))
    edges.add((8, 3))

    graph = NetworkGraph(num_nodes=num_nodes, edges=edges)

    # Compute initial topology
    analyzer = TopologyAnalyzer()
    initial = analyzer.compute_invariants(graph)

    print(f"\nInitial network:")
    print(f"  - Nodes: {graph.num_nodes}")
    print(f"  - Edges: {len(graph.edges)}")
    print(f"  - β₀ (components): {initial.beta_0}")
    print(f"  - β₁ (loops): {initial.beta_1}")
    print(f"  - Sparsity: {graph.sparsity():.1%}")

    # Test various proposals
    print("\n" + "-" * 70)
    print("Testing Proposals")
    print("-" * 70)

    proposals = [
        {'type': 'add_connection', 'params': {'source': 1, 'target': 6}, 'desc': 'Add edge 1→6'},
        {'type': 'remove_connection', 'params': {'source': 0, 'target': 1}, 'desc': 'Remove edge 0→1'},
        {'type': 'add_neuron', 'params': {}, 'desc': 'Add new neuron'},
        {'type': 'remove_connection', 'params': {'source': 0, 'target': 5}, 'desc': 'Remove cross-edge 0→5'},
        {'type': 'remove_connection', 'params': {'source': 2, 'target': 7}, 'desc': 'Remove cross-edge 2→7'},
    ]

    for proposal in proposals:
        print(f"\nProposal: {proposal['desc']}")
        result = pgu.verify_proposal(proposal, graph, current_spectral_norm=0.8)

        print(f"  Result: {result.result.value}")
        print(f"  β₀: {result.before_invariants.beta_0} → {result.after_invariants.beta_0}")
        print(f"  β₁: {result.before_invariants.beta_1} → {result.after_invariants.beta_1}")

        if result.smt_result.violated_constraints:
            print(f"  Violations: {result.smt_result.violated_constraints}")

        print(f"  Time: {result.verification_time_ms:.2f}ms")

        # Apply verified changes to graph
        if result.result == VerificationResult.VERIFIED:
            graph = pgu._apply_proposal_to_graph(proposal, graph)

    # Final stats
    print("\n" + "-" * 70)
    print("Final Statistics")
    print("-" * 70)
    stats = pgu.get_stats()
    print(f"\nPGU Statistics:")
    print(f"  - Verified: {stats['verified_count']}")
    print(f"  - Rejected: {stats['rejected_count']}")
    print(f"  - Acceptance rate: {stats['acceptance_rate']:.1%}")

    final = analyzer.compute_invariants(graph)
    print(f"\nFinal network topology:")
    print(f"  - β₀: {final.beta_0}")
    print(f"  - β₁: {final.beta_1}")

    print("\n" + "=" * 70)
    print("PGU Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    demo()
