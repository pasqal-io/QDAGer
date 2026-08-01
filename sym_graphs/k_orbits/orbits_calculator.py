import itertools
from collections.abc import Iterable, Sequence

import multiset
import networkx as nx
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.combinatorics.permutations import Permutation


def perm_from_cycles(cycles: Sequence[Sequence[int]], points: Sequence[int]) -> Permutation:
    """Build a permutation (on `points`) from a list of cycles."""
    val2idx = {v: i for i, v in enumerate(points)}
    n = len(points)
    img = list(range(n))
    for cyc in cycles:
        if len(cyc) < 2:
            continue
        cyc_idx = [val2idx[a] for a in cyc]
        # (a b c) sends a→b, b→c, c→a
        for u, v in zip(cyc_idx, cyc_idx[1:] + cyc_idx[:1]):
            img[u] = v
    return Permutation(img)


class OrbitEngine:
    def __init__(self, G: nx.Graph, gens: list[list[tuple[int]]]):
        """
        G: any indexable object where len(G) gives the alphabet size
        all_cycles: list of permutations written as lists of cycles, e.g.
                    [
                      [[0,1,2]],                  # a 3-cycle
                      [[0,3],[1,2]],              # a product of 2-cycles
                      ...
                    ]
        """
        self.G = G
        points = sorted({a for gen in gens for cyc in gen for a in cyc})
        perms = [perm_from_cycles(cyc, points) for cyc in gens]
        Group = PermutationGroup(perms)
        # enumerate all elements (Dimino algorithm)
        elems = list(Group.generate(method="dimino"))  # each is a sympy Permutation
        # sanity
        assert Group.order() == len(elems)
        # cycles (with fixed points)
        all_cycles = [p.full_cyclic_form for p in elems]
        self.all_cycles = all_cycles

    # ----- helpers -----
    @staticmethod
    def fs_to_list(fs: Iterable[tuple[int, int]]) -> list[int]:
        out: list[int] = []
        for e, cnt in fs:
            out.extend([e] * cnt)
        return out

    @staticmethod
    def cycle_to_perm(cycles: list[list[int]]) -> dict[int, int]:
        R: dict[int, int] = {}
        for cycle in cycles:
            if len(cycle) < 2:
                R[cycle[0]] = cycle[0]
            elif len(cycle) == 2:
                R[cycle[0]] = cycle[1]
                R[cycle[1]] = cycle[0]
            else:
                for i in range(len(cycle) - 1):
                    R[cycle[i]] = cycle[i + 1]
                R[cycle[-1]] = cycle[0]
        return R

    def orbit_of_elem(self, e: int, all_cycles: list[list[list[int]]] = None) -> set[int]:
        if all_cycles is None:
            all_cycles = self.all_cycles
        orbit: set[int] = set()
        for cycles in all_cycles:
            vanilla_perm = self.cycle_to_perm(cycles)
            orbit.add(vanilla_perm[e])
        return orbit

    def orbit_of_multiset(
        self,
        ms: Iterable[int],
        all_cycles: list[list[list[int]]] = None,
    ) -> frozenset[frozenset[tuple[int, int]]]:
        if all_cycles is None:
            all_cycles = self.all_cycles
        output: set[frozenset[tuple[int, int]]] = set()
        for cycles in all_cycles:
            vanilla_perm = self.cycle_to_perm(cycles)
            elem = [vanilla_perm[u] for u in ms]
            MS = multiset.Multiset(elem)
            output.add(frozenset(MS.items()))
        return frozenset(output)

    # ----- main routine -----
    def orbits_calculator(self, k: int) -> list[list[list[int]]]:
        n = len(self.G)
        all_elems = list(itertools.product(range(n), repeat=k))
        all_multisets = {frozenset(multiset.Multiset(e).items()) for e in all_elems}

        out: list[frozenset[frozenset[tuple[int, int]]]] = []
        seen: set[frozenset[tuple[int, int]]] = set()

        for ms in all_elems:
            ms_key = frozenset(multiset.Multiset(ms).items())
            if ms_key not in seen:
                orbit = self.orbit_of_multiset(ms)  # frozenset of multiset fingerprints
                seen = seen.union(orbit)  # accumulate seen multisets
                out.append(orbit)
            if seen == all_multisets:
                break

        FS = frozenset(out)
        return [[self.fs_to_list(e) for e in list(Orb)] for Orb in list(FS)]

    # ----- convenience -----
    def pairs(self) -> list[tuple[int, int]]:
        n = len(self.G)
        return list(itertools.product(range(n), repeat=2))

    def triples(self) -> list[tuple[int, int, int]]:
        n = len(self.G)
        return list(itertools.product(range(n), repeat=3))
