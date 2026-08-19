"""Guards that keep link traversal bounded and crash-proof.

The BFS must not explore an unbounded subgraph (and, with include_semantic,
run one similarity search per visited node), and notebook flattening must not
recurse forever on a malformed parent cycle.
"""

import unittest

from src.joplin.joplin_api import JoplinNotebook
from src.joplin.joplin_links import (
    MAX_EXPANSIONS,
    LinkGraph,
    find_neighbours,
    flatten_notebook_paths,
)


class ExpansionCapTests(unittest.TestCase):
    def _star(self, children: int) -> LinkGraph:
        # Root "0" links out to `children` nodes, each of which links onward to
        # a private leaf — enough frontier at hop 2 to blow past the cap.
        graph = LinkGraph()
        graph.titles["0"] = "root"
        for i in range(1, children + 1):
            cid = str(i)
            graph.titles[cid] = f"c{i}"
            graph.outgoing.setdefault("0", set()).add(cid)
            graph.incoming.setdefault(cid, set()).add("0")
            leaf = f"leaf{i}"
            graph.titles[leaf] = leaf
            graph.outgoing.setdefault(cid, set()).add(leaf)
            graph.incoming.setdefault(leaf, set()).add(cid)
        return graph

    def test_expansion_cap_bounds_semantic_calls(self):
        graph = self._star(MAX_EXPANSIONS + 100)
        calls = {"n": 0}

        def provider(current):
            calls["n"] += 1
            return {}

        result = find_neighbours(
            graph, note_id="0", direction="both", depth=3, limit=50,
            semantic_provider=provider,
        )

        # One provider call per expanded node, and expansions never exceed the
        # cap — so the per-node similarity search can't run unbounded.
        self.assertLessEqual(calls["n"], MAX_EXPANSIONS)
        self.assertTrue(result["truncated"])


class NotebookCycleTests(unittest.TestCase):
    def test_parent_cycle_does_not_recurse_forever(self):
        # A malformed tree where a child points back at an ancestor.
        a = JoplinNotebook(id="a", title="A", children=[])
        b = JoplinNotebook(id="b", title="B", children=[a])
        a.children = [b]  # cycle: a -> b -> a

        paths = flatten_notebook_paths([a])

        # Terminates, and each id appears once.
        self.assertEqual(set(paths), {"a", "b"})


if __name__ == "__main__":
    unittest.main()
