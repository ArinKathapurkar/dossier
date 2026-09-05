"""Entity graph over the filing corpus.

Why a graph at all: vector and keyword retrieval both answer "which passage looks like
this query". Neither answers "which companies does the target actually depend on, and
what did *those* filings say" -- that is a join across documents, and it is where
multi-company diligence questions live. The graph gives one hop of that join, with the
provenance chunk ids carried on every edge so nothing the graph surfaces is uncited.

Two backends behind one interface:
  * NetworkXGraphStore -- in-process, persisted to JSON. The default, and what the tests
    and CI run against.
  * Neo4jGraphStore    -- Cypher against the docker-compose container. Same interface;
    an integration test skips itself when the container is unreachable.

The interface is intentionally small (upsert / find_entities / expand / chunks_for /
neighbors) so a third backend would be a day's work.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from ..config import get_config

ENTITY_TYPES = (
    "Company",
    "Segment",
    "Product",
    "Customer",
    "Supplier",
    "Competitor",
    "Geography",
    "RiskFactor",
    "Regulator",
)
RELATION_TYPES = (
    "COMPETES_WITH",
    "DEPENDS_ON",
    "SELLS_TO",
    "OPERATES_IN",
    "EXPOSED_TO",
    "REGULATED_BY",
    "HAS_SEGMENT",
    "OFFERS",
)

_SUFFIXES = re.compile(
    r"\b(inc|inc\.|corp|corp\.|corporation|co|co\.|company|plc|ltd|ltd\.|llc|l\.l\.c\.|holdings|group|sa|nv|ag)\b",
    re.I,
)
_PUNCT = re.compile(r"[^a-z0-9 ]")
_WS = re.compile(r"\s+")


def normalize_entity(name: str) -> str:
    """Casefold, drop corporate suffixes and punctuation, squeeze whitespace.

    'Apple Inc.' and 'APPLE, INC' collapse to 'apple'. Without this the graph fragments
    into near-duplicate nodes and expansion returns nothing useful.
    """
    s = _PUNCT.sub(" ", (name or "").lower())
    s = _SUFFIXES.sub(" ", s)
    return _WS.sub(" ", s).strip()


@dataclass
class Entity:
    name: str
    type: str
    chunk_ids: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return normalize_entity(self.name)


@dataclass
class Relation:
    src: str
    rel: str
    dst: str
    chunk_ids: list[str] = field(default_factory=list)


class GraphStore(ABC):
    @abstractmethod
    def upsert(self, entities: list[Entity], relations: list[Relation]) -> None: ...

    @abstractmethod
    def find_entities(self, text: str, limit: int = 10) -> list[str]: ...

    @abstractmethod
    def expand(self, ids: list[str], hops: int = 1, rel_filter: list[str] | None = None) -> dict[str, int]:
        """Return `{entity_id: hop_distance}` for the neighbourhood of `ids`."""

    @abstractmethod
    def chunks_for(self, entity_ids: list[str]) -> list[str]: ...

    @abstractmethod
    def neighbors(self, entity: str) -> list[tuple[str, str, str]]:
        """Return `(src, rel, dst)` triples touching `entity`."""

    @abstractmethod
    def stats(self) -> dict: ...

    def degree(self, entity_id: str) -> int:
        return len(self.neighbors(entity_id))


class NetworkXGraphStore(GraphStore):
    def __init__(self, path: Path | None = None, autoload: bool = True):
        import networkx as nx

        self.path = path or get_config().paths.graph_json
        self.g = nx.MultiDiGraph()
        if autoload and self.path.exists():
            self.load()

    # -- persistence ---------------------------------------------------------------
    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "nodes": [
                {"id": n, "name": d.get("name", n), "type": d.get("type", ""), "chunk_ids": sorted(d.get("chunk_ids", []))}
                for n, d in self.g.nodes(data=True)
            ],
            "edges": [
                {"src": u, "rel": d.get("rel", ""), "dst": v, "chunk_ids": sorted(d.get("chunk_ids", []))}
                for u, v, d in self.g.edges(data=True)
            ],
        }
        self.path.write_text(json.dumps(payload, indent=0, sort_keys=True))

    def load(self) -> NetworkXGraphStore:
        payload = json.loads(self.path.read_text())
        self.g.clear()
        for n in payload.get("nodes", []):
            self.g.add_node(n["id"], name=n.get("name", n["id"]), type=n.get("type", ""), chunk_ids=set(n.get("chunk_ids", [])))
        for e in payload.get("edges", []):
            self.g.add_edge(e["src"], e["dst"], key=e["rel"], rel=e["rel"], chunk_ids=set(e.get("chunk_ids", [])))
        return self

    # -- interface -----------------------------------------------------------------
    def upsert(self, entities: list[Entity], relations: list[Relation]) -> None:
        for e in entities:
            eid = e.id
            if not eid:
                continue
            if eid in self.g.nodes:
                node = self.g.nodes[eid]
                node.setdefault("chunk_ids", set()).update(e.chunk_ids)
                if not node.get("type"):
                    node["type"] = e.type
            else:
                self.g.add_node(eid, name=e.name, type=e.type, chunk_ids=set(e.chunk_ids))
        for r in relations:
            src, dst = normalize_entity(r.src), normalize_entity(r.dst)
            if not src or not dst or src == dst:
                continue
            for node_id, raw in ((src, r.src), (dst, r.dst)):
                if node_id not in self.g.nodes:
                    self.g.add_node(node_id, name=raw, type="", chunk_ids=set())
            if self.g.has_edge(src, dst, key=r.rel):
                self.g.edges[src, dst, r.rel].setdefault("chunk_ids", set()).update(r.chunk_ids)
            else:
                self.g.add_edge(src, dst, key=r.rel, rel=r.rel, chunk_ids=set(r.chunk_ids))

    def find_entities(self, text: str, limit: int = 10) -> list[str]:
        norm = normalize_entity(text)
        if not norm:
            return []
        hits: list[str] = []
        if norm in self.g.nodes:
            hits.append(norm)
        # substring containment: a query naming several entities should find each of them
        for node in self.g.nodes:
            if node in hits:
                continue
            if node and (node in norm or (len(node) > 4 and norm in node)):
                hits.append(node)
        if len(hits) < limit:
            try:
                from rapidfuzz import fuzz, process

                candidates = [n for n in self.g.nodes if n not in hits]
                for name, score, _ in process.extract(
                    norm, candidates, scorer=fuzz.token_set_ratio, limit=limit
                ):
                    if score >= get_config().fuzzy_threshold:
                        hits.append(name)
            except ImportError:
                pass
        return hits[:limit]

    def expand(self, ids: list[str], hops: int = 1, rel_filter: list[str] | None = None) -> dict[str, int]:
        hops = max(1, min(int(hops), 2))
        seen: dict[str, int] = {i: 0 for i in ids if i in self.g.nodes}
        frontier = list(seen)
        for hop in range(1, hops + 1):
            nxt: list[str] = []
            for node in frontier:
                for _, dst, d in self.g.out_edges(node, data=True):
                    if rel_filter and d.get("rel") not in rel_filter:
                        continue
                    if dst not in seen:
                        seen[dst] = hop
                        nxt.append(dst)
                for src, _, d in self.g.in_edges(node, data=True):
                    if rel_filter and d.get("rel") not in rel_filter:
                        continue
                    if src not in seen:
                        seen[src] = hop
                        nxt.append(src)
            frontier = nxt
            if not frontier:
                break
        return seen

    def chunks_for(self, entity_ids: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for eid in entity_ids:
            if eid not in self.g.nodes:
                continue
            for cid in sorted(self.g.nodes[eid].get("chunk_ids", set())):
                if cid not in seen:
                    seen.add(cid)
                    out.append(cid)
            for _, _, d in self.g.out_edges(eid, data=True):
                for cid in sorted(d.get("chunk_ids", set())):
                    if cid not in seen:
                        seen.add(cid)
                        out.append(cid)
        return out

    def neighbors(self, entity: str) -> list[tuple[str, str, str]]:
        eid = normalize_entity(entity)
        if eid not in self.g.nodes:
            return []
        out = [(u, d.get("rel", ""), v) for u, v, d in self.g.out_edges(eid, data=True)]
        out += [(u, d.get("rel", ""), v) for u, v, d in self.g.in_edges(eid, data=True)]
        return sorted(set(out))

    def stats(self) -> dict:
        return {"entities": self.g.number_of_nodes(), "relations": self.g.number_of_edges(), "backend": "networkx"}


class Neo4jGraphStore(GraphStore):
    """Cypher-backed implementation. Same semantics; used when a container is running."""

    def __init__(self, uri: str | None = None, user: str | None = None, password: str | None = None):
        from neo4j import GraphDatabase

        cfg = get_config()
        self.driver = GraphDatabase.driver(
            uri or cfg.neo4j_uri, auth=(user or cfg.neo4j_user, password or cfg.neo4j_password)
        )
        with self.driver.session() as s:
            s.run("CREATE CONSTRAINT dossier_entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")

    @staticmethod
    def available() -> bool:
        try:
            store = Neo4jGraphStore()
            store.driver.verify_connectivity()
            store.close()
            return True
        except Exception:
            return False

    def close(self) -> None:
        try:
            self.driver.close()
        except Exception:
            pass

    def upsert(self, entities: list[Entity], relations: list[Relation]) -> None:
        with self.driver.session() as s:
            s.run(
                """
                UNWIND $rows AS row
                MERGE (e:Entity {id: row.id})
                ON CREATE SET e.name = row.name, e.type = row.type, e.chunk_ids = row.chunk_ids
                ON MATCH  SET e.chunk_ids = apoc.coll.toSet(coalesce(e.chunk_ids, []) + row.chunk_ids)
                """.replace(
                    "apoc.coll.toSet(coalesce(e.chunk_ids, []) + row.chunk_ids)",
                    "coalesce(e.chunk_ids, []) + [c IN row.chunk_ids WHERE NOT c IN coalesce(e.chunk_ids, [])]",
                ),
                rows=[{"id": e.id, "name": e.name, "type": e.type, "chunk_ids": e.chunk_ids} for e in entities if e.id],
            )
            s.run(
                """
                UNWIND $rows AS row
                MERGE (a:Entity {id: row.src}) MERGE (b:Entity {id: row.dst})
                MERGE (a)-[r:REL {rel: row.rel}]->(b)
                ON CREATE SET r.chunk_ids = row.chunk_ids
                ON MATCH  SET r.chunk_ids = coalesce(r.chunk_ids, []) + [c IN row.chunk_ids WHERE NOT c IN coalesce(r.chunk_ids, [])]
                """,
                rows=[
                    {"src": normalize_entity(r.src), "dst": normalize_entity(r.dst), "rel": r.rel, "chunk_ids": r.chunk_ids}
                    for r in relations
                    if normalize_entity(r.src) and normalize_entity(r.dst) and normalize_entity(r.src) != normalize_entity(r.dst)
                ],
            )

    def find_entities(self, text: str, limit: int = 10) -> list[str]:
        norm = normalize_entity(text)
        if not norm:
            return []
        with self.driver.session() as s:
            rows = s.run(
                "MATCH (e:Entity) WHERE e.id = $n OR $n CONTAINS e.id RETURN e.id AS id LIMIT $k",
                n=norm,
                k=limit,
            )
            hits = [r["id"] for r in rows]
        if len(hits) < limit:
            with self.driver.session() as s:
                all_ids = [r["id"] for r in s.run("MATCH (e:Entity) RETURN e.id AS id")]
            try:
                from rapidfuzz import fuzz, process

                for name, score, _ in process.extract(
                    norm, [i for i in all_ids if i not in hits], scorer=fuzz.token_set_ratio, limit=limit
                ):
                    if score >= get_config().fuzzy_threshold:
                        hits.append(name)
            except ImportError:
                pass
        return hits[:limit]

    def expand(self, ids: list[str], hops: int = 1, rel_filter: list[str] | None = None) -> dict[str, int]:
        hops = max(1, min(int(hops), 2))
        where = " AND r.rel IN $rels" if rel_filter else ""
        with self.driver.session() as s:
            rows = s.run(
                f"MATCH (a:Entity) WHERE a.id IN $ids "
                f"MATCH p = (a)-[r:REL*1..{hops}]-(b:Entity) "
                f"WHERE all(r IN relationships(p) WHERE true{where}) "
                f"RETURN b.id AS id, min(length(p)) AS hop",
                ids=ids,
                rels=rel_filter or [],
            )
            out = {i: 0 for i in ids}
            for r in rows:
                out.setdefault(r["id"], r["hop"])
            return out

    def chunks_for(self, entity_ids: list[str]) -> list[str]:
        with self.driver.session() as s:
            rows = s.run(
                "MATCH (e:Entity) WHERE e.id IN $ids "
                "OPTIONAL MATCH (e)-[r:REL]->() "
                "RETURN coalesce(e.chunk_ids, []) AS ec, coalesce(r.chunk_ids, []) AS rc",
                ids=entity_ids,
            )
            seen: list[str] = []
            for r in rows:
                for cid in list(r["ec"]) + list(r["rc"]):
                    if cid not in seen:
                        seen.append(cid)
            return seen

    def neighbors(self, entity: str) -> list[tuple[str, str, str]]:
        eid = normalize_entity(entity)
        with self.driver.session() as s:
            rows = s.run(
                "MATCH (a:Entity {id:$id})-[r:REL]-(b:Entity) RETURN startNode(r).id AS s, r.rel AS rel, endNode(r).id AS d",
                id=eid,
            )
            return sorted({(r["s"], r["rel"], r["d"]) for r in rows})

    def stats(self) -> dict:
        with self.driver.session() as s:
            n = s.run("MATCH (e:Entity) RETURN count(e) AS c").single()["c"]
            m = s.run("MATCH ()-[r:REL]->() RETURN count(r) AS c").single()["c"]
        return {"entities": n, "relations": m, "backend": "neo4j"}


def open_graph_store(backend: str | None = None, allow_fallback: bool = True) -> GraphStore:
    """Open the requested backend, falling back to NetworkX with a visible span.

    Neo4j being down mid-run is a realistic failure. Rather than crash the run, we degrade
    to the local graph and record a `fallback` span so the degradation is countable in
    eval and visible in `dossier trace`.
    """
    backend = (backend or get_config().graph_backend or "networkx").lower()
    if backend != "neo4j":
        return NetworkXGraphStore()
    try:
        store = Neo4jGraphStore()
        store.driver.verify_connectivity()
        return store
    except Exception as exc:
        if not allow_fallback:
            raise
        from ..obs.tracer import get_tracer

        get_tracer().event(
            "fallback",
            name="graph_backend",
            attrs={"from": "neo4j", "to": "networkx", "error": type(exc).__name__},
        )
        return NetworkXGraphStore()


def sync_graph(source: GraphStore | None = None, target_backend: str = "neo4j", batch: int = 500) -> dict:
    """Copy a built graph into another backend.

    Exists so the Neo4j path can be exercised against the real graph without paying for
    extraction again: entity extraction is the expensive step, the store is not.
    """
    import time

    source = source or NetworkXGraphStore()
    if target_backend == "neo4j":
        target = Neo4jGraphStore()
    else:
        target = NetworkXGraphStore()
    started = time.time()
    entities = [
        Entity(name=d.get("name", n), type=d.get("type", ""), chunk_ids=sorted(d.get("chunk_ids", set())))
        for n, d in source.g.nodes(data=True)
    ]
    relations = [
        Relation(src=u, rel=d.get("rel", ""), dst=v, chunk_ids=sorted(d.get("chunk_ids", set())))
        for u, v, d in source.g.edges(data=True)
    ]
    for i in range(0, len(entities), batch):
        target.upsert(entities[i : i + batch], [])
    for i in range(0, len(relations), batch):
        target.upsert([], relations[i : i + batch])
    stats = target.stats()
    if hasattr(target, "close"):
        target.close()
    return {"source": source.stats(), "target": stats, "elapsed_s": round(time.time() - started, 2)}
