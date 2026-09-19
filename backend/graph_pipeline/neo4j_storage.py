import logging
import os
from typing import Dict, List, Optional
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langchain_community.graphs.graph_document import GraphDocument

load_dotenv()
logger = logging.getLogger(__name__)


class Neo4jStorageManager:
    """
    Manages connection and persistence to the Neo4j Graph Database
    for the Graph Pipeline.
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        verbose: bool = True,
    ):
        self.uri = uri or os.getenv("NEO4J_URI")
        self.username = username or os.getenv("NEO4J_USERNAME")
        self.password = password or os.getenv("NEO4J_PASSWORD")
        self.database = database or os.getenv("NEO4J_DATABASE", self.username)
        self.verbose = verbose

        if not all([self.uri, self.username, self.password]):
            raise ValueError(
                "Missing Neo4j credentials! Please ensure NEO4J_URI, "
                "NEO4J_USERNAME, and NEO4J_PASSWORD are set in your .env file."
            )

        self._graph: Optional[Neo4jGraph] = None

    def get_graph(self) -> Neo4jGraph:
        if self._graph is None:
            self._graph = Neo4jGraph(
                url=self.uri,
                username=self.username,
                password=self.password,
                database=self.database,
            )
        return self._graph

    def test_connection(self) -> bool:
        if self.verbose:
            print("\n" + "=" * 80)
            print("🔗 [NEO4J CONNECTION CHECK]")
            print(f"   📡 Connecting to URI : {self.uri}")
            print(f"   👤 Database / User   : {self.database}")
            print("-" * 80)

        try:
            graph = self.get_graph()
            res = graph.query("RETURN 1 as test")
            connected = len(res) > 0 and res[0].get("test") == 1

            if self.verbose:
                if connected:
                    stats = self.get_graph_stats()
                    print("   🟢 Connection Status : [SUCCESS] Active & Responsive")
                    print(f"   📊 Existing Graph     : {stats['node_count']} nodes | {stats['relationship_count']} relationships")
                    print("=" * 80 + "\n")
                else:
                    print("   ❌ Connection Status : [FAILED] Unexpected test response")
            return connected
        except Exception as e:
            if self.verbose:
                print(f"   ❌ Neo4j Connection Error: {e}")
            return False

    def add_graph_documents(
        self,
        graph_documents: List[GraphDocument],
        base_entity_label: bool = True,
        include_source: bool = True,
    ) -> int:
        if not graph_documents:
            if self.verbose:
                print("⚠️ [NEO4J STORAGE] No graph documents to add.")
            return 0

        before_stats = self.get_graph_stats()

        if self.verbose:
            print("\n" + "=" * 80)
            print("💾 [NEO4J GRAPH PERSISTENCE]")
            print(f"   📦 Persisting {len(graph_documents)} GraphDocument(s)...")
            print("   🏷️ Setting baseEntityLabel=True and include_source=True")
            print("-" * 80)

        graph = self.get_graph()
        graph.add_graph_documents(
            graph_documents,
            baseEntityLabel=base_entity_label,
            include_source=include_source,
        )

        after_stats = self.get_graph_stats()
        delta_nodes = after_stats["node_count"] - before_stats["node_count"]
        delta_rels = after_stats["relationship_count"] - before_stats["relationship_count"]

        if self.verbose:
            print("✅ Successfully committed graph documents to Neo4j.")
            print(f"📊 Updated Graph State:")
            print(f"   • Total Nodes         : {after_stats['node_count']} ({'+' if delta_nodes >= 0 else ''}{delta_nodes})")
            print(f"   • Total Relationships : {after_stats['relationship_count']} ({'+' if delta_rels >= 0 else ''}{delta_rels})")
            print("=" * 80 + "\n")

        return len(graph_documents)

    def get_graph_stats(self) -> Dict[str, int]:
        try:
            graph = self.get_graph()
            node_res = graph.query("MATCH (n) RETURN count(n) AS node_count")
            rel_res = graph.query("MATCH ()-[r]->() RETURN count(r) AS rel_count")
            return {
                "node_count": node_res[0]["node_count"] if node_res else 0,
                "relationship_count": rel_res[0]["rel_count"] if rel_res else 0,
            }
        except Exception as e:
            return {"node_count": -1, "relationship_count": -1}
