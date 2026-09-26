import logging
import os
import re
import time
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph

from graph_pipeline.neo4j_storage import Neo4jStorageManager

load_dotenv()
logger = logging.getLogger(__name__)


class QuestionAnalysis(BaseModel):
    """Structured extraction of entities, keywords, and relationship types from a user question."""
    entities: List[str] = Field(
        description="Key entities, microservices, databases, routes, or technical concepts mentioned in the query"
    )
    search_keywords: List[str] = Field(
        description="Search keywords or root terms to look up in the graph (e.g. catalog, redis, stream, playback, auth, billing)"
    )
    relationships: List[str] = Field(
        description="Potential graph relationship types (e.g. STORES, MANAGES, CONNECTS_TO, USES, PROVIDES, HANDLES, AUTHENTICATES)"
    )


class GraphRetriever:
    """
    Knowledge Graph Retriever for Neo4j.
    Performs:
      1. LLM-based entity & relation intent extraction
      2. Fuzzy candidate node & relationship matching in Neo4j
      3. Dynamic read-only Cypher query generation & execution (with robust multi-hop fallback)
      4. Structured Graph Fact extraction (Nodes, Edges, Triplet Context)
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        verbose: bool = True,
    ):
        self.model_name = model_name or os.getenv("OPENAI_GENERATIVE_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.verbose = verbose

        self.llm = ChatOpenAI(
            model_name=self.model_name,
            api_key=self.api_key,
            temperature=0.0,
        )
        self.storage_manager = Neo4jStorageManager(verbose=False)

    def is_available(self) -> bool:
        """Check if Neo4j database is accessible."""
        return self.storage_manager.test_connection()

    def extract_analysis(self, question: str) -> Dict[str, Any]:
        """Step 1: Extract entities, keywords, and potential relations from query."""
        structured_llm = self.llm.with_structured_output(QuestionAnalysis)
        analysis: QuestionAnalysis = structured_llm.invoke(question)
        return {
            "entities": analysis.entities,
            "search_keywords": analysis.search_keywords,
            "relationships": analysis.relationships,
        }

    def match_candidates(self, extracted: Dict[str, Any], limit_nodes: int = 15) -> Dict[str, Any]:
        """Step 2: Match extracted entities and keywords against nodes in Neo4j."""
        graph = self.storage_manager.get_graph()
        all_terms = list(dict.fromkeys(extracted.get("entities", []) + extracted.get("search_keywords", [])))

        matched_nodes = []
        if all_terms:
            node_query = """
            UNWIND $terms AS term
            MATCH (n)
            WHERE n.id IS NOT NULL AND toLower(n.id) CONTAINS toLower(term)
            RETURN DISTINCT n.id AS id, labels(n) AS labels
            LIMIT $limit
            """
            try:
                matched_nodes = graph.query(node_query, {"terms": all_terms, "limit": limit_nodes})
            except Exception as e:
                logger.warning(f"Error querying nodes: {e}")

        node_ids = [n["id"] for n in matched_nodes if n.get("id")]
        connected_relationships = []
        if node_ids:
            rel_query = """
            UNWIND $node_ids AS nid
            MATCH (s {id: nid})-[r]->(target)
            WHERE target.id IS NOT NULL
            RETURN DISTINCT s.id AS source, type(r) AS relationship, labels(target) AS target_labels, target.id AS target
            LIMIT 25
            """
            try:
                connected_relationships = graph.query(rel_query, {"node_ids": node_ids[:10]})
            except Exception as e:
                logger.warning(f"Error querying relationships: {e}")

        return {
            "terms_searched": all_terms,
            "matched_nodes": matched_nodes,
            "connected_relationships": connected_relationships,
        }

    def generate_cypher(
        self,
        question: str,
        extracted: Dict[str, Any],
        matched: Dict[str, Any],
    ) -> str:
        """Step 3: Generate a safe read-only Cypher query with LLM."""
        matched_node_list = [f"- {n['id']} ({', '.join(n.get('labels', []))})" for n in matched.get("matched_nodes", [])]
        matched_nodes_str = "\n".join(matched_node_list) if matched_node_list else "None found."

        relationships_sample = [
            f"({r.get('source')})-[:{r.get('relationship')}]->({r.get('target')})"
            for r in matched.get("connected_relationships", [])[:10]
        ]
        relationships_sample_str = "\n".join(relationships_sample) if relationships_sample else "None found."

        prompt = f"""You are a Neo4j Cypher query expert.
Write an executable Cypher READ query (MATCH ... RETURN) to fetch relevant knowledge graph facts for this user question.

User Question:
"{question}"

Extracted Entities & Keywords:
{extracted.get('entities', []) + extracted.get('search_keywords', [])}

Matched Candidates in Neo4j:
{matched_nodes_str}

Connected Graph Patterns:
{relationships_sample_str}

Neo4j Schema Guidelines:
1. Primary entity nodes have labels like `__Entity__`, `Microservice`, `Service`, `DatabaseTable`, `Technology`, `Feature`, `Component`.
2. The entity's primary identifier is stored in `n.id` or `n.name`.
3. Documents or chunks are labeled `Document` and connect via `(d:Document)-[:MENTIONS]->(e)`.
4. Relationships connect entities e.g.: `STORES`, `MANAGES`, `USES`, `PROVIDES`, `DEPENDS_ON`, `AUTHENTICATES`, `HANDLES`.

Query Rules:
- Return ONLY valid Cypher code. No markdown fences.
- Read-only MATCH statements only (no mutations).
- ALWAYS name any relationship variable you reference, e.g. `-[r]->` or `-[r:TYPE]->` so `type(r)` works.
- Entities in this Neo4j database use `n.id` as their primary identifier. Do not use `n.name`.
- Use case-insensitive matching if needed: `WHERE toLower(s.id) CONTAINS toLower(...)`.
- RETURN informative columns like: `s.id AS source_entity`, `type(r) AS relationship`, `target.id AS target_entity`.
- Limit results to 25.
"""
        response = self.llm.invoke(prompt)
        cypher = response.content.strip()

        # Strip markdown fences if present
        if cypher.startswith("```"):
            lines = cypher.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cypher = "\n".join(lines).strip()

        return cypher

    def execute_cypher_and_extract_facts(
        self,
        cypher_query: str,
        matched: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Step 4: Execute Cypher in Neo4j with automated multi-hop fallback."""
        graph = self.storage_manager.get_graph()
        records = []

        try:
            records = graph.query(cypher_query)
        except Exception as e:
            logger.warning(f"Generated Cypher failed: {e}")

        # Fallback to candidate neighborhood traversal if query was empty or failed
        if not records and matched.get("matched_nodes"):
            fallback_ids = [n["id"] for n in matched["matched_nodes"][:8] if n.get("id")]
            fallback_query = """
            UNWIND $ids AS nid
            MATCH (s {id: nid})
            OPTIONAL MATCH (s)-[r]->(target)
            WHERE target.id IS NOT NULL
            RETURN s.id AS source_entity, type(r) AS relationship, target.id AS target_entity
            LIMIT 25
            """
            try:
                records = graph.query(fallback_query, {"ids": fallback_ids})
            except Exception as e:
                logger.warning(f"Fallback query failed: {e}")

        return records

    def retrieve(self, question: str) -> Dict[str, Any]:
        """
        Complete Graph Retrieval Pipeline:
        Question -> Entity Extraction -> Candidate Matching -> Cypher Execution -> Structured Graph Context
        """
        start_time = time.time()

        if not self.is_available():
            return {
                "status": "error",
                "message": "Neo4j database is unreachable.",
                "entities": [],
                "matched_nodes": [],
                "cypher_query": "",
                "records": [],
                "formatted_context": "Neo4j Knowledge Graph is not available.",
                "elapsed_seconds": 0.0,
            }

        if self.verbose:
            print("\n" + "=" * 80)
            print(f"🕸️ [GRAPH RETRIEVAL] Analyzing Question: '{question}'")
            print("-" * 80)

        # 1. Entity Analysis
        analysis = self.extract_analysis(question)
        if self.verbose:
            print(f"  • Extracted Entities : {analysis.get('entities')}")
            print(f"  • Search Keywords    : {analysis.get('search_keywords')}")

        # 2. Candidate Matching
        matched = self.match_candidates(analysis)
        matched_nodes = matched.get("matched_nodes", [])
        if self.verbose:
            print(f"  • Matched Graph Nodes: {len(matched_nodes)} found: {[n.get('id') for n in matched_nodes[:5]]}")

        # 3. Cypher Generation
        cypher = self.generate_cypher(question, analysis, matched)
        if self.verbose:
            print(f"  • Generated Cypher   :\n    {cypher.replace(chr(10), ' ')}")

        # 4. Execution & Fact Retrieval
        records = self.execute_cypher_and_extract_facts(cypher, matched)

        # Format graph context lines
        triplets = []
        fact_lines = []
        for r in records:
            src = r.get("source_entity") or r.get("source") or r.get("s.id") or r.get("s")
            rel = r.get("relationship") or r.get("type(r)") or r.get("r")
            tgt = r.get("target_entity") or r.get("target") or r.get("target.id") or r.get("t")

            if src and rel and tgt:
                triplets.append({"source": str(src), "relationship": str(rel), "target": str(tgt)})
                fact_lines.append(f"Fact: ({src}) -[:{rel}]-> ({tgt})")
            elif src and tgt:
                triplets.append({"source": str(src), "relationship": "RELATED_TO", "target": str(tgt)})
                fact_lines.append(f"Fact: ({src}) is related to ({tgt})")
            elif src:
                fact_lines.append(f"Entity: {src}")

        formatted_context = "\n".join(fact_lines) if fact_lines else "No direct connected relationships found in graph."
        elapsed = time.time() - start_time

        if self.verbose:
            print(f"✅ Graph Retrieval Complete: {len(triplets)} graph triplets extracted in {elapsed:.3f}s.")
            print("=" * 80 + "\n")

        return {
            "status": "success",
            "question": question,
            "analysis": analysis,
            "matched_nodes": matched_nodes,
            "cypher_query": cypher,
            "triplets": triplets,
            "records_count": len(records),
            "formatted_context": formatted_context,
            "elapsed_seconds": round(elapsed, 3),
        }
