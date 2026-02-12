"""Cognitive Intelligence Layer — connects NotebookLM to the UCW cognitive database.

Innovations:
1. GraphRAG query enrichment (knowledge graph + spreading activation)
2. Bidirectional cognitive flow (every result feeds back as cognitive events)
3. Coherence-driven auto-curation (notebooks create themselves)
6. Research session bridge (ResearchGravity sessions → NotebookLM notebooks)
7. Temporal knowledge evolution tracking
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("notebooklm_mcp.cognitive")


class CognitiveLayer:
    """Intelligence layer between MCP tools and NotebookLM API.

    Connects the NotebookLM API client to the UCW cognitive database,
    enabling GraphRAG enrichment, bidirectional capture, auto-curation,
    research session bridging, and knowledge evolution tracking.

    All methods are async since they interact with PostgreSQL via asyncpg.
    Gracefully degrades when cognitive DB is unavailable.
    """

    def __init__(self, api_client=None, pg_pool=None, embedding_pipeline=None):
        self._api = api_client
        self._pool = pg_pool
        self._embeddings = embedding_pipeline
        self._graph = None
        self._search = None
        self._fsrs = None
        self._available = pg_pool is not None

        if self._available:
            try:
                from coherence_engine.knowledge_graph import GraphManager
                self._graph = GraphManager(pg_pool)
            except ImportError:
                pass
            try:
                from coherence_engine.hybrid_search import HybridSearch
                self._search = HybridSearch(pg_pool)
            except ImportError:
                pass
            try:
                from coherence_engine.fsrs import InsightScheduler
                self._fsrs = InsightScheduler(pg_pool)
            except ImportError:
                pass

    @property
    def available(self) -> bool:
        return self._available

    # =================================================================
    # Innovation 1: GraphRAG Query Enrichment
    # =================================================================

    async def enriched_query(
        self,
        notebook_id: str,
        query: str,
        source_ids: list[str] | None = None,
        max_context_items: int = 5,
    ) -> dict[str, Any]:
        """Query a notebook with cognitive context injection.

        Enriches the query with:
        - Related entities from the knowledge graph via spreading activation
        - Cross-platform coherence moments on the same topic
        - FSRS-due insights for reinforcement

        The enriched context is injected as part of the query prompt.
        """
        enrichments = []

        # 1. Knowledge graph enrichment
        if self._graph and self._pool:
            try:
                # Find related entities via spreading activation
                entities = await self._find_related_entities(query, limit=max_context_items)
                if entities:
                    enrichments.append(
                        "Related knowledge from your cognitive history:\n"
                        + "\n".join(f"- {e['name']}: {e.get('context', '')}" for e in entities)
                    )
            except Exception as e:
                logger.debug(f"Graph enrichment failed: {e}")

        # 2. Coherence moments
        if self._search:
            try:
                results = await self._search.search(query, limit=3)
                if results:
                    moments = []
                    for r in results[:3]:
                        content = r.get("content", "")[:200]
                        platform = r.get("platform", "unknown")
                        moments.append(f"[{platform}] {content}")
                    if moments:
                        enrichments.append(
                            "Cross-platform coherence (related insights from your other AI sessions):\n"
                            + "\n".join(f"- {m}" for m in moments)
                        )
            except Exception as e:
                logger.debug(f"Coherence search failed: {e}")

        # 3. FSRS-due insights
        if self._fsrs:
            try:
                due = await self._fsrs.get_due_insights(limit=3)
                if due:
                    enrichments.append(
                        "Insights due for review (spaced repetition):\n"
                        + "\n".join(f"- {d.get('content', '')[:150]}" for d in due)
                    )
            except Exception as e:
                logger.debug(f"FSRS enrichment failed: {e}")

        # Build enriched query
        if enrichments:
            context_block = "\n\n".join(enrichments)
            enriched_prompt = (
                f"[Cognitive Context — from your cross-platform knowledge base]\n"
                f"{context_block}\n\n"
                f"[Your Question]\n{query}"
            )
        else:
            enriched_prompt = query

        # Execute query
        result = self._api.query(notebook_id, enriched_prompt, source_ids=source_ids)

        # Capture result as cognitive event (Innovation 2)
        if result and result.get("answer"):
            await self.capture_result("enriched_query", {
                "query": query,
                "answer": result["answer"],
                "notebook_id": notebook_id,
                "enrichments_used": len(enrichments),
            })

        return {
            **(result or {}),
            "enrichments_used": len(enrichments),
            "enrichment_details": enrichments,
        }

    # =================================================================
    # Innovation 2: Bidirectional Cognitive Flow
    # =================================================================

    async def capture_result(
        self,
        tool_name: str,
        result_data: dict[str, Any],
        notebook_id: str | None = None,
    ) -> bool:
        """Capture a NotebookLM tool result as a cognitive event.

        Every NotebookLM interaction feeds back into the cognitive database
        with UCW Data/Light/Instinct layers.
        """
        if not self._pool:
            return False

        try:
            content = json.dumps(result_data, default=str)[:5000]
            now = datetime.now(timezone.utc)

            # Extract UCW layers
            data_layer = {
                "tool": tool_name,
                "notebook_id": notebook_id,
                "timestamp": now.isoformat(),
                "content_length": len(content),
            }
            light_layer = self._extract_light_layer(tool_name, result_data)
            instinct_layer = {
                "coherence_potential": 0.5,  # Base — will be refined by coherence engine
                "emergence_indicator": tool_name in ("enriched_query", "cognitive_auto_curate"),
            }

            async with self._pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO cognitive_events
                       (platform, event_type, content, metadata, created_at)
                       VALUES ($1, $2, $3, $4, $5)""",
                    "notebooklm",
                    f"tool_{tool_name}",
                    content,
                    json.dumps({
                        "ucw_data": data_layer,
                        "ucw_light": light_layer,
                        "ucw_instinct": instinct_layer,
                    }),
                    now,
                )

            # Embed the event if pipeline available
            if self._embeddings:
                try:
                    await self._embeddings.embed_event({
                        "content": content,
                        "platform": "notebooklm",
                        "event_type": f"tool_{tool_name}",
                    })
                except Exception:
                    pass

            return True
        except Exception as e:
            logger.debug(f"Capture failed: {e}")
            return False

    # =================================================================
    # Innovation 3: Coherence-Driven Auto-Curation
    # =================================================================

    async def check_curation_triggers(self) -> list[dict]:
        """Check if coherence patterns warrant automatic notebook creation.

        Looks for convergence patterns (multiple platforms, same topic,
        high significance) and returns suggested notebook creations.
        """
        if not self._pool:
            return []

        suggestions = []
        try:
            async with self._pool.acquire() as conn:
                # Find recent high-significance coherence arcs
                rows = await conn.fetch(
                    """SELECT arc_id, topic, significance_score, platform_count,
                              moment_count, created_at
                       FROM coherence_arcs
                       WHERE significance_score > 0.7
                         AND created_at > NOW() - INTERVAL '7 days'
                         AND NOT EXISTS (
                             SELECT 1 FROM cognitive_events
                             WHERE event_type = 'auto_curate_notebook'
                               AND metadata::jsonb->>'arc_id' = coherence_arcs.arc_id::text
                         )
                       ORDER BY significance_score DESC
                       LIMIT 3"""
                )
                for row in rows:
                    suggestions.append({
                        "arc_id": str(row["arc_id"]),
                        "topic": row["topic"],
                        "significance": float(row["significance_score"]),
                        "platforms": row["platform_count"],
                        "moments": row["moment_count"],
                        "suggested_title": f"Convergence: {row['topic']}",
                    })
        except Exception as e:
            logger.debug(f"Curation check failed: {e}")

        return suggestions

    async def auto_curate_notebook(self, arc_id: str) -> dict[str, Any] | None:
        """Create a NotebookLM notebook from a coherence arc.

        Gathers coherent events from the arc, creates a notebook,
        and imports them as pasted-text sources.
        """
        if not self._pool:
            return None

        try:
            async with self._pool.acquire() as conn:
                # Get arc details
                arc = await conn.fetchrow(
                    "SELECT topic, significance_score FROM coherence_arcs WHERE arc_id = $1", arc_id
                )
                if not arc:
                    return None

                # Get moments in this arc
                moments = await conn.fetch(
                    """SELECT ce.content, ce.platform, ce.created_at
                       FROM coherence_moments cm
                       JOIN cognitive_events ce ON ce.event_id = cm.event_id_a OR ce.event_id = cm.event_id_b
                       WHERE cm.arc_id = $1
                       ORDER BY ce.created_at
                       LIMIT 20""",
                    arc_id,
                )

            if not moments:
                return None

            # Create notebook
            topic = arc["topic"]
            nb = self._api.create_notebook(f"Convergence: {topic}")
            if not nb:
                return None

            # Import moments as pasted text sources
            text_parts = []
            for m in moments:
                ts = m["created_at"].strftime("%Y-%m-%d %H:%M") if m["created_at"] else "unknown"
                text_parts.append(f"[{m['platform']} @ {ts}]\n{m['content'][:500]}")

            combined = "\n\n---\n\n".join(text_parts)
            self._api.add_text_source(nb.id, combined, title=f"Coherence Moments: {topic}")

            # Record the auto-curation event
            await self.capture_result("cognitive_auto_curate", {
                "arc_id": arc_id,
                "topic": topic,
                "notebook_id": nb.id,
                "moment_count": len(moments),
            })

            return {
                "notebook_id": nb.id,
                "title": nb.title,
                "topic": topic,
                "sources_added": 1,
                "moments_included": len(moments),
            }
        except Exception as e:
            logger.error(f"Auto-curation failed: {e}")
            return None

    # =================================================================
    # Innovation 6: Research Session Bridge
    # =================================================================

    async def import_research_session(
        self,
        session_id: str,
        notebook_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Import a ResearchGravity session into a NotebookLM notebook.

        Bridges session URLs, findings, and thesis into a notebook:
        - Session URLs (Tier 1/2/3) → URL sources
        - Session findings → pasted text sources
        - Session thesis + gap → notebook description via configure_chat
        """
        if not self._pool:
            return None

        try:
            async with self._pool.acquire() as conn:
                session = await conn.fetchrow(
                    "SELECT topic, thesis, gap, status FROM research_sessions WHERE session_id = $1",
                    session_id,
                )
                if not session:
                    return None

                urls = await conn.fetch(
                    "SELECT url, tier, category, relevance FROM session_urls WHERE session_id = $1 ORDER BY relevance DESC",
                    session_id,
                )
                findings = await conn.fetch(
                    "SELECT content, category FROM session_findings WHERE session_id = $1",
                    session_id,
                )

            topic = session["topic"] or "Research Session"

            # Create or use existing notebook
            if notebook_id is None:
                nb = self._api.create_notebook(topic)
                if not nb:
                    return None
                notebook_id = nb.id

            sources_added = 0

            # Add URLs as sources (top 10 by relevance)
            for u in urls[:10]:
                try:
                    self._api.add_url_source(notebook_id, u["url"])
                    sources_added += 1
                except Exception:
                    pass

            # Add findings as pasted text
            if findings:
                finding_text = "\n\n".join(
                    f"**{f['category'] or 'Finding'}:** {f['content']}" for f in findings
                )
                self._api.add_text_source(notebook_id, finding_text, title=f"Findings: {topic}")
                sources_added += 1

            # Configure chat with thesis context
            thesis = session.get("thesis")
            if thesis:
                try:
                    self._api.configure_chat(
                        notebook_id, goal="custom",
                        custom_prompt=f"Research context: {thesis}\nGap identified: {session.get('gap', 'N/A')}",
                    )
                except Exception:
                    pass

            await self.capture_result("research_to_notebook", {
                "session_id": session_id,
                "notebook_id": notebook_id,
                "sources_added": sources_added,
            })

            return {
                "notebook_id": notebook_id,
                "session_id": session_id,
                "topic": topic,
                "sources_added": sources_added,
                "url_count": min(len(urls), 10),
                "finding_count": len(findings),
            }
        except Exception as e:
            logger.error(f"Research import failed: {e}")
            return None

    # =================================================================
    # Innovation 7: Knowledge Evolution Tracking
    # =================================================================

    async def track_knowledge_evolution(
        self,
        notebook_id: str,
        query: str,
        response: str,
    ) -> dict[str, Any]:
        """Track how understanding evolves over repeated queries.

        Detects:
        - Knowledge crystallization (answers stabilize → mastery)
        - Knowledge fragmentation (answers diverge → new exploration needed)
        """
        if not self._pool:
            return {"tracked": False}

        try:
            async with self._pool.acquire() as conn:
                # Store this query-response pair
                await conn.execute(
                    """INSERT INTO knowledge_evolution
                       (notebook_id, query, response, queried_at)
                       VALUES ($1, $2, $3, $4)""",
                    notebook_id, query, response[:2000],
                    datetime.now(timezone.utc),
                )

                # Check for previous responses to similar queries
                prev = await conn.fetch(
                    """SELECT response, queried_at FROM knowledge_evolution
                       WHERE notebook_id = $1 AND query = $2
                       ORDER BY queried_at DESC LIMIT 5""",
                    notebook_id, query,
                )

            if len(prev) < 2:
                return {"tracked": True, "evolution_state": "initial", "data_points": len(prev)}

            # Simple similarity check: response length stability
            lengths = [len(r["response"]) for r in prev]
            avg_len = sum(lengths) / len(lengths)
            variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)
            normalized_variance = variance / (avg_len ** 2) if avg_len > 0 else 0

            if normalized_variance < 0.05:
                state = "crystallized"  # Answers are stabilizing → mastery
            elif normalized_variance > 0.3:
                state = "fragmenting"  # Answers are diverging → needs exploration
            else:
                state = "evolving"  # Normal evolution

            return {
                "tracked": True,
                "evolution_state": state,
                "data_points": len(prev),
                "normalized_variance": round(normalized_variance, 4),
            }
        except Exception as e:
            logger.debug(f"Evolution tracking failed: {e}")
            return {"tracked": False, "error": str(e)}

    # =================================================================
    # Cognitive Search (for MCP tool)
    # =================================================================

    async def cognitive_search(self, query: str, limit: int = 10) -> list[dict]:
        """Hybrid search across the cognitive database for notebook context."""
        if not self._search:
            return []
        try:
            return await self._search.search(query, limit=limit)
        except Exception as e:
            logger.debug(f"Cognitive search failed: {e}")
            return []

    async def get_due_insights(self, limit: int = 10) -> list[dict]:
        """Surface FSRS-due insights for reinforcement."""
        if not self._fsrs:
            return []
        try:
            return await self._fsrs.get_due_insights(limit=limit)
        except Exception as e:
            logger.debug(f"FSRS query failed: {e}")
            return []

    # =================================================================
    # Private helpers
    # =================================================================

    async def _find_related_entities(self, query: str, limit: int = 5) -> list[dict]:
        """Find entities related to the query via knowledge graph."""
        if not self._graph or not self._pool:
            return []

        # First, find entities matching the query
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT entity_id, name, entity_type, mention_count
                   FROM knowledge_entities
                   WHERE name ILIKE $1 OR name ILIKE $2
                   ORDER BY mention_count DESC LIMIT 3""",
                f"%{query}%", f"%{query.split()[0]}%" if query.split() else f"%{query}%",
            )

        if not rows:
            return []

        # Spread activation from top entity
        entities = []
        try:
            activated = await self._graph.spreading_activation(
                rows[0]["entity_id"], depth=2, decay=0.6
            )
            for node in activated[:limit]:
                entities.append({
                    "name": node.get("name", ""),
                    "type": node.get("entity_type", ""),
                    "activation": node.get("activation", 0),
                    "context": node.get("context", ""),
                })
        except Exception:
            # Fallback to direct matches
            for row in rows[:limit]:
                entities.append({
                    "name": row["name"],
                    "type": row["entity_type"],
                    "activation": 1.0,
                    "context": "",
                })

        return entities

    def _extract_light_layer(self, tool_name: str, data: dict) -> dict:
        """Extract UCW Light layer (meaning/intent) from tool result."""
        layer = {"tool": tool_name}

        if "answer" in data:
            layer["intent"] = "synthesis"
            layer["topic"] = data.get("query", "")[:100]
        elif "notebook_id" in data:
            layer["intent"] = "organization"
        elif "sources_added" in data:
            layer["intent"] = "knowledge_import"
        else:
            layer["intent"] = "action"

        return layer
