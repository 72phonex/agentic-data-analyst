"""
Persistent Memory Store for Tabular Data Analysis Agent.
Stores only curated, useful semantic facts:
  - Column meanings
  - Metric definitions
  - User corrections
  - Preferences
  - Named reusable filters

Rules:
  - Extract facts from user statements
  - Ask user to confirm before saving (proposal mechanism)
  - Deduplicate identical memories
  - Update/replace on conflict with full timestamped audit history
  - Retrieve only relevant memories per question and inject them
  - Support commands: /memory, /forget, /edit, /add
"""

from __future__ import annotations
import os
import sqlite3
import re
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_DB_PATH = os.getenv("MEMORY_DB_PATH", "data_agent_memory.sqlite")


class MemoryStore:
    def __init__(self, db_path: Optional[str] = None):
        db_path = db_path or os.getenv("MEMORY_DB_PATH", DEFAULT_DB_PATH)
        self.db_path = db_path
        self._shared_conn = None
        if db_path == ":memory:":
            self._shared_conn = sqlite3.connect(":memory:")
            self._shared_conn.row_factory = sqlite3.Row
        self._init_db()
        self.pending_proposals: Dict[str, Dict[str, Any]] = {}

    def _get_connection(self) -> sqlite3.Connection:
        if self._shared_conn is not None:
            return self._shared_conn
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL, -- 'column_meaning', 'metric_definition', 'user_correction', 'preference', 'filter'
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                description TEXT,
                dataset_name TEXT DEFAULT 'global',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id INTEGER NOT NULL,
                key TEXT NOT NULL,
                old_value TEXT NOT NULL,
                new_value TEXT NOT NULL,
                changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
            );
            """)
            conn.commit()

    # --------------------------------------------------------------------------
    # FACT PROPOSAL & CONFIRMATION
    # --------------------------------------------------------------------------

    def propose_memory(
        self,
        category: str,
        key: str,
        value: str,
        description: Optional[str] = None,
        dataset_name: str = "global"
    ) -> Dict[str, Any]:
        """
        Create a pending memory proposal for user confirmation before saving.
        """
        clean_key = key.strip()
        clean_val = value.strip()
        proposal_id = f"prop_{len(self.pending_proposals) + 1}"

        # Check if existing memory has this key
        existing = self.get_by_key(clean_key, dataset_name=dataset_name)
        action = "update" if existing else "add"

        proposal = {
            "proposal_id": proposal_id,
            "category": category,
            "key": clean_key,
            "value": clean_val,
            "description": description or "",
            "dataset_name": dataset_name,
            "action": action,
            "existing_id": existing["id"] if existing else None,
            "existing_value": existing["value"] if existing else None
        }
        self.pending_proposals[proposal_id] = proposal
        return proposal

    def confirm_proposal(self, proposal_id: str) -> Dict[str, Any]:
        """
        Commit a previously proposed memory upon user confirmation.
        """
        if proposal_id not in self.pending_proposals:
            raise KeyError(f"Proposal '{proposal_id}' not found or already processed.")

        prop = self.pending_proposals.pop(proposal_id)
        return self.save_memory(
            category=prop["category"],
            key=prop["key"],
            value=prop["value"],
            description=prop["description"],
            dataset_name=prop["dataset_name"]
        )

    def cancel_proposal(self, proposal_id: str) -> None:
        """Cancel a pending memory proposal."""
        self.pending_proposals.pop(proposal_id, None)

    # --------------------------------------------------------------------------
    # SAVE, UPDATE, DEDUPLICATE
    # --------------------------------------------------------------------------

    def save_memory(
        self,
        category: str,
        key: str,
        value: str,
        description: Optional[str] = None,
        dataset_name: str = "global"
    ) -> Dict[str, Any]:
        """
        Save a confirmed memory: deduplicates exact matches, or updates
        existing keys with an audit record in memory_history.
        """
        clean_key = key.strip()
        clean_val = value.strip()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            # 1. Check for exact duplicate (same key and value)
            cursor.execute(
                "SELECT id, category, key, value, updated_at FROM memories WHERE LOWER(key) = LOWER(?) AND dataset_name = ?",
                (clean_key, dataset_name)
            )
            existing = cursor.fetchone()

            if existing:
                mem_id = existing["id"]
                old_val = existing["value"]

                if old_val == clean_val:
                    # Deduplicated: no change needed
                    return {
                        "status": "deduplicated",
                        "memory_id": mem_id,
                        "key": clean_key,
                        "value": clean_val,
                        "message": f"Identical memory already exists for '{clean_key}'."
                    }

                # Update existing memory and record audit history
                now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                cursor.execute(
                    "UPDATE memories SET value = ?, category = ?, description = ?, updated_at = ? WHERE id = ?",
                    (clean_val, category, description or "", now, mem_id)
                )
                cursor.execute(
                    "INSERT INTO memory_history (memory_id, key, old_value, new_value, changed_at) VALUES (?, ?, ?, ?, ?)",
                    (mem_id, clean_key, old_val, clean_val, now)
                )
                conn.commit()
                return {
                    "status": "updated",
                    "memory_id": mem_id,
                    "key": clean_key,
                    "old_value": old_val,
                    "new_value": clean_val,
                    "message": f"Updated memory #{mem_id} for '{clean_key}' (previous value saved to history)."
                }
            else:
                # Insert brand new memory
                now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                cursor.execute(
                    """INSERT INTO memories (category, key, value, description, dataset_name, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (category, clean_key, clean_val, description or "", dataset_name, now, now)
                )
                mem_id = cursor.lastrowid
                conn.commit()
                return {
                    "status": "created",
                    "memory_id": mem_id,
                    "key": clean_key,
                    "value": clean_val,
                    "message": f"Saved new memory #{mem_id} under '{category}'."
                }

    # --------------------------------------------------------------------------
    # RETRIEVAL (Semantic / Keyword Matching)
    # --------------------------------------------------------------------------

    def retrieve_relevant(
        self,
        query: str,
        dataset_name: str = "global",
        top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Retrieve only the memories relevant to the user's question using keyword/token overlap.
        """
        all_memories = self.list_memories(dataset_name=dataset_name)
        if not all_memories:
            return []

        # Tokenize query
        q_tokens = set(re.findall(r"\w+", query.lower()))
        # Filter out common stop words
        stops = {"what", "is", "the", "of", "and", "in", "to", "for", "a", "an", "how", "many", "show", "me", "are", "there"}
        content_tokens = q_tokens - stops
        if not content_tokens:
            content_tokens = q_tokens

        scored = []
        for mem in all_memories:
            mem_text = f"{mem['category']} {mem['key']} {mem['value']} {mem.get('description', '')}".lower()
            mem_tokens = set(re.findall(r"\w+", mem_text))

            # Exact key match in query bonus
            key_clean = mem["key"].lower()
            key_bonus = 3.0 if key_clean in query.lower() else 0.0

            # Token overlap
            overlap = len(content_tokens.intersection(mem_tokens))
            score = overlap + key_bonus

            if score > 0:
                scored.append((score, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:top_k]]

    def format_memories_prompt(self, relevant_memories: List[Dict[str, Any]]) -> str:
        """Format retrieved memories cleanly for LLM system prompt injection."""
        if not relevant_memories:
            return ""

        lines = ["=== RELEVANT PERSISTENT USER MEMORIES & DOMAIN DEFINITIONS ==="]
        for m in relevant_memories:
            lines.append(f"- [{m['category']}] \"{m['key']}\": {m['value']}")
        lines.append("CRITICAL: You MUST apply these user-defined definitions and metric rules when constructing queries or computing answers.")
        lines.append("==============================================================")
        return "\n".join(lines)

    # --------------------------------------------------------------------------
    # CRUD & COMMANDS (/memory, /forget, /edit)
    # --------------------------------------------------------------------------

    def list_memories(self, dataset_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all stored memories."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if dataset_name and dataset_name != "global":
                cursor.execute(
                    "SELECT * FROM memories WHERE dataset_name IN (?, 'global') ORDER BY id ASC",
                    (dataset_name,)
                )
            else:
                cursor.execute("SELECT * FROM memories ORDER BY id ASC")
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def get_by_id(self, memory_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a memory by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_by_key(self, key: str, dataset_name: str = "global") -> Optional[Dict[str, Any]]:
        """Retrieve a memory by key name."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM memories WHERE LOWER(key) = LOWER(?) AND dataset_name = ?",
                (key.strip(), dataset_name)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def delete_memory(self, memory_id: int) -> bool:
        """Delete a memory by ID (/forget <id>)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            conn.commit()
            return cursor.rowcount > 0

    def edit_memory(self, memory_id: int, new_value: str) -> Optional[Dict[str, Any]]:
        """Edit an existing memory (/edit <id> <new_value>)."""
        existing = self.get_by_id(memory_id)
        if not existing:
            return None

        clean_val = new_value.strip()
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE memories SET value = ?, updated_at = ? WHERE id = ?",
                (clean_val, now, memory_id)
            )
            cursor.execute(
                "INSERT INTO memory_history (memory_id, key, old_value, new_value, changed_at) VALUES (?, ?, ?, ?, ?)",
                (memory_id, existing["key"], existing["value"], clean_val, now)
            )
            conn.commit()

        return {
            "memory_id": memory_id,
            "key": existing["key"],
            "old_value": existing["value"],
            "new_value": clean_val,
            "updated_at": now
        }

    def get_history(self, memory_id: int) -> List[Dict[str, Any]]:
        """Get revision history for a memory."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM memory_history WHERE memory_id = ? ORDER BY id DESC",
                (memory_id,)
            )
            return [dict(r) for r in cursor.fetchall()]

    # --------------------------------------------------------------------------
    # HEURISTIC FACT EXTRACTOR (Detects definitions, filters, corrections)
    # --------------------------------------------------------------------------

    def detect_fact_candidate(self, text: str) -> Optional[Dict[str, str]]:
        """
        Check if user input states a definition, rule, correction, or preference.
        Examples:
          - "Active customers means status is 'Active' and cancelled_flag is 0"
          - "Note that tenure refers to number of months"
          - "Consider high score as total_score >= 80"
          - "Always use weekly aggregation"
        """
        t = text.strip()

        # Pattern: "X means Y" / "X is defined as Y" / "X refers to Y"
        pat_means = re.search(
            r"(?:note\s+that\s+)?['\"]?([\w\s_-]+?)['\"]?\s+(?:means|is defined as|refers to|stands for|represents)\s+(.+)",
            t, re.IGNORECASE
        )
        if pat_means:
            key, val = pat_means.group(1).strip(), pat_means.group(2).strip()
            category = "metric_definition" if any(w in val.lower() for w in ["=", ">", "<", "where", "status", "flag", "score"]) else "column_meaning"
            return {"category": category, "key": key, "value": val}

        # Pattern: "filter X: Y" or "let X be Y"
        pat_let = re.search(r"(?:let|consider)\s+['\"]?([\w\s_-]+?)['\"]?\s+(?:as|be)\s+(.+)", t, re.IGNORECASE)
        if pat_let:
            key, val = pat_let.group(1).strip(), pat_let.group(2).strip()
            return {"category": "filter", "key": key, "value": val}

        # Pattern: "preference: X" or "always use X"
        pat_pref = re.search(r"(?:always\s+use|prefer)\s+(weekly|daily|monthly|pearson|spearman)\s+([\w\s]+)", t, re.IGNORECASE)
        if pat_pref:
            val, key = pat_pref.group(1).strip(), pat_pref.group(2).strip()
            return {"category": "preference", "key": key, "value": val}

        return None
