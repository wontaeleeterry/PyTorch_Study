import pickle
import sqlite3
from datetime import datetime

import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL,
    SQLITE_DB,
    FAISS_INDEX,
    MEMORY_METADATA,
)


class MemoryManager:

    def __init__(self):

        print("Loading embedding model...")

        self.encoder = SentenceTransformer(
            EMBEDDING_MODEL
        )

        self.dimension = (
            self.encoder
            .get_sentence_embedding_dimension()
        )

        self._initialize_database()

        self.memories = []

        self._load_faiss()


    # ========================================================
    # Database initialization
    # ========================================================

    def _initialize_database(self):

        conn = sqlite3.connect(
            SQLITE_DB
        )

        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )

        conn.commit()
        conn.close()


    # ========================================================
    # FAISS load
    # ========================================================

    def _load_faiss(self):

        if FAISS_INDEX.exists():

            print("Loading FAISS index...")

            self.index = faiss.read_index(
                str(FAISS_INDEX)
            )

        else:

            print("Creating new FAISS index...")

            # Inner Product + normalized vector
            # = cosine similarity

            self.index = faiss.IndexFlatIP(
                self.dimension
            )


        if MEMORY_METADATA.exists():

            print("Loading memory metadata...")

            with open(
                MEMORY_METADATA,
                "rb"
            ) as f:

                self.memories = pickle.load(f)


    # ========================================================
    # Save FAISS
    # ========================================================

    def _save_faiss(self):

        faiss.write_index(
            self.index,
            str(FAISS_INDEX)
        )

        with open(
            MEMORY_METADATA,
            "wb"
        ) as f:

            pickle.dump(
                self.memories,
                f
            )


    # ========================================================
    # Save conversation
    # ========================================================

    def save_message(
        self,
        role,
        content
    ):

        conn = sqlite3.connect(
            SQLITE_DB
        )

        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO conversations
            (role, content, timestamp)
            VALUES (?, ?, ?)
            """,
            (
                role,
                content,
                datetime.now().isoformat()
            )
        )

        conn.commit()
        conn.close()


    # ========================================================
    # Recent conversation
    # ========================================================

    def get_recent_messages(
        self,
        limit
    ):

        conn = sqlite3.connect(
            SQLITE_DB
        )

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT role, content
            FROM conversations
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,)
        )

        rows = cursor.fetchall()

        conn.close()

        rows.reverse()

        return [
            {
                "role": role,
                "content": content
            }
            for role, content in rows
        ]


    # ========================================================
    # Add long-term memory
    # ========================================================

    def add_memory(
        self,
        text,
        memory_type="summary"
    ):

        if not text.strip():
            return

        # ----------------------------------------------------
        # Embedding
        # ----------------------------------------------------

        vector = self.encoder.encode(
            [text],
            normalize_embeddings=True
        )

        vector = np.asarray(
            vector,
            dtype="float32"
        )

        # ----------------------------------------------------
        # FAISS
        # ----------------------------------------------------

        self.index.add(vector)

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        memory = {

            "text": text,

            "type": memory_type,

            "timestamp":
                datetime.now().isoformat(),

        }

        self.memories.append(
            memory
        )

        self._save_faiss()


    # ========================================================
    # Semantic search
    # ========================================================

    def search(
        self,
        query,
        top_k=5,
        score_threshold=0.25
    ):

        if self.index.ntotal == 0:

            return []

        # ----------------------------------------------------
        # Query embedding
        # ----------------------------------------------------

        vector = self.encoder.encode(
            [query],
            normalize_embeddings=True
        )

        vector = np.asarray(
            vector,
            dtype="float32"
        )

        # ----------------------------------------------------
        # FAISS search
        # ----------------------------------------------------

        scores, indices = self.index.search(
            vector,
            min(
                top_k,
                self.index.ntotal
            )
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index < 0:
                continue

            if score < score_threshold:
                continue

            memory = self.memories[index].copy()

            memory["score"] = float(score)

            results.append(
                memory
            )

        return results


    # ========================================================
    # Delete all memories
    # ========================================================

    def clear_memory(self):

        self.index = faiss.IndexFlatIP(
            self.dimension
        )

        self.memories = []

        self._save_faiss()


    # ========================================================
    # Memory statistics
    # ========================================================

    def statistics(self):

        type_count = {}

        for memory in self.memories:

            memory_type = memory.get(
                "type",
                "unknown"
            )

            type_count[memory_type] = (
                type_count.get(
                    memory_type,
                    0
                ) + 1
            )

        return {

            "total": len(self.memories),

            "faiss_vectors":
                self.index.ntotal,

            "types":
                type_count,
        }