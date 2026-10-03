"""
Cold storage: every chunk lives in a MySQL table and is searched with BM25.
"""

import logging
import re
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import mysql.connector
import spacy
from rank_bm25 import BM25Okapi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    id: int
    text: str
    score: float


class ColdStorage:

    def __init__(self, host: str, user: str, password: str, database: str):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None
        self.cursor = None
        self.bm25_index = None
        self.documents = []              # (id, text) rows, in BM25 index order
        self.needs_index_rebuild = True
        self.nlp = spacy.load("en_core_web_sm")

        self._connect()
        self._create_table()

    def _connect(self):
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
            )
            self.cursor = self.connection.cursor()
            logger.info("Connected to MySQL")
        except mysql.connector.Error as err:
            logger.error(f"Could not connect to MySQL: {err}")
            raise

    def _create_table(self):
        try:
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                id INT AUTO_INCREMENT PRIMARY KEY,
                text TEXT NOT NULL,
                access_count INT DEFAULT 0,
                last_accessed DOUBLE
                );
                """)
            self.connection.commit()
        except mysql.connector.Error as err:
            logger.error(f"Could not create the documents table: {err}")
            self.connection.rollback()
            raise

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def _tokenize_query(self, query: str) -> List[str]:
        # Drop stop words and punctuation. Keep each word as typed and also
        # its base form, so "networks" matches chunks that say "network" too.
        tokens = []
        for token in self.nlp(query.lower()):
            if token.is_stop or token.is_punct or token.is_space:
                continue
            tokens.append(token.text)
            if token.lemma_ != token.text:
                tokens.append(token.lemma_)
        return tokens

    def _rebuild_index(self):
        logger.info("Rebuilding BM25 index")
        try:
            self.cursor.execute("SELECT id, text FROM documents")
            rows = self.cursor.fetchall()
        except mysql.connector.Error as err:
            logger.error(f"Could not rebuild the index: {err}")
            raise

        self.documents = rows
        self.bm25_index = BM25Okapi([self._tokenize(text) for _, text in rows]) if rows else None
        self.needs_index_rebuild = False
        logger.info(f"BM25 index has {len(rows)} documents")

    def add_document(self, text: str) -> int:
        try:
            self.cursor.execute(
                "INSERT INTO documents (text, access_count, last_accessed) VALUES (%s, %s, %s)",
                (text, 0, time.time())
            )
            doc_id = self.cursor.lastrowid
            self.connection.commit()
            self.needs_index_rebuild = True
            logger.info(f"Added document {doc_id}")
            return doc_id
        except mysql.connector.Error as err:
            logger.error(f"Could not add document: {err}")
            self.connection.rollback()
            raise

    def get_document(self, id: int) -> Optional[str]:
        try:
            self.cursor.execute("SELECT text FROM documents WHERE id = %s", (id,))
            row = self.cursor.fetchone()
            return row[0] if row else None
        except mysql.connector.Error as err:
            logger.error(f"Could not get document {id}: {err}")
            return None

    def get_all_documents(self) -> List[Tuple[int, str]]:
        try:
            self.cursor.execute("SELECT id, text FROM documents")
            return self.cursor.fetchall()
        except mysql.connector.Error as err:
            logger.error(f"Could not get documents: {err}")
            return []

    def update_access_stats(self, id: int):
        try:
            self.cursor.execute(
                """UPDATE documents
                   SET access_count = access_count + 1,
                       last_accessed = %s
                   WHERE id = %s""",
                (time.time(), id)
            )
            self.connection.commit()
        except mysql.connector.Error as err:
            logger.error(f"Could not update access stats for {id}: {err}")
            self.connection.rollback()

    def delete_document(self, id: int):
        try:
            self.cursor.execute("DELETE FROM documents WHERE id = %s", (id,))
            self.connection.commit()
            self.needs_index_rebuild = True
            logger.info(f"Deleted document {id}")
        except mysql.connector.Error as err:
            logger.error(f"Could not delete document {id}: {err}")
            self.connection.rollback()

    def search_bm25(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Returns up to top_k chunks with a BM25 score above zero, best first."""
        if self.needs_index_rebuild:
            self._rebuild_index()

        if not self.documents or self.bm25_index is None:
            logger.warning("No documents to search")
            return []

        tokens = self._tokenize_query(query)
        logger.debug(f"Query tokens: {tokens}")
        if not tokens:
            logger.warning("Nothing left of the query after removing stop words")
            return []

        scores = self.bm25_index.get_scores(tokens)
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for i in top:
            doc_id, text = self.documents[i]
            self.update_access_stats(doc_id)
            if scores[i] > 0:
                results.append(SearchResult(id=doc_id, text=text, score=float(scores[i])))

        logger.info(f"BM25 returned {len(results)} results for '{query[:50]}'")
        return results

    def delete_table(self):
        try:
            self.cursor.execute("DROP TABLE IF EXISTS documents")
            self.connection.commit()
            logger.info("Dropped the documents table")
        except mysql.connector.Error as err:
            logger.error(f"Could not drop the documents table: {err}")
            self.connection.rollback()

    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
            logger.info("MySQL connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
