#!/usr/bin/env python3
"""
BM25 Retriever Wrapper
Provides a unified interface for BM25 retrieval using Pyserini
"""

import logging
from typing import List, Dict, Optional
from pyserini.search.lucene import LuceneSearcher

logger = logging.getLogger(__name__)


class BM25Retriever:
    """BM25 retriever using Pyserini/Lucene"""
    
    def __init__(
        self,
        index_path: str,
        k1: float = 0.9,
        b: float = 0.4,
        threads: int = 16,
    ):
        """
        Initialize BM25 retriever
        
        Args:
            index_path: Path to BM25 index directory
            k1: BM25 k1 parameter (default: 0.9)
            b: BM25 b parameter (default: 0.4)
            threads: Number of threads to use for batch search (default: 16)
        """
        self.index_path = index_path
        self.k1 = k1
        self.b = b
        self.threads = int(threads)
        
        logger.info(f"Loading BM25 index from {index_path}...")
        self.searcher = LuceneSearcher(index_path)
        
        logger.info(f"Setting BM25 parameters: k1={k1}, b={b}")
        self.searcher.set_bm25(k1, b)
        
        logger.info(f"BM25 retriever initialized")
    
    def search(
        self,
        queries: List[str],
        top_k: int = 50,
        return_scores: bool = True,
    ) -> List[List[Dict]]:
        """
        Search for top-k documents for each query
        
        Args:
            queries: List of query strings
            top_k: Number of results to return per query
            return_scores: Whether to return scores
            
        Returns:
            List of result lists, one per query
            Each result contains: {"doc_id": str, "score": float, "rank": int}
        """
        if not queries:
            return []

        # Use Lucene batch_search for speed (multi-threaded) when possible.
        # Falls back to per-query search if batch_search is unavailable.
        query_ids = [str(i) for i in range(len(queries))]
        hits_by_qid: Optional[Dict[str, list]] = None

        try:
            hits_by_qid = self.searcher.batch_search(
                queries,
                query_ids,
                k=top_k,
                threads=self.threads,
            )
        except Exception as e:
            logger.warning(f"batch_search failed, falling back to sequential search: {e}")

        results: List[List[Dict]] = []
        if hits_by_qid is not None:
            for i in range(len(queries)):
                hits = hits_by_qid.get(str(i), [])
                query_results: List[Dict] = []
                for rank, hit in enumerate(hits, 1):
                    result = {
                        "doc_id": hit.docid,
                        "rank": rank,
                    }
                    if return_scores:
                        result["score"] = float(hit.score)
                    query_results.append(result)
                results.append(query_results)
            return results

        # Sequential fallback
        for query in queries:
            hits = self.searcher.search(query, k=top_k)
            query_results: List[Dict] = []
            for rank, hit in enumerate(hits, 1):
                result = {
                    "doc_id": hit.docid,
                    "rank": rank,
                }
                if return_scores:
                    result["score"] = float(hit.score)
                query_results.append(result)
            results.append(query_results)
        return results
    
    def search_single(
        self,
        query: str,
        top_k: int = 50,
        return_scores: bool = True,
    ) -> List[Dict]:
        """
        Search for a single query
        
        Args:
            query: Query string
            top_k: Number of results to return
            return_scores: Whether to return scores
            
        Returns:
            List of results: [{"doc_id": str, "score": float, "rank": int}, ...]
        """
        results = self.search([query], top_k, return_scores)
        return results[0] if results else []
    
    def get_stats(self) -> Dict:
        """Get retriever statistics"""
        return {
            "retriever_type": "bm25",
            "index_path": self.index_path,
            "k1": self.k1,
            "b": self.b,
            "threads": self.threads,
            "total_documents": self.searcher.num_docs,
        }
