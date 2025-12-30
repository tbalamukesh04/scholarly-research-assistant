from evaluation.hybrid.normalize import normalize_result


class HybridRetriever:
    def __init__(self, dense_retriever, bm25_retriever, rrf_k=60):
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.rrf_k = rrf_k
    
    from evaluation.hybrid.normalize import normalize_result
    
    def search(self, query, k=10):
        '''
        Perform a hybrid search using dense and BM25 retrievers.
        
        Args:
            query (str): The search query.
            k (int): The number of results to return.
        
        Returns:
            list: A list of normalized results.
        '''
        dense_results = self.dense.search(query)
        bm_25_results = self.bm25.search(query, k)
        
        dense = [normalize_result(r) for r in dense_results["results"]]
        bm_25 = [normalize_result(r) for r in bm_25_results]
        
        from evaluation.hybrid.rrf import reciprocal_rank_fusion
        
        assert all("paper_id" in r and "chunk_id" in r for r in dense)
        assert all("paper_id" in r and "chunk_id" in r for r in bm_25)
        
        fused = reciprocal_rank_fusion(
            [dense, bm_25], 
            k=self.rrf_k
        )
        
        return fused[:k]