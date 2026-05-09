package com.memory.retrieval

import com.memory.store.KnowledgeStore
import kotlinx.serialization.Serializable

/**
 * 混合检索器 (Hybrid Retriever)
 * 
 * 基于 VLDB 2026 + Cognis 论文验证的最佳实践：
 * Vector 70% + BM25 30% 通过 Reciprocal Rank Fusion (RRF) 融合
 * 
 * 设计哲学（正交维度）：
 * - 检索是查询时的视图函数，不改变存储的数据
 * - Vector = 语义相似度维度
 * - BM25 = 词汇精确匹配维度
 * - RRF = 统一的融合计算框架，不针对具体case设计特殊逻辑
 * 
 * 核心参数：
 * - vectorWeight: 0.7 (语义模糊匹配能力强)
 * - bm25Weight: 0.3 (精确名词/数字匹配能力强)
 * - rrf_k: 60 (RRF 平滑常数, 标准值)
 * - topK: 可配置的返回数量
 */
class HybridRetriever(
    private val vectorWeight: Double = 0.7,
    private val bm25Weight: Double = 0.3,
    private val rrfK: Int = 60,
    private val topK: Int = 10
) {
    
    /**
     * 混合检索入口
     * 
     * @param query 用户查询
     * @param vectorResults 向量检索结果 (已按相似度降序排列)
     * @param bm25Results BM25 检索结果 (已按 BM25 score 降序排列)
     * @return RRF 融合后的排序结果
     */
    fun retrieve(
        query: String,
        vectorResults: List<RetrievalCandidate>,
        bm25Results: List<RetrievalCandidate>
    ): List<RankedResult> {
        // Step 1: 计算 RRF 分数
        val rrfScores = mutableMapOf<String, RRFAccumulator>()
        
        // Vector 路径贡献
        vectorResults.forEachIndexed { rank, candidate ->
            val rrfScore = vectorWeight * (1.0 / (rrfK + rank + 1))
            val acc = rrfScores.getOrPut(candidate.id) { 
                RRFAccumulator(candidate = candidate) 
            }
            acc.vectorRank = rank + 1
            acc.vectorScore = rrfScore
            acc.totalScore += rrfScore
        }
        
        // BM25 路径贡献
        bm25Results.forEachIndexed { rank, candidate ->
            val rrfScore = bm25Weight * (1.0 / (rrfK + rank + 1))
            val acc = rrfScores.getOrPut(candidate.id) { 
                RRFAccumulator(candidate = candidate) 
            }
            acc.bm25Rank = rank + 1
            acc.bm25Score = rrfScore
            acc.totalScore += rrfScore
        }
        
        // Step 2: 排序并截取 topK
        return rrfScores.values
            .sortedByDescending { it.totalScore }
            .take(topK)
            .map { acc ->
                RankedResult(
                    id = acc.candidate.id,
                    content = acc.candidate.content,
                    metadata = acc.candidate.metadata,
                    rrfScore = acc.totalScore,
                    vectorRank = acc.vectorRank,
                    bm25Rank = acc.bm25Rank,
                    fusionDetail = FusionDetail(
                        vectorContribution = acc.vectorScore,
                        bm25Contribution = acc.bm25Score
                    )
                )
            }
    }
    
    /**
     * 带查询扩展的混合检索
     * 
     * 对于短查询，先用 LLM 扩展查询，再分别执行 Vector 和 BM25
     * 这解决了短查询在 BM25 上覆盖度不足的问题
     */
    fun expandedRetrieve(
        originalQuery: String,
        expandedQueries: List<String>,
        vectorResultsPerQuery: Map<String, List<RetrievalCandidate>>,
        bm25ResultsPerQuery: Map<String, List<RetrievalCandidate>>
    ): List<RankedResult> {
        // 合并所有查询的 vector 结果，去重保留最高排名
        val mergedVector = mergeAndDeduplicate(
            vectorResultsPerQuery.values.toList()
        )
        
        // 合并所有查询的 BM25 结果
        val mergedBM25 = mergeAndDeduplicate(
            bm25ResultsPerQuery.values.toList()
        )
        
        return retrieve(originalQuery, mergedVector, mergedBM25)
    }
    
    /**
     * 合并多组结果，同一 id 保留最高排名
     */
    private fun mergeAndDeduplicate(
        resultSets: List<List<RetrievalCandidate>>
    ): List<RetrievalCandidate> {
        val best = mutableMapOf<String, Pair<Int, RetrievalCandidate>>() // id -> (bestRank, candidate)
        
        for (results in resultSets) {
            results.forEachIndexed { rank, candidate ->
                val existing = best[candidate.id]
                if (existing == null || rank < existing.first) {
                    best[candidate.id] = Pair(rank, candidate)
                }
            }
        }
        
        // 按最佳排名排序
        return best.values.sortedBy { it.first }.map { it.second }
    }
    
    /**
     * BM25 简易实现
     * 
     * 正式部署应使用 Elasticsearch/Meilisearch，
     * 这里提供内存版本用于测试和小规模场景
     */
    fun bm25Score(
        query: String,
        document: String,
        avgDocLength: Double,
        k1: Double = 1.2,
        b: Double = 0.75
    ): Double {
        val queryTerms = tokenize(query)
        val docTerms = tokenize(document)
        val docLength = docTerms.size.toDouble()
        
        if (queryTerms.isEmpty() || docTerms.isEmpty()) return 0.0
        
        val termFreqs = docTerms.groupingBy { it }.eachCount()
        
        var score = 0.0
        for (term in queryTerms) {
            val tf = termFreqs[term]?.toDouble() ?: 0.0
            if (tf == 0.0) continue
            
            // Simplified IDF (without corpus-level stats, use constant)
            val idf = 1.0  // In production, use log((N - df + 0.5) / (df + 0.5) + 1)
            
            val numerator = tf * (k1 + 1)
            val denominator = tf + k1 * (1 - b + b * (docLength / avgDocLength))
            
            score += idf * (numerator / denominator)
        }
        
        return score
    }
    
    /**
     * 中文分词（简化版）
     * 生产环境应使用 jieba/HanLP
     */
    private fun tokenize(text: String): List<String> {
        // Bigram + Unigram 混合策略 for Chinese
        val chars = text.filter { !it.isWhitespace() && it !in "，。！？、；：""''（）" }
        val tokens = mutableListOf<String>()
        
        // Unigrams
        for (c in chars) {
            tokens.add(c.toString())
        }
        
        // Bigrams (重要：中文词多为2字)
        for (i in 0 until chars.length - 1) {
            tokens.add(chars.substring(i, i + 2))
        }
        
        return tokens
    }
}

// === Data Classes ===

data class RetrievalCandidate(
    val id: String,
    val content: String,
    val metadata: Map<String, String> = emptyMap(),
    val originalScore: Double = 0.0  // Vector similarity or BM25 score
)

data class RankedResult(
    val id: String,
    val content: String,
    val metadata: Map<String, String> = emptyMap(),
    val rrfScore: Double,
    val vectorRank: Int?,
    val bm25Rank: Int?,
    val fusionDetail: FusionDetail
)

data class FusionDetail(
    val vectorContribution: Double,
    val bm25Contribution: Double
) {
    val dominantSource: String
        get() = if (vectorContribution >= bm25Contribution) "vector" else "bm25"
}

private data class RRFAccumulator(
    val candidate: RetrievalCandidate,
    var vectorRank: Int? = null,
    var bm25Rank: Int? = null,
    var vectorScore: Double = 0.0,
    var bm25Score: Double = 0.0,
    var totalScore: Double = 0.0
)
