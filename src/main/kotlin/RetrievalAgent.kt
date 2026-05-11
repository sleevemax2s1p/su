package com.memory.retrieval

import com.memory.store.MemoryEntry
import com.memory.store.MemorySource

/**
 * Retrieval Agent (受 MemMachine 启发)
 *
 * 核心发现：检索阶段优化 dominates 存储阶段 (+4.2% vs +0.8%)
 * 
 * 三策略路由：
 * 1. Direct — 简单查询，单次向量检索
 * 2. SplitQuery — 复杂多面查询，拆分为子查询后合并
 * 3. ChainOfQuery — 多跳查询，前一步结果指导后一步
 *
 * 路由决策：基于查询复杂度分析（entity count, clause count, temporal references）
 * 
 * 与 ADD-only 架构的协同：
 * - Direct: 直接检索 → MultiSignalRetriever 排序
 * - SplitQuery: 多次检索 → 合并去重 → 重排序
 * - ChainOfQuery: 序列检索 → 上下文传递 → 最终排序
 *
 * Multi-Query Reranking (MQR):
 * 对同一记忆被多个子查询命中的情况，给予 recall boost
 */

// === Strategy Enum ===
enum class RetrievalStrategy {
    DIRECT,
    SPLIT_QUERY,
    CHAIN_OF_QUERY
}

// === Query Analysis ===
data class QueryAnalysis(
    val strategy: RetrievalStrategy,
    val confidence: Double,
    val entityCount: Int,
    val clauseCount: Int,
    val hasTemporalRef: Boolean,
    val hasCausalRef: Boolean,
    val subQueries: List<String> = emptyList()
)

/**
 * Retrieval Agent — 路由 + 执行 + 重排序
 */
class RetrievalAgent(
    private val multiSignalRetriever: MultiSignalRetriever = MultiSignalRetriever(),
    // MQR parameters
    private val mqrBoostFactor: Double = 1.3,  // 被多次命中时的 boost
    private val mqrMinHits: Int = 2,            // 触发 boost 的最少命中次数
    // Split parameters
    private val maxSubQueries: Int = 4,
    // Chain parameters
    private val maxChainDepth: Int = 3
) {
    // === Temporal & Causal patterns ===
    private val temporalPatterns = listOf(
        Regex("以前|之前|过去|曾经|原来|上次|去年|前天|昨天"),
        Regex("现在|目前|当前|最近|刚才|今天"),
        Regex("后来|之后|接着|然后|最终")
    )
    
    private val causalPatterns = listOf(
        Regex("为什么|因为|所以|导致|引起|由于"),
        Regex("如果|假如|要是|万一")
    )
    
    private val conjunctionPatterns = listOf(
        Regex("和|与|以及|还有|另外"),
        Regex("而且|并且|同时"),
        Regex("[,，;；]")  // punctuation as clause separator
    )
    
    /**
     * 分析查询复杂度，决定路由策略
     */
    fun analyzeQuery(query: String, contextEntities: List<String> = emptyList()): QueryAnalysis {
        // Count entities in query
        val entityCount = contextEntities.size + countInlineEntities(query)
        
        // Count clauses
        val clauseCount = countClauses(query)
        
        // Temporal references
        val hasTemporalRef = temporalPatterns.any { it.containsMatchIn(query) }
        
        // Causal references  
        val hasCausalRef = causalPatterns.any { it.containsMatchIn(query) }
        
        // Strategy decision
        val (strategy, confidence) = decideStrategy(
            entityCount, clauseCount, hasTemporalRef, hasCausalRef, query.length
        )
        
        // Generate sub-queries if needed
        val subQueries = when (strategy) {
            RetrievalStrategy.SPLIT_QUERY -> splitQuery(query, clauseCount)
            RetrievalStrategy.CHAIN_OF_QUERY -> chainDecompose(query)
            RetrievalStrategy.DIRECT -> emptyList()
        }
        
        return QueryAnalysis(
            strategy = strategy,
            confidence = confidence,
            entityCount = entityCount,
            clauseCount = clauseCount,
            hasTemporalRef = hasTemporalRef,
            hasCausalRef = hasCausalRef,
            subQueries = subQueries
        )
    }
    
    /**
     * 执行检索（根据分析结果选择策略）
     */
    fun retrieve(
        query: String,
        allCandidates: List<RetrievalCandidate>,
        analysis: QueryAnalysis? = null,
        queryEntities: List<String> = emptyList(),
        currentTime: Long = System.currentTimeMillis(),
        contextEntities: List<String> = emptyList()
    ): RetrievalResult {
        val queryAnalysis = analysis ?: analyzeQuery(query, contextEntities)
        
        return when (queryAnalysis.strategy) {
            RetrievalStrategy.DIRECT -> executeDirect(
                query, allCandidates, queryEntities, currentTime, contextEntities
            )
            RetrievalStrategy.SPLIT_QUERY -> executeSplitQuery(
                query, queryAnalysis.subQueries, allCandidates, 
                queryEntities, currentTime, contextEntities
            )
            RetrievalStrategy.CHAIN_OF_QUERY -> executeChainOfQuery(
                query, queryAnalysis.subQueries, allCandidates,
                queryEntities, currentTime, contextEntities
            )
        }
    }
    
    // === Strategy Execution ===
    
    private fun executeDirect(
        query: String,
        candidates: List<RetrievalCandidate>,
        queryEntities: List<String>,
        currentTime: Long,
        contextEntities: List<String>
    ): RetrievalResult {
        val ranked = multiSignalRetriever.rank(
            query, candidates, queryEntities, currentTime, contextEntities
        )
        return RetrievalResult(
            ranked = ranked,
            strategy = RetrievalStrategy.DIRECT,
            subQueryResults = emptyMap(),
            mqrApplied = false
        )
    }
    
    private fun executeSplitQuery(
        originalQuery: String,
        subQueries: List<String>,
        candidates: List<RetrievalCandidate>,
        queryEntities: List<String>,
        currentTime: Long,
        contextEntities: List<String>
    ): RetrievalResult {
        if (subQueries.isEmpty()) {
            return executeDirect(originalQuery, candidates, queryEntities, currentTime, contextEntities)
        }
        
        // Execute each sub-query
        val subResults = mutableMapOf<String, List<RankedMemory>>()
        val hitCounts = mutableMapOf<String, Int>() // memory_id → hit count
        
        for (sq in subQueries.take(maxSubQueries)) {
            val ranked = multiSignalRetriever.rank(
                sq, candidates, queryEntities, currentTime, contextEntities
            )
            subResults[sq] = ranked
            
            // Track which memories are hit by multiple sub-queries
            for (r in ranked.take(10)) { // top-10 per sub-query
                hitCounts[r.id] = (hitCounts[r.id] ?: 0) + 1
            }
        }
        
        // Merge: take all unique results, apply MQR boost
        val merged = mergeWithMQR(subResults.values.flatten(), hitCounts)
        
        return RetrievalResult(
            ranked = merged,
            strategy = RetrievalStrategy.SPLIT_QUERY,
            subQueryResults = subResults,
            mqrApplied = hitCounts.any { it.value >= mqrMinHits }
        )
    }
    
    private fun executeChainOfQuery(
        originalQuery: String,
        chainSteps: List<String>,
        candidates: List<RetrievalCandidate>,
        queryEntities: List<String>,
        currentTime: Long,
        contextEntities: List<String>
    ): RetrievalResult {
        if (chainSteps.isEmpty()) {
            return executeDirect(originalQuery, candidates, queryEntities, currentTime, contextEntities)
        }
        
        // Execute chain: each step's top results provide context for next step
        var accumulatedEntities = queryEntities.toMutableList()
        val allResults = mutableListOf<RankedMemory>()
        val subResults = mutableMapOf<String, List<RankedMemory>>()
        
        for ((i, step) in chainSteps.take(maxChainDepth).withIndex()) {
            val ranked = multiSignalRetriever.rank(
                step, candidates, accumulatedEntities, currentTime, contextEntities
            )
            subResults["step_${i+1}: $step"] = ranked
            
            // Extract entities from top results for next step
            val topEntities = ranked.take(3)
                .flatMap { it.entry?.entities ?: emptyList() }
            accumulatedEntities.addAll(topEntities)
            
            allResults.addAll(ranked.take(5))
        }
        
        // Final ranking with accumulated context
        val finalRanked = multiSignalRetriever.rank(
            originalQuery, candidates, accumulatedEntities.distinct(), currentTime, contextEntities
        )
        
        return RetrievalResult(
            ranked = finalRanked,
            strategy = RetrievalStrategy.CHAIN_OF_QUERY,
            subQueryResults = subResults,
            mqrApplied = false
        )
    }
    
    // === Multi-Query Reranking (MQR) ===
    
    private fun mergeWithMQR(
        allResults: List<RankedMemory>,
        hitCounts: Map<String, Int>
    ): List<RankedMemory> {
        // Deduplicate by id, keeping highest score
        val bestById = mutableMapOf<String, RankedMemory>()
        for (r in allResults) {
            val existing = bestById[r.id]
            if (existing == null || r.score > existing.score) {
                bestById[r.id] = r
            }
        }
        
        // Apply MQR boost
        return bestById.values.map { memory ->
            val hits = hitCounts[memory.id] ?: 1
            if (hits >= mqrMinHits) {
                // Boost score for memories hit by multiple sub-queries
                val boost = 1.0 + (mqrBoostFactor - 1.0) * (hits - 1) / maxSubQueries
                memory.copy(score = memory.score * boost)
            } else {
                memory
            }
        }.sortedByDescending { it.score }
    }
    
    // === Helper Functions ===
    
    private fun decideStrategy(
        entityCount: Int,
        clauseCount: Int,
        hasTemporalRef: Boolean,
        hasCausalRef: Boolean,
        queryLength: Int
    ): Pair<RetrievalStrategy, Double> {
        // Chain: causal or temporal multi-hop
        if (hasCausalRef && entityCount >= 2) {
            return RetrievalStrategy.CHAIN_OF_QUERY to 0.8
        }
        if (hasTemporalRef && clauseCount >= 2) {
            return RetrievalStrategy.CHAIN_OF_QUERY to 0.7
        }
        
        // Split: multiple independent aspects
        if (clauseCount >= 2 && entityCount >= 2) {
            return RetrievalStrategy.SPLIT_QUERY to 0.85
        }
        if (queryLength > 30 && clauseCount >= 2) {
            return RetrievalStrategy.SPLIT_QUERY to 0.7
        }
        
        // Default: Direct
        return RetrievalStrategy.DIRECT to 0.9
    }
    
    private fun countInlineEntities(query: String): Int {
        // Simple heuristic: count noun-like segments between particles
        val particles = Regex("[的地得了过着吗呢吧啊]")
        val segments = query.split(particles).filter { it.length >= 2 }
        return segments.size.coerceAtMost(5)
    }
    
    private fun countClauses(query: String): Int {
        var count = 1
        for (pattern in conjunctionPatterns) {
            count += pattern.findAll(query).count()
        }
        return count.coerceAtMost(6)
    }
    
    /**
     * 拆分复合查询为子查询
     */
    private fun splitQuery(query: String, clauseCount: Int): List<String> {
        // Split on conjunctions and punctuation
        val parts = query.split(Regex("[,，;；、和与以及还有另外而且并且同时]+"))
            .map { it.trim() }
            .filter { it.length >= 2 }
        
        return if (parts.size >= 2) parts.take(maxSubQueries)
        else listOf(query) // fallback: return original
    }
    
    /**
     * 分解因果/时序链为步骤
     */
    private fun chainDecompose(query: String): List<String> {
        // Split on causal/temporal connectors
        val connectors = Regex("(为什么|因为|所以|然后|之后|接着|后来|如果)")
        val parts = query.split(connectors)
            .map { it.trim() }
            .filter { it.length >= 2 }
        
        return if (parts.size >= 2) parts.take(maxChainDepth)
        else listOf(query)
    }
}

// === Result Data Class ===

data class RetrievalResult(
    val ranked: List<RankedMemory>,
    val strategy: RetrievalStrategy,
    val subQueryResults: Map<String, List<RankedMemory>>,
    val mqrApplied: Boolean
) {
    fun topK(k: Int): List<RankedMemory> = ranked.take(k)
}
