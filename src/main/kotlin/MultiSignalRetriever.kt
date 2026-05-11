package com.memory.retrieval

import com.memory.store.MemoryEntry
import com.memory.store.MemorySource
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min

/**
 * Multi-Signal Retriever (受 Mem0 v3 + MemMachine 启发)
 *
 * 在 ADD-only 存储上实现智能检索：
 * - 矛盾解决：不删除旧记忆，通过 recency + relevance 自然降权
 * - 时间推理：temporal signal 区分 "过去住在X" vs "现在住在Y"
 * - Entity boosting：query 中的 entity 与记忆中的 entity 匹配时加权
 * - Source weighting：USER/AGENT/SYSTEM 来源有不同基础权重
 *
 * 核心公式:
 * final_score = semantic_score × recency_boost × entity_boost × source_weight × access_frequency
 *
 * 这是"查询时的视图函数"——不改变底层数据，只改变看数据的方式
 */
class MultiSignalRetriever(
    // Semantic retrieval weight
    private val semanticWeight: Double = 0.5,
    // BM25/keyword weight
    private val keywordWeight: Double = 0.2,
    // Entity matching weight
    private val entityWeight: Double = 0.15,
    // Temporal recency weight
    private val temporalWeight: Double = 0.15,
    
    // Time decay parameters
    private val halfLifeDays: Double = 30.0,      // 30天半衰期
    private val minRecencyScore: Double = 0.1,     // 最低时间分（不降为0）
    
    // Source weights
    private val userSourceWeight: Double = 1.0,
    private val agentSourceWeight: Double = 0.9,   // Agent facts slightly lower
    private val systemSourceWeight: Double = 0.7
) {
    
    /**
     * 多信号检索
     *
     * @param query 用户查询
     * @param candidates 语义检索初步候选 (from vector search)
     * @param queryEntities 从 query 中提取的 entities
     * @param currentTime 当前时间
     * @param contextEntities 当前对话上下文中的 entities
     * @return 按综合分数排序的结果
     */
    fun rank(
        query: String,
        candidates: List<RetrievalCandidate>,
        queryEntities: List<String> = emptyList(),
        currentTime: Long = System.currentTimeMillis(),
        contextEntities: List<String> = emptyList()
    ): List<RankedMemory> {
        if (candidates.isEmpty()) return emptyList()
        
        return candidates.map { candidate ->
            val signals = computeSignals(
                candidate = candidate,
                query = query,
                queryEntities = queryEntities,
                contextEntities = contextEntities,
                currentTime = currentTime
            )
            
            val finalScore = combineSignals(signals)
            
            RankedMemory(
                id = candidate.id,
                content = candidate.content,
                score = finalScore,
                signals = signals,
                entry = candidate.entry
            )
        }.sortedByDescending { it.score }
    }
    
    /**
     * 计算各维度信号
     */
    private fun computeSignals(
        candidate: RetrievalCandidate,
        query: String,
        queryEntities: List<String>,
        contextEntities: List<String>,
        currentTime: Long
    ): SignalScores {
        // 1. Semantic score (from vector search)
        val semantic = candidate.semanticScore
        
        // 2. Keyword/BM25 score
        val keyword = computeKeywordScore(query, candidate.content)
        
        // 3. Entity boost
        val entity = computeEntityScore(queryEntities, contextEntities, candidate.entry.entities)
        
        // 4. Temporal recency
        val recency = computeRecencyScore(candidate.entry.timestamp, currentTime)
        
        // 5. Source weight
        val source = when (candidate.entry.source) {
            MemorySource.USER -> userSourceWeight
            MemorySource.AGENT -> agentSourceWeight
            MemorySource.SYSTEM -> systemSourceWeight
        }
        
        return SignalScores(
            semantic = semantic,
            keyword = keyword,
            entity = entity,
            recency = recency,
            sourceWeight = source
        )
    }
    
    /**
     * 信号融合
     * 加权平均 × source weight
     */
    private fun combineSignals(signals: SignalScores): Double {
        val weightedSum = (
            signals.semantic * semanticWeight +
            signals.keyword * keywordWeight +
            signals.entity * entityWeight +
            signals.recency * temporalWeight
        )
        return weightedSum * signals.sourceWeight
    }
    
    /**
     * 简化 BM25 关键词匹配
     */
    private fun computeKeywordScore(query: String, content: String): Double {
        val queryTokens = tokenize(query)
        val contentTokens = tokenize(content).toSet()
        if (queryTokens.isEmpty()) return 0.0
        
        val matchCount = queryTokens.count { it in contentTokens }
        return matchCount.toDouble() / queryTokens.size
    }
    
    /**
     * Entity 匹配分数
     * query/context 中的 entity 出现在记忆中 → boost
     */
    private fun computeEntityScore(
        queryEntities: List<String>,
        contextEntities: List<String>,
        memoryEntities: List<String>
    ): Double {
        if (memoryEntities.isEmpty()) return 0.0
        
        val allQueryEntities = (queryEntities + contextEntities).toSet()
        if (allQueryEntities.isEmpty()) return 0.3 // neutral baseline
        
        val matchCount = memoryEntities.count { it in allQueryEntities }
        return min(1.0, matchCount.toDouble() / max(1, allQueryEntities.size))
    }
    
    /**
     * 时间衰减分数
     * 越新的记忆分越高，但永不降为0（保留可发现性）
     */
    private fun computeRecencyScore(memoryTimestamp: Long, currentTime: Long): Double {
        val ageMs = currentTime - memoryTimestamp
        val ageDays = ageMs.toDouble() / (1000 * 60 * 60 * 24)
        
        // Exponential decay with half-life
        val decay = exp(-0.693 * ageDays / halfLifeDays)
        
        // Clamp to [minRecencyScore, 1.0]
        return max(minRecencyScore, decay)
    }
    
    /**
     * 简化分词
     */
    private fun tokenize(text: String): List<String> {
        // Chinese bigrams + word splitting
        val tokens = mutableListOf<String>()
        
        // Bigrams for Chinese
        for (i in 0 until text.length - 1) {
            if (text[i].code > 0x4E00) {
                tokens.add(text.substring(i, i + 2))
            }
        }
        
        // Whitespace split for mixed content
        tokens.addAll(text.split(Regex("[\\s,，。！？、；：]+")).filter { it.length >= 2 })
        
        return tokens
    }
    
    /**
     * 冲突检测视图
     * 
     * 不删除冲突记忆，而是标记出哪些是"最新版本"
     * 调用方可以选择只展示最新，或展示完整历史
     */
    fun resolveConflicts(
        ranked: List<RankedMemory>,
        entityGroups: Map<String, List<RankedMemory>>
    ): ConflictResolution {
        val currentVersion = mutableListOf<RankedMemory>()
        val historical = mutableListOf<RankedMemory>()
        
        // 对于同 entity 的记忆，最新的是 "current"，其余是 "historical"
        for ((entity, memories) in entityGroups) {
            if (memories.size <= 1) {
                currentVersion.addAll(memories)
                continue
            }
            
            val sortedByTime = memories.sortedByDescending { it.entry?.timestamp ?: 0 }
            currentVersion.add(sortedByTime.first())
            historical.addAll(sortedByTime.drop(1))
        }
        
        // 非 entity-grouped 的记忆直接归入 current
        val grouped = entityGroups.values.flatten().map { it.id }.toSet()
        currentVersion.addAll(ranked.filter { it.id !in grouped })
        
        return ConflictResolution(
            current = currentVersion.sortedByDescending { it.score },
            historical = historical,
            conflictCount = historical.size
        )
    }
}

// === Data Classes ===

data class RetrievalCandidate(
    val id: String,
    val content: String,
    val semanticScore: Double,
    val entry: MemoryEntry
)

data class RankedMemory(
    val id: String,
    val content: String,
    val score: Double,
    val signals: SignalScores,
    val entry: MemoryEntry? = null
)

data class SignalScores(
    val semantic: Double,
    val keyword: Double,
    val entity: Double,
    val recency: Double,
    val sourceWeight: Double
) {
    override fun toString(): String =
        "sem=${String.format("%.3f", semantic)} kw=${String.format("%.3f", keyword)} " +
        "ent=${String.format("%.3f", entity)} rec=${String.format("%.3f", recency)} " +
        "src=${String.format("%.2f", sourceWeight)}"
}

data class ConflictResolution(
    val current: List<RankedMemory>,
    val historical: List<RankedMemory>,
    val conflictCount: Int
)
