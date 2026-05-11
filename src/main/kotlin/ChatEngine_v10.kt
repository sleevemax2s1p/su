package com.memory.engine

import com.memory.store.*
import com.memory.retrieval.*

/**
 * ChatEngine v10 — ADD-only Architecture
 *
 * 架构演进：
 * v8: CRUD store + HybridRetriever + SelectiveForgetting
 * v9: v8 + AdaptiveContextSelector + MemoryPolicy + EntityRelationGraph
 * v10: 根本性重构 — ADD-only store + multi-signal retrieval + retrieval agent
 *
 * 核心差异：
 * 1. 存储层只有 ADD（无 UPDATE/DELETE）
 * 2. 所有"智能"移到检索阶段
 * 3. 矛盾/更新/遗忘 = 检索时的视图函数
 * 4. Retrieval Agent 自动路由查询策略
 *
 * 设计哲学对齐：
 * - 信息不分事件/知识/记忆 → 统一的 MemoryEntry
 * - 正交维度（本体×时间×权重）→ AppendOnlyStore + SignalScores
 * - 上层组织 = 查询时视图函数 → MultiSignalRetriever + RetrievalAgent
 * - 碰撞用统一计算框架 → combineSignals() + resolveConflicts()
 */
class ChatEngineV10(
    private val store: AppendOnlyStore = AppendOnlyStore(),
    private val retriever: MultiSignalRetriever = MultiSignalRetriever(),
    private val agent: RetrievalAgent = RetrievalAgent(),
    private val extractor: FactExtractor = FactExtractor(),
    private val contextBudget: Int = 5,       // max memories to inject
    private val conflictMode: ConflictMode = ConflictMode.CURRENT_ONLY
) {
    
    /**
     * Process user turn:
     * 1. Extract facts from user message → ADD to store
     * 2. Retrieve relevant memories for context
     * 3. Return context + conflict info
     */
    fun processUserTurn(
        userId: String,
        message: String,
        sessionId: String? = null,
        existingContext: List<String> = emptyList()
    ): TurnResult {
        val now = System.currentTimeMillis()
        
        // === Phase 1: Extraction (ADD-only) ===
        val facts = extractor.extract(message)
        val newEntries = facts.map { fact ->
            store.add(
                content = fact.content,
                source = MemorySource.USER,
                entities = fact.entities,
                timestamp = now,
                sessionId = sessionId,
                userId = userId,
                metadata = mapOf("original_message" to message)
            )
        }
        
        // === Phase 2: Retrieval ===
        val allMemories = store.getAll(userId)
        
        // Skip if empty
        if (allMemories.isEmpty()) {
            return TurnResult(
                context = emptyList(),
                newFacts = newEntries,
                strategy = RetrievalStrategy.DIRECT,
                conflicts = emptyList()
            )
        }
        
        // Build candidates (in production: vector search first)
        val queryEntities = facts.flatMap { it.entities }.distinct()
        val candidates = buildCandidates(message, allMemories)
        
        // Route through RetrievalAgent
        val analysis = agent.analyzeQuery(message, queryEntities)
        val result = agent.retrieve(
            query = message,
            allCandidates = candidates,
            analysis = analysis,
            queryEntities = queryEntities,
            currentTime = now
        )
        
        // === Phase 3: Conflict Resolution ===
        val entityGroups = groupByEntity(result.ranked)
        val resolution = retriever.resolveConflicts(result.ranked, entityGroups)
        
        // === Phase 4: Context Selection ===
        val contextMemories = when (conflictMode) {
            ConflictMode.CURRENT_ONLY -> resolution.current.take(contextBudget)
            ConflictMode.WITH_HISTORY -> result.ranked.take(contextBudget)
            ConflictMode.ANNOTATED -> annotateConflicts(resolution).take(contextBudget)
        }
        
        return TurnResult(
            context = contextMemories.map { formatForContext(it) },
            newFacts = newEntries,
            strategy = analysis.strategy,
            conflicts = if (resolution.conflictCount > 0) {
                resolution.historical.map { ConflictInfo(it.content, "superseded") }
            } else emptyList()
        )
    }
    
    /**
     * Process agent response:
     * Agent-generated facts are first-class memories (Mem0 v3 insight)
     */
    fun processAgentTurn(
        userId: String,
        response: String,
        sessionId: String? = null
    ): List<MemoryEntry> {
        val facts = extractor.extractAgentFacts(response)
        return facts.map { fact ->
            store.add(
                content = fact.content,
                source = MemorySource.AGENT,
                entities = fact.entities,
                timestamp = System.currentTimeMillis(),
                sessionId = sessionId,
                userId = userId,
                metadata = mapOf("agent_response" to response)
            )
        }
    }
    
    /**
     * Query memories without adding new ones
     * (for analytics, debugging, or explicit recall)
     */
    fun query(
        userId: String,
        query: String,
        topK: Int = 5
    ): List<RankedMemory> {
        val allMemories = store.getAll(userId)
        if (allMemories.isEmpty()) return emptyList()
        
        val candidates = buildCandidates(query, allMemories)
        val result = agent.retrieve(query, candidates, queryEntities = emptyList())
        return result.ranked.take(topK)
    }
    
    /**
     * Get full history for an entity (temporal view)
     */
    fun entityHistory(entity: String): List<MemoryEntry> {
        return store.getByEntity(entity).sortedBy { it.timestamp }
    }
    
    /**
     * Stats
     */
    fun stats(userId: String? = null): EngineStats {
        return EngineStats(
            totalMemories = store.count(userId),
            entityCount = store.entityCount(),
            version = "v10-add-only"
        )
    }
    
    // === Private Helpers ===
    
    private fun buildCandidates(query: String, memories: List<MemoryEntry>): List<RetrievalCandidate> {
        // In production: vector search gives real semantic scores
        // Here: simplified cosine approximation via keyword overlap
        return memories.map { mem ->
            val semanticScore = approximateSemantic(query, mem.content)
            RetrievalCandidate(mem.id, mem.content, semanticScore, mem)
        }
    }
    
    /**
     * Simplified semantic scoring (production would use embedding model)
     */
    private fun approximateSemantic(query: String, content: String): Double {
        val qChars = query.toSet()
        val cChars = content.toSet()
        val overlap = qChars.intersect(cChars).size
        val union = qChars.union(cChars).size
        return if (union == 0) 0.0 else (overlap.toDouble() / union) * 0.8 + 0.2
    }
    
    private fun groupByEntity(ranked: List<RankedMemory>): Map<String, List<RankedMemory>> {
        val groups = mutableMapOf<String, MutableList<RankedMemory>>()
        for (r in ranked) {
            val entities = r.entry?.entities ?: continue
            for (entity in entities) {
                groups.getOrPut(entity) { mutableListOf() }.add(r)
            }
        }
        // Only return groups with potential conflicts (>1 memory per entity)
        return groups.filter { it.value.size > 1 }
    }
    
    private fun annotateConflicts(resolution: ConflictResolution): List<RankedMemory> {
        // Annotate historical entries with "[历史]" prefix
        val annotated = resolution.current.toMutableList()
        for (hist in resolution.historical) {
            annotated.add(hist.copy(
                content = "[历史] ${hist.content}"
            ))
        }
        return annotated.sortedByDescending { it.score }
    }
    
    private fun formatForContext(memory: RankedMemory): String {
        val timeInfo = memory.entry?.let { 
            val ageMs = System.currentTimeMillis() - it.timestamp
            val ageDays = ageMs / (1000 * 60 * 60 * 24)
            if (ageDays == 0L) "今天" else "${ageDays}天前"
        } ?: ""
        
        return "${memory.content} ($timeInfo)"
    }
}

// === Supporting Types ===

/**
 * Fact extraction (simplified — in production would use LLM)
 */
class FactExtractor {
    fun extract(message: String): List<ExtractedFact> {
        // Simplified: treat each sentence as a fact
        // Production: LLM-based extraction with entity recognition
        val sentences = message.split(Regex("[。！？；\n]+"))
            .map { it.trim() }
            .filter { it.length >= 2 }
        
        return sentences.map { sentence ->
            ExtractedFact(
                content = sentence,
                entities = extractEntities(sentence)
            )
        }
    }
    
    fun extractAgentFacts(response: String): List<ExtractedFact> {
        // Only extract declarative facts from agent responses
        // Skip questions, confirmations, etc.
        val sentences = response.split(Regex("[。！？；\n]+"))
            .map { it.trim() }
            .filter { it.length >= 4 && !it.contains("?") && !it.contains("？") }
        
        return sentences.map { sentence ->
            ExtractedFact(content = sentence, entities = extractEntities(sentence))
        }
    }
    
    private fun extractEntities(text: String): List<String> {
        // Simplified entity extraction
        // Production: NER model or LLM-based
        val entities = mutableListOf<String>()
        
        // Pattern: Chinese names (2-3 chars)
        val namePattern = Regex("[\\u4e00-\\u9fa5]{2,3}(?=说|是|在|住|喜欢|工作|叫)")
        namePattern.findAll(text).forEach { entities.add(it.value) }
        
        // Location patterns
        val locPattern = Regex("(?:在|住|到|去)([\\u4e00-\\u9fa5]{2,4})")
        locPattern.findAll(text).forEach { entities.add(it.groupValues[1]) }
        
        return entities.distinct()
    }
}

data class ExtractedFact(
    val content: String,
    val entities: List<String>
)

enum class ConflictMode {
    CURRENT_ONLY,   // Only show most recent version
    WITH_HISTORY,   // Show all, ranked by score (default multi-signal)
    ANNOTATED       // Show all with [历史] annotation
}

data class TurnResult(
    val context: List<String>,
    val newFacts: List<MemoryEntry>,
    val strategy: RetrievalStrategy,
    val conflicts: List<ConflictInfo>
)

data class ConflictInfo(
    val content: String,
    val status: String  // "superseded", "contradicted"
)

data class EngineStats(
    val totalMemories: Int,
    val entityCount: Int,
    val version: String
)
