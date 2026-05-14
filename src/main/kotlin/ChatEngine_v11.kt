package com.memory.engine

import com.memory.store.*
import com.memory.retrieval.*
import com.memory.temporal.*

/**
 * ChatEngine v11 — Full-Stack ADD-only Memory Architecture
 *
 * 架构演进：
 * v10: ADD-only + MultiSignalRetriever + RetrievalAgent + Admission + ContextExpander
 * v11: v10 + TemporalReasoner + ValidityWindow + AccessFrequencyTracker + ProvenanceTracker
 *
 * 完整管线：
 * ┌─────────────────────────────────────────────────────────────────────┐
 * │  INPUT: User Message                                                │
 * │    ↓                                                                │
 * │  FactExtractor → extract facts from message                        │
 * │    ↓                                                                │
 * │  ValidityWindow → infer natural lifespan of each fact              │
 * │    ↓                                                                │
 * │  AdmissionController → gate (5-dimension scoring)                  │
 * │    ↓                                                                │
 * │  ProvenanceTracker → record birth certificate                      │
 * │    ↓                                                                │
 * │  AppendOnlyStore → immutable ADD                                   │
 * └─────────────────────────────────────────────────────────────────────┘
 *
 * ┌─────────────────────────────────────────────────────────────────────┐
 * │  RETRIEVAL: Query Processing                                        │
 * │    ↓                                                                │
 * │  RetrievalAgent → route strategy (Direct/Split/Chain)              │
 * │    ↓                                                                │
 * │  TemporalReasoner → detect time intent, shift anchor               │
 * │    ↓                                                                │
 * │  MultiSignalRetriever v2 → rank (semantic×keyword×entity×temporal) │
 * │    ↓                                                                │
 * │  ValidityWindow → validity score as additional signal              │
 * │    ↓                                                                │
 * │  AccessFrequencyTracker → frequency boost                          │
 * │    ↓                                                                │
 * │  ProvenanceTracker → provenance signal (confidence)                │
 * │    ↓                                                                │
 * │  ConflictResolution → mark current vs historical                   │
 * │    ↓                                                                │
 * │  ContextExpander → nucleus + neighbors                             │
 * │    ↓                                                                │
 * │  OUTPUT: Ranked, expanded, annotated context                       │
 * └─────────────────────────────────────────────────────────────────────┘
 *
 * 新增信号融合（v11 扩展）：
 * base_score = MultiSignalRetriever.rank() output
 * final_score = base_score × validity_score × frequency_boost × provenance_signal
 *
 * 设计哲学：
 * - 每个模块是正交维度的处理器，可独立替换
 * - 所有后处理都是乘法叠加（可组合、可关闭）
 * - 关闭任何一个新模块 = 该维度退化为 1.0（neutral）
 */
class ChatEngineV11(
    // Storage
    private val store: AppendOnlyStore = AppendOnlyStore(),
    
    // Ingestion pipeline
    private val extractor: FactExtractor = FactExtractor(),
    private val admissionController: AdmissionController = AdmissionController(),
    private val validityWindow: ValidityWindow = ValidityWindow(),
    private val provenanceTracker: ProvenanceTracker = ProvenanceTracker(),
    
    // Retrieval pipeline
    private val retriever: MultiSignalRetriever = MultiSignalRetriever(),
    private val agent: RetrievalAgent = RetrievalAgent(),
    private val frequencyTracker: AccessFrequencyTracker = AccessFrequencyTracker(),
    private val contextExpander: ContextExpander = ContextExpander(),
    
    // Config
    private val contextBudget: Int = 5,
    private val conflictMode: ConflictMode = ConflictMode.CURRENT_ONLY,
    private val enableValidityFilter: Boolean = true,
    private val enableFrequencyBoost: Boolean = true,
    private val enableProvenanceSignal: Boolean = true
) {
    // Turn counter for provenance
    private var turnCounter: Int = 0
    
    // Validity cache: memory_id → ValidityInfo
    private val validityCache: MutableMap<String, ValidityInfo> = mutableMapOf()
    
    /**
     * Process user turn (ingestion + retrieval)
     */
    fun processUserTurn(
        userId: String,
        message: String,
        sessionId: String = "default",
    ): TurnResult {
        val now = System.currentTimeMillis()
        turnCounter++
        val turnId = "turn_${turnCounter}"
        
        // === INGESTION PIPELINE ===
        
        // 1. Extract facts
        val facts = extractor.extract(message)
        
        val storedEntries = mutableListOf<MemoryEntry>()
        val rejectedFacts = mutableListOf<String>()
        
        for (fact in facts) {
            // 2. Admission gate
            val admission = admissionController.evaluate(
                content = fact.content,
                entities = fact.entities,
                existingMemories = store.getAll(userId).map { it.content }
            )
            
            if (!admission.admitted) {
                rejectedFacts.add(fact.content)
                // Record provenance even for rejected (for debugging)
                provenanceTracker.record(
                    memoryId = "rejected_${turnId}_${rejectedFacts.size}",
                    turnId = turnId,
                    sessionId = sessionId,
                    userId = userId,
                    originalMessage = message,
                    extractionConfidence = admission.overallScore,
                    admissionDecision = AdmissionOutcome.REJECTED
                )
                continue
            }
            
            // 3. Infer validity window
            val validity = validityWindow.inferValidity(fact.content, fact.entities)
            
            // 4. Store (ADD-only)
            val entry = store.add(
                content = fact.content,
                source = MemorySource.USER,
                entities = fact.entities,
                timestamp = now,
                sessionId = sessionId,
                userId = userId,
                metadata = mapOf(
                    "original_message" to message,
                    "validity_category" to validity.category.name,
                    "validity_days" to validity.estimatedDays.toString()
                )
            )
            storedEntries.add(entry)
            
            // 5. Record provenance
            provenanceTracker.record(
                memoryId = entry.id,
                turnId = turnId,
                sessionId = sessionId,
                userId = userId,
                originalMessage = message,
                extractionConfidence = admission.overallScore,
                admissionDecision = if (admission.fastPath) AdmissionOutcome.FAST_PATH 
                                    else AdmissionOutcome.ADMITTED
            )
            
            // 6. Cache validity
            validityCache[entry.id] = validity
        }
        
        // === RETRIEVAL PIPELINE ===
        val allMemories = store.getAll(userId)
        
        if (allMemories.isEmpty()) {
            return TurnResult(
                context = emptyList(),
                newFacts = storedEntries,
                rejectedFacts = rejectedFacts,
                strategy = RetrievalStrategy.DIRECT,
                conflicts = emptyList(),
                turnId = turnId
            )
        }
        
        // 7. Build candidates
        val queryEntities = facts.flatMap { it.entities }.distinct()
        val candidates = buildCandidates(message, allMemories)
        
        // 8. Route through RetrievalAgent + MultiSignalRetriever (with TemporalReasoner)
        val retrievalResult = agent.retrieve(message, candidates, queryEntities, now)
        var ranked = retrievalResult.ranked
        
        // 9. Post-retrieval signal fusion (v11 additions)
        ranked = applyV11Signals(ranked, now)
        
        // 10. Record access frequency for top results
        val topK = ranked.take(contextBudget)
        frequencyTracker.recordAccessBatch(topK.map { it.id })
        
        // 11. Conflict resolution
        val entityGroups = ranked
            .filter { it.entry?.entities?.isNotEmpty() == true }
            .flatMap { mem -> mem.entry!!.entities.map { e -> e to mem } }
            .groupBy({ it.first }, { it.second })
        
        val conflicts = retriever.resolveConflicts(ranked, entityGroups)
        
        // 12. Select context based on conflict mode
        val contextMemories = when (conflictMode) {
            ConflictMode.CURRENT_ONLY -> conflicts.current.take(contextBudget)
            ConflictMode.WITH_HISTORY -> ranked.take(contextBudget)
            ConflictMode.ANNOTATED -> annotateMemories(conflicts, contextBudget)
        }
        
        return TurnResult(
            context = contextMemories.map { formatContext(it, conflicts) },
            newFacts = storedEntries,
            rejectedFacts = rejectedFacts,
            strategy = retrievalResult.strategy,
            conflicts = conflicts.historical.map { it.content },
            turnId = turnId
        )
    }
    
    /**
     * v11 信号融合：validity × frequency × provenance
     * 每个都是乘法因子，关闭 = 1.0
     */
    private fun applyV11Signals(ranked: List<RankedMemory>, queryTime: Long): List<RankedMemory> {
        return ranked.map { mem ->
            var adjustedScore = mem.score
            
            // Validity window signal
            if (enableValidityFilter && mem.entry != null) {
                val validity = validityCache[mem.id] 
                    ?: validityWindow.inferValidity(mem.content, mem.entry.entities)
                val validityScore = validityWindow.computeValidityScore(
                    mem.entry.timestamp, queryTime, validity
                )
                adjustedScore *= validityScore
            }
            
            // Frequency boost
            if (enableFrequencyBoost) {
                val boost = frequencyTracker.computeBoost(mem.id)
                adjustedScore *= boost
            }
            
            // Provenance signal
            if (enableProvenanceSignal) {
                val provenanceSignal = provenanceTracker.computeProvenanceSignal(mem.id)
                adjustedScore *= provenanceSignal
            }
            
            mem.copy(score = adjustedScore)
        }.sortedByDescending { it.score }
    }
    
    /**
     * Build candidates with synthetic semantic scores
     * (In production: vector search)
     */
    private fun buildCandidates(query: String, memories: List<MemoryEntry>): List<RetrievalCandidate> {
        return memories.map { entry ->
            RetrievalCandidate(
                id = entry.id,
                content = entry.content,
                semanticScore = computeSyntheticSemantic(query, entry.content),
                entry = entry
            )
        }
    }
    
    /**
     * Synthetic semantic score (placeholder for real embeddings)
     */
    private fun computeSyntheticSemantic(query: String, content: String): Double {
        val qChars = query.toSet()
        val cChars = content.toSet()
        val intersection = qChars.intersect(cChars).size
        val union = qChars.union(cChars).size
        return if (union == 0) 0.0 else intersection.toDouble() / union
    }
    
    private fun annotateMemories(conflicts: ConflictResolution, budget: Int): List<RankedMemory> {
        val result = mutableListOf<RankedMemory>()
        result.addAll(conflicts.current.take(budget))
        val remaining = budget - result.size
        if (remaining > 0) {
            result.addAll(conflicts.historical.take(remaining))
        }
        return result
    }
    
    private fun formatContext(mem: RankedMemory, conflicts: ConflictResolution): String {
        val isHistorical = conflicts.historical.any { it.id == mem.id }
        val prefix = if (isHistorical) "[历史] " else ""
        return "$prefix${mem.content}"
    }
    
    /**
     * 回答 "你怎么知道 X 的？" — 利用 ProvenanceTracker
     */
    fun explainMemory(memoryId: String): String? {
        val provenance = provenanceTracker.getProvenance(memoryId) ?: return null
        return "这条信息来自你在 session ${provenance.sessionId} 中说的: \"${provenance.originalMessage}\""
    }
    
    /**
     * 获取引擎统计
     */
    fun getStats(userId: String): EngineStats {
        return EngineStats(
            totalMemories = store.count(userId),
            provenanceStats = provenanceTracker.getStats(),
            frequencyStats = frequencyTracker.getStats(),
            turnsProcessed = turnCounter
        )
    }
}

// === Supporting Types ===

enum class ConflictMode { CURRENT_ONLY, WITH_HISTORY, ANNOTATED }
enum class RetrievalStrategy { DIRECT, SPLIT_QUERY, CHAIN_OF_QUERY }

data class TurnResult(
    val context: List<String>,
    val newFacts: List<MemoryEntry>,
    val rejectedFacts: List<String>,
    val strategy: RetrievalStrategy,
    val conflicts: List<String>,
    val turnId: String
)

data class EngineStats(
    val totalMemories: Int,
    val provenanceStats: ProvenanceStats,
    val frequencyStats: FrequencyStats,
    val turnsProcessed: Int
)
