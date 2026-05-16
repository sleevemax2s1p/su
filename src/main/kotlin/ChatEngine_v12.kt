package com.memory.engine

import com.memory.store.*
import com.memory.retrieval.*
import com.memory.temporal.*
import com.memory.embedding.*
import com.memory.safety.*

/**
 * ChatEngine v12 — SemanticRule + PinnedMemoryGuard + EventBoundary Integration
 *
 * 架构演进：
 * v10: ADD-only + MultiSignalRetriever + RetrievalAgent + Admission + ContextExpander
 * v11: + TemporalReasoner + ValidityWindow + AccessFrequencyTracker + ProvenanceTracker
 * v12: + SemanticRuleProvider + PinnedMemoryGuard + EventBoundaryDetector
 *
 * v12 新增：
 * 1. SemanticRuleProvider 替代 CharOverlap 提供语义分（可插拔）
 * 2. PinnedMemoryGuard 保护安全关键记忆不被衰减淹没
 * 3. EventBoundaryDetector 让 ContextExpander 按事件边界展开
 *
 * 信号融合（完整公式）：
 * base_score = MultiSignalRetriever.rank() // semantic×0.5 + keyword×0.2 + entity×0.15 + temporal×0.15
 * v11_score = base_score × validity_score × frequency_boost × provenance_signal
 * v12_score = pinnedGuard.applyPinProtection(id, v11_score) // floor=0.5 for pinned
 */
class ChatEngineV12(
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

    // v12 additions
    private val embeddingProvider: EmbeddingProvider = SemanticRuleProvider(),
    private val pinnedGuard: PinnedMemoryGuard = PinnedMemoryGuard(),
    private val eventBoundaryDetector: EventBoundaryDetector = EventBoundaryDetector(),

    // Config
    private val contextBudget: Int = 5,
    private val conflictMode: ConflictMode = ConflictMode.CURRENT_ONLY,
    private val enableValidityFilter: Boolean = true,
    private val enableFrequencyBoost: Boolean = true,
    private val enableProvenanceSignal: Boolean = true,
    private val enablePinProtection: Boolean = true,
    private val enableEventBoundary: Boolean = true
) {
    private var turnCounter: Int = 0
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
        val facts = extractor.extract(message)
        val storedEntries = mutableListOf<MemoryEntry>()
        val rejectedFacts = mutableListOf<String>()

        for (fact in facts) {
            // Admission gate
            val admission = admissionController.evaluate(
                content = fact.content,
                entities = fact.entities,
                existingMemories = store.getAll(userId).map { it.content }
            )

            if (!admission.admitted) {
                rejectedFacts.add(fact.content)
                provenanceTracker.record(
                    memoryId = "rejected_${turnId}_${rejectedFacts.size}",
                    turnId = turnId, sessionId = sessionId, userId = userId,
                    originalMessage = message,
                    extractionConfidence = admission.overallScore,
                    admissionDecision = AdmissionOutcome.REJECTED
                )
                continue
            }

            // Validity inference
            val validity = validityWindow.inferValidity(fact.content, fact.entities)

            // Store (ADD-only)
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

            // Provenance
            provenanceTracker.record(
                memoryId = entry.id, turnId = turnId,
                sessionId = sessionId, userId = userId,
                originalMessage = message,
                extractionConfidence = admission.overallScore,
                admissionDecision = if (admission.fastPath) AdmissionOutcome.FAST_PATH
                                    else AdmissionOutcome.ADMITTED
            )

            // v12: Auto-pin detection
            if (enablePinProtection) {
                pinnedGuard.evaluateForPin(entry.id, fact.content, fact.entities)
            }

            // Cache validity
            validityCache[entry.id] = validity
        }

        // === RETRIEVAL PIPELINE ===
        val allMemories = store.getAll(userId)

        if (allMemories.isEmpty()) {
            return TurnResult(
                context = emptyList(), newFacts = storedEntries,
                rejectedFacts = rejectedFacts, strategy = RetrievalStrategy.DIRECT,
                conflicts = emptyList(), turnId = turnId
            )
        }

        // Build candidates using SemanticRuleProvider (v12 upgrade)
        val candidates = buildCandidates(message, allMemories)

        // Route + rank (with TemporalReasoner)
        val queryEntities = facts.flatMap { it.entities }.distinct()
        val retrievalResult = agent.retrieve(message, candidates, queryEntities, now)
        var ranked = retrievalResult.ranked

        // v11 signal fusion
        ranked = applyV11Signals(ranked, now)

        // v12: Pin protection (after all decay/boost signals)
        if (enablePinProtection) {
            ranked = applyPinProtection(ranked)
        }

        // Record access frequency for top results
        val topK = ranked.take(contextBudget)
        frequencyTracker.recordAccessBatch(topK.map { it.id })

        // Conflict resolution
        val entityGroups = ranked
            .filter { it.entry?.entities?.isNotEmpty() == true }
            .flatMap { mem -> mem.entry!!.entities.map { e -> e to mem } }
            .groupBy({ it.first }, { it.second })
        val conflicts = retriever.resolveConflicts(ranked, entityGroups)

        // Select context
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
     * v12: Build candidates using pluggable EmbeddingProvider
     */
    private fun buildCandidates(query: String, memories: List<MemoryEntry>): List<RetrievalCandidate> {
        return memories.map { entry ->
            val semanticScore = when (val provider = embeddingProvider) {
                is SemanticRuleProvider -> provider.computeSimilarity(query, entry.content)
                is CharOverlapProvider -> provider.computeSimilarity(query, entry.content)
                else -> {
                    // Real embedding provider: use cosine similarity
                    val qVec = provider.embed(query)
                    val cVec = provider.embed(entry.content)
                    SimilarityCalculator.cosine(qVec, cVec)
                }
            }
            RetrievalCandidate(
                id = entry.id,
                content = entry.content,
                semanticScore = semanticScore,
                entry = entry
            )
        }
    }

    /**
     * v11 signal fusion: validity × frequency × provenance
     */
    private fun applyV11Signals(ranked: List<RankedMemory>, queryTime: Long): List<RankedMemory> {
        return ranked.map { mem ->
            var adjustedScore = mem.score

            if (enableValidityFilter && mem.entry != null) {
                val validity = validityCache[mem.id]
                    ?: validityWindow.inferValidity(mem.content, mem.entry.entities)
                val validityScore = validityWindow.computeValidityScore(
                    mem.entry.timestamp, queryTime, validity
                )
                adjustedScore *= validityScore
            }

            if (enableFrequencyBoost) {
                adjustedScore *= frequencyTracker.computeBoost(mem.id)
            }

            if (enableProvenanceSignal) {
                adjustedScore *= provenanceTracker.computeProvenanceSignal(mem.id)
            }

            mem.copy(score = adjustedScore)
        }.sortedByDescending { it.score }
    }

    /**
     * v12: Apply pin protection — pinned memories get floor score
     */
    private fun applyPinProtection(ranked: List<RankedMemory>): List<RankedMemory> {
        return ranked.map { mem ->
            val protectedScore = pinnedGuard.applyPinProtection(mem.id, mem.score)
            mem.copy(score = protectedScore)
        }.sortedByDescending { it.score }
    }

    /**
     * v12: Event-aware context expansion
     * Instead of fixed session window, expand by event boundary
     */
    fun expandWithEventBoundary(
        nuclei: List<RankedMemory>,
        userId: String
    ): ExpandedContext {
        if (!enableEventBoundary || nuclei.isEmpty()) {
            return contextExpander.expand(nuclei, store, userId)
        }

        val allMemories = store.getAll(userId).sortedBy { it.timestamp }
        val events = eventBoundaryDetector.segmentIntoEvents(allMemories)

        // For each nucleus, find its event and include all event members
        val eventMemories = mutableListOf<ContextEntry>()
        val seen = mutableSetOf<String>()

        for (nucleus in nuclei) {
            val entry = nucleus.entry ?: continue
            if (entry.id in seen) continue

            val event = eventBoundaryDetector.findEventForNucleus(entry, allMemories)

            for (mem in event.memories) {
                if (mem.id in seen) continue
                seen.add(mem.id)

                val role = if (mem.id == entry.id) ContextRole.NUCLEUS else ContextRole.SESSION_NEIGHBOR
                val score = if (mem.id == entry.id) nucleus.score else nucleus.score * 0.7
                eventMemories.add(ContextEntry(
                    memory = mem, role = role,
                    score = score, distanceFromNucleus = 0
                ))
            }
        }

        // Budget enforcement
        val sorted = eventMemories.sortedByDescending { it.score }.take(12)
        return ExpandedContext(
            entries = sorted,
            stats = ExpansionStats(
                nucleiCount = nuclei.size,
                sessionNeighbors = sorted.count { it.role == ContextRole.SESSION_NEIGHBOR },
                totalBeforeTruncation = eventMemories.size,
                totalAfterTruncation = sorted.size,
                truncated = eventMemories.size > 12
            )
        )
    }

    private fun annotateMemories(conflicts: ConflictResolution, budget: Int): List<RankedMemory> {
        val result = mutableListOf<RankedMemory>()
        result.addAll(conflicts.current.take(budget))
        val remaining = budget - result.size
        if (remaining > 0) result.addAll(conflicts.historical.take(remaining))
        return result
    }

    private fun formatContext(mem: RankedMemory, conflicts: ConflictResolution): String {
        val isHistorical = conflicts.historical.any { it.id == mem.id }
        val isPinned = pinnedGuard.isPinned(mem.id)
        val prefix = buildString {
            if (isHistorical) append("[历史] ")
            if (isPinned) append("[📌] ")
        }
        return "$prefix${mem.content}"
    }

    fun explainMemory(memoryId: String): String? {
        val provenance = provenanceTracker.getProvenance(memoryId) ?: return null
        val pinInfo = if (pinnedGuard.isPinned(memoryId)) {
            " [📌 pinned: ${pinnedGuard.getPinReason(memoryId)}]"
        } else ""
        return "来自 session ${provenance.sessionId}: \"${provenance.originalMessage}\"$pinInfo"
    }

    fun getStats(userId: String): EngineStatsV12 {
        return EngineStatsV12(
            totalMemories = store.count(userId),
            provenanceStats = provenanceTracker.getStats(),
            frequencyStats = frequencyTracker.getStats(),
            pinStats = pinnedGuard.getStats(),
            turnsProcessed = turnCounter,
            embeddingProvider = embeddingProvider.name
        )
    }
}

data class EngineStatsV12(
    val totalMemories: Int,
    val provenanceStats: ProvenanceStats,
    val frequencyStats: FrequencyStats,
    val pinStats: PinStats,
    val turnsProcessed: Int,
    val embeddingProvider: String
)
