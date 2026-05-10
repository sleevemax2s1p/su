package com.memory.chat

import com.memory.context.AdaptiveContextSelector
import com.memory.context.MemoryCandidate
import com.memory.context.QueryComplexity
import com.memory.context.ConversationContext
import com.memory.context.SelectionResult
import com.memory.policy.*
import com.memory.forgetting.SelectiveForgetting
import com.memory.graph.EntityRelationGraph
import com.memory.graph.GraphAugmentedRetrieval
import com.memory.governance.GovernanceLayer
import com.memory.provenance.TrustLevel
import com.memory.retrieval.HybridRetriever
import com.memory.retrieval.RankedResult
import kotlinx.coroutines.*
import java.util.UUID

/**
 * 苏大姐聊天引擎 v9.0
 * 
 * v8 → v9 升级：
 * 
 * [1] AdaptiveContextSelector 替换固定 take(8)
 *     - 动态 K 基于 query complexity + score gap + token budget
 *     - 51% 上下文压缩 (MemAct 验证)
 *     - Sandwich ordering for Lost-in-the-Middle
 * 
 * [2] MemoryPolicy 统一决策接口
 *     - 所有 store/skip/update/delete 决策通过 Policy.decide()
 *     - 当前: RuleBasedPolicy (显式规则)
 *     - 未来: NeuralPolicy (GRPO-trained)
 *     - A/B 测试支持
 * 
 * [3] SelectiveForgetting 集成
 *     - 时间衰减 + 访问增强 + 矛盾/取代惩罚
 *     - Governance-aware: Constitutional 永不遗忘
 *     - 定期 consolidation 清理低保留率记忆
 * 
 * [4] EntityRelationGraph 增强
 *     - 多跳关系推理
 *     - Graph-boosted retrieval (+15% boost)
 *     - 冲突检测 (same subject + same relation + different target)
 * 
 * 架构演进方向 (from AgeMem/AtomMem/MemAct 2025-2026):
 * "规则策略 → 学习策略" 是 Memory System 的工业界共识。
 * v9 的 Policy 接口为这个演进铺路：
 * - behavior cloning warm-start (从 RuleBasedPolicy traces)
 * - step-wise GRPO 训练 (单步 reward → 全局 reward)
 * - progressive RL (单 action → multi-action → full episode)
 */
class ChatEngineV9(
    // 继承 v8 的所有依赖
    private val v8Engine: ChatEngineV8,
    
    // v9 新模块
    private val contextSelector: AdaptiveContextSelector = AdaptiveContextSelector(),
    private val policy: MemoryPolicy = PolicyFactory.getPolicy(),
    private val forgetting: SelectiveForgetting = SelectiveForgetting(),
    private val entityGraph: EntityRelationGraph = EntityRelationGraph(),
    private val graphRetrieval: GraphAugmentedRetrieval = GraphAugmentedRetrieval(
        HybridRetriever(), entityGraph
    )
) {
    
    // === 统计 ===
    private var totalTurns = 0
    private var totalContextCompression = 0.0
    private var policyDecisionCount = 0
    private var forgottenMemories = 0
    private var graphHits = 0
    
    /**
     * 处理一轮对话 (替代 v8 的 processMessage)
     * 
     * 信息流：
     * 1. Query → HybridRetrieval + GraphAugmented → candidates
     * 2. candidates → AdaptiveContextSelector → selected memories
     * 3. selected → Reply generation
     * 4. Message → Policy.decide(EXTRACTION) → store/update/skip
     * 5. Periodic → Policy.decide(MAINTENANCE) → summarize/forget
     */
    suspend fun processMessage(
        userId: String,
        message: String,
        conversationHistory: List<String> = emptyList()
    ): ProcessResult {
        totalTurns++
        
        // === Phase 1: Retrieval (Graph-Augmented Hybrid) ===
        val queryComplexity = AdaptiveContextSelector.assessQueryComplexity(message)
        
        val retrievalResults = graphRetrieval.retrieve(
            query = message,
            userId = userId,
            enableGraph = shouldEnableGraph(message, queryComplexity)
        )
        
        // === Phase 2: Adaptive Context Selection ===
        val candidates = retrievalResults.map { it.toMemoryCandidate() }
        val conversationContext = ConversationContext(
            recentTopics = extractRecentTopics(conversationHistory),
            turnCount = conversationHistory.size
        )
        
        val selectionResult = contextSelector.select(
            candidates = candidates,
            queryComplexity = queryComplexity,
            conversationContext = conversationContext
        )
        
        totalContextCompression += selectionResult.compressionRatio
        
        // === Phase 3: Policy Decision — Injection ===
        val injectionState = PolicyState(
            currentMessage = message,
            turnIndex = totalTurns,
            conversationLength = conversationHistory.size + 1,
            retrievedMemories = candidates.map { it.toRetrievedMemoryState() },
            queryComplexity = queryComplexity,
            tokenBudgetRemaining = 1200 - selectionResult.estimatedTokens
        )
        
        val injectionDecisions = policy.decide(injectionState, DecisionPhase.INJECTION)
        policyDecisionCount += injectionDecisions.size
        
        // === Phase 4: Generate Reply (delegate to v8) ===
        val contextMemories = selectionResult.selected.map { it.content }
        val reply = v8Engine.generateReplyWithContext(
            message = message,
            memories = contextMemories,
            history = conversationHistory
        )
        
        // === Phase 5: Policy Decision — Extraction ===
        val extractedFacts = v8Engine.extractFacts(message, contextMemories)
        
        val extractionState = PolicyState(
            currentMessage = message,
            turnIndex = totalTurns,
            conversationLength = conversationHistory.size + 1,
            retrievedMemories = candidates.map { it.toRetrievedMemoryState() },
            extractedFacts = extractedFacts.map { 
                ExtractedFactState(
                    content = it.content,
                    confidence = it.confidence,
                    matchingMemoryId = it.matchingId,
                    matchScore = it.matchScore
                )
            },
            queryComplexity = queryComplexity,
            existingMemoryCount = v8Engine.getMemoryCount(userId)
        )
        
        val extractionDecisions = policy.decide(extractionState, DecisionPhase.EXTRACTION)
        policyDecisionCount += extractionDecisions.size
        
        // Execute extraction decisions
        for (decision in extractionDecisions) {
            when (decision.action) {
                MemoryAction.STORE -> {
                    val content = decision.targets.firstOrNull() ?: continue
                    v8Engine.storeMemory(userId, content)
                    // Register with forgetting system
                    forgetting.registerMemory(
                        id = UUID.randomUUID().toString(),
                        layer = GovernanceLayer.OPERATIONAL,
                        trustLevel = TrustLevel.MEDIUM
                    )
                    // Register entities in graph
                    entityGraph.batchIngest(content)
                }
                MemoryAction.UPDATE -> {
                    val targetId = decision.targets.getOrNull(0) ?: continue
                    val newContent = decision.targets.getOrNull(1) ?: continue
                    v8Engine.updateMemory(userId, targetId, newContent)
                    // Mark old as superseded
                    forgetting.markSuperseded(targetId, UUID.randomUUID().toString())
                }
                MemoryAction.DELETE -> {
                    for (targetId in decision.targets) {
                        v8Engine.deleteMemory(userId, targetId)
                        forgottenMemories++
                    }
                }
                else -> { /* SKIP — no action */ }
            }
        }
        
        // === Phase 6: Periodic Maintenance ===
        if (totalTurns % 10 == 0) {
            runMaintenance(userId)
        }
        
        // Record access for forgetting system
        for (memory in selectionResult.selected) {
            forgetting.recordAccess(memory.id)
        }
        
        return ProcessResult(
            reply = reply,
            memoriesUsed = selectionResult.selectedCount,
            totalCandidates = selectionResult.totalCandidates,
            compressionRatio = selectionResult.compressionRatio,
            queryComplexity = queryComplexity,
            extractionActions = extractionDecisions.map { it.action },
            graphEnabled = shouldEnableGraph(message, queryComplexity)
        )
    }
    
    /**
     * 定期维护：遗忘 + 摘要 + 图清理
     */
    private suspend fun runMaintenance(userId: String) {
        val maintenanceState = PolicyState(
            currentMessage = "",
            turnIndex = totalTurns,
            conversationLength = totalTurns,
            existingMemoryCount = v8Engine.getMemoryCount(userId),
            retrievedMemories = getStaleMemories(userId)
        )
        
        val decisions = policy.decide(maintenanceState, DecisionPhase.MAINTENANCE)
        
        for (decision in decisions) {
            when (decision.action) {
                MemoryAction.SUMMARIZE -> {
                    // Trigger consolidation
                    val consolidated = forgetting.runConsolidation()
                    forgottenMemories += consolidated.size
                }
                MemoryAction.DELETE -> {
                    for (id in decision.targets) {
                        v8Engine.deleteMemory(userId, id)
                        forgottenMemories++
                    }
                }
                else -> {}
            }
        }
    }
    
    private fun shouldEnableGraph(query: String, complexity: QueryComplexity): Boolean {
        return complexity == QueryComplexity.MULTI_HOP || 
               graphRetrieval.shouldEnableGraph(query)
    }
    
    private fun extractRecentTopics(history: List<String>): List<String> {
        // 从最近 3 轮提取关键词
        return history.takeLast(3).flatMap { msg ->
            // 简单提取：4 字以上的名词短语
            msg.chunked(4).filter { it.length >= 2 }
        }.distinct().take(5)
    }
    
    private fun getStaleMemories(userId: String): List<RetrievedMemoryState> {
        // Delegate to forgetting system for stale detection
        return emptyList() // TODO: wire to real store
    }
    
    // === Stats ===
    
    fun getStats(): EngineStats = EngineStats(
        totalTurns = totalTurns,
        avgCompression = if (totalTurns > 0) totalContextCompression / totalTurns else 0.0,
        policyDecisions = policyDecisionCount,
        forgottenMemories = forgottenMemories,
        graphHits = graphHits,
        policyStats = policy.getStats()
    )
}

// === Data Classes ===

data class ProcessResult(
    val reply: String,
    val memoriesUsed: Int,
    val totalCandidates: Int,
    val compressionRatio: Double,
    val queryComplexity: QueryComplexity,
    val extractionActions: List<MemoryAction>,
    val graphEnabled: Boolean
)

data class EngineStats(
    val totalTurns: Int,
    val avgCompression: Double,
    val policyDecisions: Int,
    val forgottenMemories: Int,
    val graphHits: Int,
    val policyStats: PolicyStats
)

// === Extension Functions ===

private fun RankedResult.toMemoryCandidate(): MemoryCandidate = MemoryCandidate(
    id = this.id,
    content = this.content,
    score = this.score,
    layer = this.governanceLayer ?: GovernanceLayer.OPERATIONAL,
    trustLevel = this.trustLevel ?: TrustLevel.MEDIUM,
    retrievalSource = this.source ?: "hybrid"
)

private fun MemoryCandidate.toRetrievedMemoryState(): RetrievedMemoryState = RetrievedMemoryState(
    id = this.id,
    content = this.content,
    score = this.score,
    layer = this.layer,
    trustLevel = this.trustLevel
)
