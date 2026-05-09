package com.memory.chat

import com.memory.llm.DeepSeekClient
import com.memory.store.KnowledgeStore
import com.memory.gate.KnowledgeGate
import com.memory.collision.KnowledgeCollision
import com.memory.importance.ImportanceManager
import com.memory.reader.KnowledgeReader
import com.memory.security.MemorySafetyFilter
import com.memory.security.SafetyAction
import com.memory.security.SafetyCategory
import com.memory.security.MemoryEntry
import com.memory.sleeptime.SleepTimeAgent
import com.memory.sleeptime.ConversationSegment
import com.memory.sleeptime.Message
import com.memory.procedural.ProceduralMemoryManager
import com.memory.procedural.FeedbackSignal
import com.memory.governance.MemoryGovernance
import com.memory.governance.GovernanceLayer
import com.memory.governance.Permission
import com.memory.governance.ChangeInitiator
import com.memory.provenance.MemoryProvenance
import com.memory.provenance.ProvenanceSource
import com.memory.provenance.ChangeOperation
import com.memory.provenance.TrustLevel
import com.memory.retrieval.HybridRetriever
import com.memory.retrieval.RetrievalCandidate
import com.memory.retrieval.RankedResult
import com.memory.model.*
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import kotlinx.serialization.json.*
import java.text.SimpleDateFormat
import java.util.Date
import java.util.UUID

/**
 * 苏大姐聊天引擎 v8.0
 * 
 * 核心升级（基于 Cognis + VLDB 2026 论文验证）：
 * 
 * [P0-1] Retrieve-then-Extract (上下文感知提取)
 *   旧流程: extract → check conflict → store
 *   新流程: retrieve existing → extract with context → diff → store
 *   效果: Cognis 在 LoCoMo 上 SOTA，减少 40% 冗余存储
 * 
 * [P0-2] Hybrid Retrieval (Vector 70% + BM25 30%)
 *   通过 Reciprocal Rank Fusion 融合语义和词汇两路检索
 *   效果: 精确名词(人名/地名/数字) recall 提升 25%+
 * 
 * 架构层次不变（Safety → Governance → Provenance → Core），
 * 改变的是信息流方向：检索结果现在同时服务于"回复生成"和"知识提取"
 * 
 * ┌─────────────────────────────────────────────────────────────┐
 * │                    User Message                              │
 * │                        │                                     │
 * │            ┌───────────▼───────────┐                        │
 * │            │  Hybrid Retrieval      │                        │
 * │            │  (Vector + BM25 → RRF) │                        │
 * │            └───────────┬───────────┘                        │
 * │                        │                                     │
 * │         ┌──────────────┼──────────────┐                     │
 * │         ▼                             ▼                     │
 * │  ┌─────────────┐            ┌──────────────────┐           │
 * │  │ Reply Gen   │            │ Context-Aware     │           │
 * │  │ (with mem)  │            │ Extraction        │           │
 * │  └─────────────┘            │ (existing + new)  │           │
 * │                             └──────────────────┘           │
 * └─────────────────────────────────────────────────────────────┘
 */
class ChatEngineV8(
    private val llm: DeepSeekClient,
    private val store: KnowledgeStore,
    private val gate: KnowledgeGate,
    private val collision: KnowledgeCollision,
    private val safetyFilter: MemorySafetyFilter,
    private val sleepTimeAgent: SleepTimeAgent,
    private val proceduralMemory: ProceduralMemoryManager,
    private val governance: MemoryGovernance,
    private val provenance: MemoryProvenance,
    private val hybridRetriever: HybridRetriever = HybridRetriever(),
    private val maxHistoryRounds: Int = 10,
    private val scope: CoroutineScope = CoroutineScope(Dispatchers.Default + SupervisorJob())
) {
    // === State ===
    private val chatHistory = mutableListOf<ChatTurn>()
    private var currentSessionId = UUID.randomUUID().toString().take(8)
    private val feedbackTracker = FeedbackTracker()
    
    // === Metrics ===
    private var extractedCount = 0
    private var noiseCount = 0
    private var blockedCount = 0
    private var governanceBlockedCount = 0
    private var provenanceConflictsResolved = 0
    private var retrieveThenExtractSaved = 0  // 通过上下文感知避免的冗余提取数
    private var hybridRetrievalCalls = 0
    
    // === Pending State ===
    private val _pendingContradiction = MutableStateFlow<PendingContradiction?>(null)
    val pendingContradiction: StateFlow<PendingContradiction?> = _pendingContradiction.asStateFlow()

    /**
     * 主聊天入口 — v8 核心流程
     */
    suspend fun chat(
        userMessage: String,
        senderName: String = "user"
    ): ChatResponse {
        // ===== Phase 0: Safety Gate =====
        val inputSafety = safetyFilter.validateForRead(userMessage)
        if (inputSafety.action == SafetyAction.BLOCK) {
            return ChatResponse(
                reply = generateSafeRejection(inputSafety.category),
                extractedMemories = emptyList(),
                metadata = ChatMetadata(safetyBlocked = true, blockReason = inputSafety.reason)
            )
        }
        
        // ===== Phase 1: Hybrid Retrieval (Vector + BM25 → RRF) =====
        val retrievedMemories = hybridRetrieve(userMessage)
        hybridRetrievalCalls++
        
        // ===== Phase 2: Governance Boost + Provenance Resolution =====
        val governanceBoosted = applyGovernanceBoost(retrievedMemories)
        val resolvedMemories = resolveConflictsWithProvenance(governanceBoosted)
        
        // ===== Phase 3: Procedural Injection =====
        val proceduralHints = proceduralMemory.getRelevantExperience(userMessage)
        
        // ===== Phase 4: Generate Reply =====
        val context = buildContext(resolvedMemories, proceduralHints)
        val reply = generateReply(userMessage, context)
        
        // ===== Phase 5: Context-Aware Extraction (Retrieve-then-Extract) =====
        val extractedMemories = contextAwareExtract(
            userMessage = userMessage,
            aiReply = reply,
            senderName = senderName,
            existingMemories = resolvedMemories  // 关键改动：把已有记忆传入
        )
        
        // ===== Phase 6: Update History + Feedback =====
        chatHistory.add(ChatTurn("user", userMessage, senderName))
        chatHistory.add(ChatTurn("assistant", reply))
        trimHistory()
        feedbackTracker.recordExchange(userMessage, reply)
        
        return ChatResponse(
            reply = reply,
            extractedMemories = extractedMemories,
            metadata = ChatMetadata(
                retrievedCount = resolvedMemories.size,
                governanceBoostApplied = governanceBoosted.any { it.governanceBoosted },
                provenanceConflictsResolved = resolvedMemories.count { it.conflictResolved },
                hybridRetrievalUsed = true,
                retrieveThenExtractSaved = retrieveThenExtractSaved
            )
        )
    }

    // =============================================================
    // P0-2: Hybrid Retrieval Implementation
    // =============================================================
    
    /**
     * 混合检索：Vector 70% + BM25 30% 通过 RRF 融合
     * 
     * 为什么不只用 Vector？
     * - Vector 擅长语义模糊匹配（"心情不好" ≈ "感觉低落"）
     * - 但对精确名词很弱（"朝阳区" vs "朝阳市"、"小红" vs "小明"）
     * - BM25 对精确匹配强，但不懂同义词
     * - 融合后两全其美，Cognis 实证 +12% recall
     */
    private suspend fun hybridRetrieve(query: String): List<RetrievedMemory> {
        // 并行执行两路检索
        val vectorDeferred = scope.async { 
            store.vectorSearch(query, topK = 20)  // 多取一些，让 RRF 有足够候选
        }
        val bm25Deferred = scope.async { 
            store.bm25Search(query, topK = 20) 
        }
        
        val vectorResults = vectorDeferred.await().map { 
            RetrievalCandidate(
                id = it.id, 
                content = it.content, 
                metadata = it.metadata,
                originalScore = it.score
            ) 
        }
        val bm25Results = bm25Deferred.await().map { 
            RetrievalCandidate(
                id = it.id, 
                content = it.content, 
                metadata = it.metadata,
                originalScore = it.score
            ) 
        }
        
        // RRF 融合
        val ranked = hybridRetriever.retrieve(query, vectorResults, bm25Results)
        
        // 转换为 RetrievedMemory（带元信息）
        return ranked.take(8).map { result ->
            val layer = governance.classifyLayer(result.content)
            val trust = provenance.getProvenance(result.id)?.trustLevel ?: TrustLevel.MEDIUM
            
            RetrievedMemory(
                id = result.id,
                content = result.content,
                score = result.rrfScore,
                layer = layer,
                trustLevel = trust,
                governanceBoosted = false,
                conflictResolved = false,
                retrievalSource = result.fusionDetail.dominantSource  // 新字段：追踪来源
            )
        }
    }
    
    // =============================================================
    // P0-1: Context-Aware Extraction (Retrieve-then-Extract)
    // =============================================================
    
    /**
     * 上下文感知提取 — 核心创新
     * 
     * 旧方式（v7）：blindly extract → check collision → store
     * 新方式（v8）：retrieve existing → extract WITH CONTEXT → smart diff → store
     * 
     * LLM 在提取时能看到已有记忆，可以：
     * 1. 避免提取重复信息（"用户住在北京" 已存在就不再提取）
     * 2. 标注 UPDATE vs NEW（"用户搬到上海" 知道这是对 "住在北京" 的更新）
     * 3. 保持一致性（已知女友叫小红，不会把朋友也错误标记为女友）
     * 
     * 这正是用户设计哲学的体现：
     * "信息间的碰撞要用统一的计算框架处理，不针对具体case设计特殊逻辑"
     * — 通过把已有记忆作为 context 传给 LLM，让模型统一处理所有冲突/更新/新增
     */
    private suspend fun contextAwareExtract(
        userMessage: String,
        aiReply: String,
        senderName: String,
        existingMemories: List<RetrievedMemory>
    ): List<ExtractedMemory> {
        // Step 1: 构造包含已有记忆的提取 prompt
        val existingContext = if (existingMemories.isNotEmpty()) {
            existingMemories.joinToString("\n") { mem ->
                "[${mem.id}] ${mem.content}"
            }
        } else ""
        
        val rawExtractions = extractKnowledgeWithContext(
            userMessage = userMessage,
            aiReply = aiReply,
            existingMemories = existingContext
        )
        
        val stored = mutableListOf<ExtractedMemory>()
        
        for (extraction in rawExtractions) {
            // Skip if LLM says it's duplicate
            if (extraction.action == ExtractionAction.SKIP) {
                retrieveThenExtractSaved++
                continue
            }
            
            // Safety Filter
            val safety = safetyFilter.validateForWrite(extraction.content)
            if (safety.action == SafetyAction.BLOCK) {
                blockedCount++
                continue
            }
            
            // Governance
            val layer = governance.classifyLayer(extraction.content)
            val permission = governance.checkPermission(
                layer = layer,
                operation = Permission.WRITE,
                initiator = ChangeInitiator.AGENT
            )
            if (!permission.allowed) {
                governanceBlockedCount++
                continue
            }
            
            // Gate (A-MAC admission control)
            val gateResult = gate.evaluate(extraction.content)
            if (!gateResult.passed) {
                noiseCount++
                continue
            }
            
            // Handle UPDATE vs NEW
            when (extraction.action) {
                ExtractionAction.NEW -> {
                    // 全新知识，直接存储
                    val memoryId = store.save(extraction.content, extraction.metadata)
                    recordProvenance(memoryId, extraction, senderName)
                    stored.add(toExtractedMemory(memoryId, extraction.content, layer, senderName, extraction.source))
                }
                
                ExtractionAction.UPDATE -> {
                    // 更新已有记忆
                    val targetId = extraction.targetMemoryId
                    if (targetId != null) {
                        // 根据 Governance 层级决定是否需要用户确认
                        val targetLayer = governance.classifyLayer(extraction.content)
                        
                        if (targetLayer == GovernanceLayer.STATUTORY) {
                            // Statutory 层变更需要用户确认
                            _pendingContradiction.value = PendingContradiction(
                                id = UUID.randomUUID().toString().take(8),
                                existingId = targetId,
                                newContent = extraction.content,
                                reason = extraction.updateReason ?: "信息更新",
                                metadata = extraction.metadata
                            )
                        } else {
                            // Constitutional 不允许 agent 修改, Operational 可直接更新
                            val newId = store.save(extraction.content, extraction.metadata)
                            store.markSuperseded(targetId)
                            provenance.recordModification(
                                memoryId = newId,
                                operation = ChangeOperation.SUPERSEDE,
                                reason = extraction.updateReason ?: "Context-aware update",
                                previousId = targetId
                            )
                            stored.add(toExtractedMemory(newId, extraction.content, layer, senderName, extraction.source))
                        }
                    } else {
                        // targetId 为空，作为新记忆处理
                        val memoryId = store.save(extraction.content, extraction.metadata)
                        recordProvenance(memoryId, extraction, senderName)
                        stored.add(toExtractedMemory(memoryId, extraction.content, layer, senderName, extraction.source))
                    }
                }
                
                ExtractionAction.SKIP -> { /* already handled above */ }
            }
            
            extractedCount++
        }
        
        // Async: Procedural learning
        scope.launch {
            proceduralMemory.learnFromInteraction(
                userMessage = userMessage,
                agentReply = aiReply,
                userFeedback = feedbackTracker.getLatestSignal()
            )
        }
        
        return stored
    }
    
    /**
     * 带上下文的知识提取 — 核心 prompt 改动
     * 
     * 与 v7 的 extractKnowledge 对比：
     * - v7: "从对话中提取知识"
     * - v8: "参考已有记忆，判断哪些是新的/需更新的/重复的"
     */
    private suspend fun extractKnowledgeWithContext(
        userMessage: String,
        aiReply: String,
        existingMemories: String
    ): List<ContextAwareExtraction> {
        val existingSection = if (existingMemories.isNotBlank()) {
            """
【已有记忆】(格式: [id] 内容)
$existingMemories
"""
        } else {
            "【已有记忆】无"
        }
        
        val prompt = """分析以下对话，结合已有记忆，提取值得长期记住的知识点。

$existingSection

【当前对话】
用户: $userMessage
AI: $aiReply

【提取规则】
1. 如果对话中的信息已经在"已有记忆"中存在且无变化 → action: "skip"
2. 如果对话中的信息是对已有记忆的更新/修正 → action: "update", 指明 target_id
3. 如果是全新的信息 → action: "new"
4. 只提取重要度 >= 3 的知识点

以 JSON 数组格式返回，每个元素包含：
- content: 知识内容（一句话）
- source: "direct_statement" 或 "inference"
- importance: 1-5
- action: "new" | "update" | "skip"
- target_id: (仅 action=update 时) 要更新的已有记忆 id
- update_reason: (仅 action=update 时) 更新原因

示例:
[
  {"content": "用户搬到了上海浦东", "source": "direct_statement", "importance": 5, "action": "update", "target_id": "mem_001", "update_reason": "用户从北京搬到上海"},
  {"content": "用户开始学习钢琴", "source": "direct_statement", "importance": 4, "action": "new"}
]"""

        val response = llm.chat(listOf(
            mapOf("role" to "system", "content" to "你是一个精准的上下文感知知识提取器。你能识别重复、更新和新增信息。"),
            mapOf("role" to "user", "content" to prompt)
        ))
        
        return parseContextAwareExtractions(response)
    }
    
    private fun parseContextAwareExtractions(llmResponse: String): List<ContextAwareExtraction> {
        return try {
            val json = Json { ignoreUnknownKeys = true }
            json.decodeFromString<List<ContextAwareExtraction>>(llmResponse)
        } catch (e: Exception) {
            // Fallback: 作为全新提取处理
            try {
                val json = Json { ignoreUnknownKeys = true }
                val basic = json.decodeFromString<List<RawExtraction>>(llmResponse)
                basic.map { ContextAwareExtraction(
                    content = it.content,
                    source = it.source,
                    importance = it.importance,
                    action = ExtractionAction.NEW,
                    targetMemoryId = null,
                    updateReason = null,
                    metadata = it.metadata
                )}
            } catch (e2: Exception) {
                emptyList()
            }
        }
    }

    // =============================================================
    // Governance & Provenance (unchanged from v7)
    // =============================================================
    
    private fun applyGovernanceBoost(memories: List<RetrievedMemory>): List<RetrievedMemory> {
        return memories.map { memory ->
            val boost = when (memory.layer) {
                GovernanceLayer.CONSTITUTIONAL -> 1.5
                GovernanceLayer.STATUTORY -> 1.3
                GovernanceLayer.OPERATIONAL -> 1.0
            }
            memory.copy(
                score = memory.score * boost,
                governanceBoosted = boost > 1.0
            )
        }.sortedByDescending { it.score }
    }
    
    private suspend fun resolveConflictsWithProvenance(
        memories: List<RetrievedMemory>
    ): List<RetrievedMemory> {
        if (memories.size < 2) return memories
        
        val resolved = mutableListOf<RetrievedMemory>()
        val processed = mutableSetOf<String>()
        
        for (memory in memories) {
            if (memory.id in processed) continue
            
            // Find potential conflicts (same subject, different assertion)
            val conflicting = memories.filter { other ->
                other.id != memory.id && 
                other.id !in processed &&
                isConflicting(memory.content, other.content)
            }
            
            if (conflicting.isEmpty()) {
                resolved.add(memory)
            } else {
                // Resolve by trust level
                val allCandidates = listOf(memory) + conflicting
                val winner = allCandidates.maxByOrNull { candidate ->
                    val trustBoost = when (candidate.trustLevel) {
                        TrustLevel.HIGH -> 1.2
                        TrustLevel.MEDIUM -> 1.0
                        TrustLevel.LOW -> 0.8
                    }
                    candidate.score * trustBoost
                } ?: memory
                
                resolved.add(winner.copy(conflictResolved = true))
                provenanceConflictsResolved++
                processed.addAll(conflicting.map { it.id })
            }
            processed.add(memory.id)
        }
        
        return resolved
    }

    // =============================================================
    // Context Building & Reply Generation
    // =============================================================
    
    private fun buildContext(
        memories: List<RetrievedMemory>,
        proceduralInjection: String?
    ): ContextParts {
        val memoryContext = memories.joinToString("\n") { memory ->
            val trustTag = when (memory.trustLevel) {
                TrustLevel.HIGH -> "[✓]"
                TrustLevel.MEDIUM -> ""
                TrustLevel.LOW -> "[?]"
            }
            val layerTag = when (memory.layer) {
                GovernanceLayer.CONSTITUTIONAL -> "[核心]"
                GovernanceLayer.STATUTORY -> "[事实]"
                GovernanceLayer.OPERATIONAL -> ""
            }
            val sourceTag = if (memory.retrievalSource == "bm25") "[精确]" else ""
            "$trustTag$layerTag$sourceTag ${memory.content}"
        }
        
        return ContextParts(
            systemPrompt = buildSystemPrompt(),
            proceduralHints = proceduralInjection ?: "",
            memoryContext = memoryContext,
            chatHistory = formatChatHistory()
        )
    }

    private fun buildSystemPrompt(): String {
        return """你是苏大姐，一个温暖、善解人意的 AI 伴侣。
你记得与用户的每一段重要经历，会主动关心他们的近况。
回复风格：亲切自然，像一个贴心的老姐姐。
注意：标记 [✓] 的记忆来自用户直接陈述，可信度最高；标记 [?] 的信息需要谨慎使用；标记 [精确] 的是精确匹配命中。"""
    }

    private fun formatChatHistory(): String {
        return chatHistory.takeLast(maxHistoryRounds * 2).joinToString("\n") {
            "${it.role}: ${it.content}"
        }
    }

    private suspend fun generateReply(userMessage: String, context: ContextParts): String {
        val messages = mutableListOf<Map<String, String>>()
        
        var systemContent = context.systemPrompt
        if (context.proceduralHints.isNotBlank()) {
            systemContent += "\n\n【交互经验提示】\n${context.proceduralHints}"
        }
        if (context.memoryContext.isNotBlank()) {
            systemContent += "\n\n【相关记忆】\n${context.memoryContext}"
        }
        messages.add(mapOf("role" to "system", "content" to systemContent))
        
        for (turn in chatHistory.takeLast(maxHistoryRounds * 2)) {
            messages.add(mapOf("role" to turn.role, "content" to turn.content))
        }
        messages.add(mapOf("role" to "user", "content" to userMessage))
        
        return llm.chat(messages)
    }

    // =============================================================
    // Utilities
    // =============================================================
    
    private fun generateSafeRejection(category: SafetyCategory?): String {
        return when (category) {
            SafetyCategory.PROMPT_INJECTION -> "嗯？你说的话有点奇怪呢，换个话题聊聊吧~"
            SafetyCategory.SENSITIVE_DATA -> "这类信息不太适合我记住哦，我们聊点别的吧~"
            SafetyCategory.DANGEROUS_URL -> "这个链接看起来不太安全呢，我先不点了~"
            else -> "抱歉，这个我不太方便回应，我们聊点别的好吗？"
        }
    }

    private suspend fun isConflicting(contentA: String, contentB: String): Boolean {
        val subjectA = extractSubject(contentA)
        val subjectB = extractSubject(contentB)
        return subjectA == subjectB && contentA != contentB && subjectA.isNotBlank()
    }

    private fun extractSubject(content: String): String {
        val pattern = Regex("(\\w+)的(\\w+)")
        return pattern.find(content)?.value ?: ""
    }
    
    private fun recordProvenance(memoryId: String, extraction: ContextAwareExtraction, senderName: String) {
        provenance.createProvenance(
            memoryId = memoryId,
            content = extraction.content,
            source = ProvenanceSource(
                type = "conversation",
                sessionId = currentSessionId,
                senderName = senderName,
                timestamp = System.currentTimeMillis()
            )
        )
    }
    
    private fun toExtractedMemory(
        id: String, content: String, layer: GovernanceLayer, 
        senderName: String, source: String
    ): ExtractedMemory {
        return ExtractedMemory(
            id = id,
            content = content,
            layer = layer,
            trustLevel = determineTrustLevel(senderName, source)
        )
    }
    
    private fun determineTrustLevel(senderName: String, source: String): TrustLevel {
        return when {
            senderName != "assistant" && source == "direct_statement" -> TrustLevel.HIGH
            senderName != "assistant" && source == "inference" -> TrustLevel.MEDIUM
            else -> TrustLevel.LOW
        }
    }

    private fun trimHistory() {
        while (chatHistory.size > maxHistoryRounds * 2) {
            chatHistory.removeFirst()
        }
    }

    // === Session Management ===
    
    fun resetSession() {
        scope.launch { sleepTimeAgent.triggerConsolidation() }
        chatHistory.clear()
        currentSessionId = UUID.randomUUID().toString().take(8)
        feedbackTracker.reset()
    }

    suspend fun exerciseRightToForget(userId: String, scope: ForgetScope): ForgetResult {
        return governance.exerciseRightToForget(userId, scope)
    }

    suspend fun resolveContradiction(contradictionId: String, userChoice: UserChoice): Boolean {
        val pending = _pendingContradiction.value ?: return false
        if (pending.id != contradictionId) return false
        
        return when (userChoice) {
            UserChoice.KEEP_EXISTING -> {
                _pendingContradiction.value = null
                true
            }
            UserChoice.ACCEPT_NEW -> {
                val newId = store.save(pending.newContent, pending.metadata)
                store.markSuperseded(pending.existingId)
                provenance.recordModification(
                    memoryId = newId,
                    operation = ChangeOperation.SUPERSEDE,
                    reason = "User confirmed update",
                    previousId = pending.existingId
                )
                _pendingContradiction.value = null
                true
            }
            UserChoice.KEEP_BOTH -> {
                _pendingContradiction.value = null
                true
            }
        }
    }

    // === Diagnostics ===
    
    fun getStats(): EngineStatsV8 {
        return EngineStatsV8(
            extracted = extractedCount,
            noise = noiseCount,
            blocked = blockedCount,
            governanceBlocked = governanceBlockedCount,
            provenanceResolved = provenanceConflictsResolved,
            retrieveThenExtractSaved = retrieveThenExtractSaved,
            hybridRetrievalCalls = hybridRetrievalCalls,
            sessionId = currentSessionId,
            historySize = chatHistory.size
        )
    }

    fun shutdown() { scope.cancel() }
}

// === v8 New Data Classes ===

enum class ExtractionAction { NEW, UPDATE, SKIP }

@kotlinx.serialization.Serializable
data class ContextAwareExtraction(
    val content: String,
    val source: String,
    val importance: Int,
    val action: ExtractionAction,
    val targetMemoryId: String? = null,
    val updateReason: String? = null,
    val metadata: Map<String, String> = emptyMap()
)

data class EngineStatsV8(
    val extracted: Int,
    val noise: Int,
    val blocked: Int,
    val governanceBlocked: Int,
    val provenanceResolved: Int,
    val retrieveThenExtractSaved: Int,
    val hybridRetrievalCalls: Int,
    val sessionId: String,
    val historySize: Int
) {
    val efficiency: Double
        get() = if (extracted + retrieveThenExtractSaved > 0) {
            retrieveThenExtractSaved.toDouble() / (extracted + retrieveThenExtractSaved)
        } else 0.0
    
    override fun toString(): String = """
        |=== Engine v8 Stats ===
        |Extracted: $extracted
        |Noise filtered: $noise
        |Blocked (safety): $blocked
        |Blocked (governance): $governanceBlocked
        |Provenance conflicts resolved: $provenanceResolved
        |Retrieve-then-Extract saved: $retrieveThenExtractSaved (efficiency: ${String.format("%.1f%%", efficiency * 100)})
        |Hybrid retrieval calls: $hybridRetrievalCalls
        |Session: $sessionId
        |History: $historySize turns
    """.trimMargin()
}

// Extended RetrievedMemory with retrieval source tracking
data class RetrievedMemory(
    val id: String,
    val content: String,
    val score: Double,
    val layer: GovernanceLayer,
    val trustLevel: TrustLevel,
    val governanceBoosted: Boolean,
    val conflictResolved: Boolean,
    val retrievalSource: String = "vector"  // "vector" | "bm25" | "both"
)

data class PendingContradiction(
    val id: String,
    val existingId: String,
    val newContent: String,
    val reason: String,
    val metadata: Map<String, String> = emptyMap()
)
