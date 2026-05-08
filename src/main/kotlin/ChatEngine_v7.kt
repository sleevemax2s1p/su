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
import com.memory.model.*
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import kotlinx.serialization.json.*
import java.text.SimpleDateFormat
import java.util.Date
import java.util.UUID

/**
 * 苏大姐聊天引擎 v7.0
 * 
 * 在 v6.0 基础上集成 Governance + Provenance 层：
 * 
 * 新架构层次（由外到内）：
 * ┌─────────────────────────────────────────────────┐
 * │ Safety Filter (输入/输出安全闸门)                 │
 * │  ┌───────────────────────────────────────────┐  │
 * │  │ Governance (三层权限治理)                    │  │
 * │  │  ┌─────────────────────────────────────┐  │  │
 * │  │  │ Provenance (来源追踪 + 信任评分)      │  │  │
 * │  │  │  ┌─────────────────────────────┐    │  │  │
 * │  │  │  │ Core Memory Engine          │    │  │  │
 * │  │  │  │ (Store + Gate + Collision)   │    │  │  │
 * │  │  │  └─────────────────────────────┘    │  │  │
 * │  │  └─────────────────────────────────────┘  │  │
 * │  └───────────────────────────────────────────┘  │
 * └─────────────────────────────────────────────────┘
 * 
 * v7 新增能力：
 * - 记忆治理（Constitutional/Statutory/Operational 三层分类）
 * - 来源追踪（每条记忆携带来源、信任等级、修改历史）
 * - 矛盾解决基于信任比较（高信任记忆优先）
 * - 治理层级加权检索（Statutory 事实性记忆 boost）
 * - Right-to-Forget 用户主权
 * 
 * 设计原则（Context Engineering 五准则 + Context Constitution 五原则）：
 * 1. Relevance: 检索结果与当前查询高度相关
 * 2. Sufficiency: 上下文足以回答问题
 * 3. Isolation: 不同来源记忆互不干扰
 * 4. Economy: 最小化 token 消耗
 * 5. Provenance: 每条信息可追溯来源和信任度
 */
class ChatEngineV7(
    private val llm: DeepSeekClient,
    private val store: KnowledgeStore,
    private val gate: KnowledgeGate,
    private val collision: KnowledgeCollision,
    private val importance: ImportanceManager,
    private val reader: KnowledgeReader,
    private val tag: String = "默认"
) {
    // === Core Components ===
    private val chatHistory = mutableListOf<ChatTurn>()
    private val maxHistoryRounds = 10
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    // 挂起矛盾
    private val _pendingContradiction = MutableStateFlow<KnowledgeCollision.PendingContradiction?>(null)
    val pendingContradiction: StateFlow<KnowledgeCollision.PendingContradiction?> = _pendingContradiction.asStateFlow()

    // === Security Layer ===
    private val safetyFilter = MemorySafetyFilter()
    
    // === Governance Layer (NEW in v7) ===
    private val governance = MemoryGovernance(store)
    
    // === Provenance Layer (NEW in v7) ===
    private val provenance = MemoryProvenance()
    
    // === Async Processing ===
    private val sleepTimeAgent = SleepTimeAgent(llm, store, tag = "$tag-Sleep")
    private val proceduralMemory = ProceduralMemoryManager(llm, store)

    // 会话标识
    private var currentSessionId = UUID.randomUUID().toString().take(8)
    private var currentSpeaker: String = "user"

    // 反馈信号追踪
    private val feedbackTracker = FeedbackTracker()

    // 统计
    var extractedCount = 0; private set
    var noiseCount = 0; private set
    var blockedCount = 0; private set
    var governanceBlockedCount = 0; private set  // v7: 被治理层阻断
    var provenanceConflictsResolved = 0; private set  // v7: 通过信任比较解决的矛盾

    /**
     * 主对话入口 - v7 完整流水线
     * 
     * 流程:
     * 1. Safety Check (输入过滤)
     * 2. Procedural Injection (经验注入)
     * 3. Knowledge Retrieval (带治理加权)
     * 4. Contradiction Resolution (带信任比较)
     * 5. LLM Generation
     * 6. Memory Extraction (带治理+来源)
     * 7. Async Sleep-Time Processing
     * 8. Feedback Signal Collection
     */
    suspend fun chat(userMessage: String, senderName: String = "user"): ChatResponse {
        currentSpeaker = senderName
        
        // === Phase 1: Input Safety Check ===
        val inputSafety = safetyFilter.validateForWrite(userMessage)
        if (inputSafety.action == SafetyAction.BLOCK) {
            blockedCount++
            return ChatResponse(
                reply = generateSafeRejection(inputSafety.category),
                extractedMemories = emptyList(),
                metadata = ChatMetadata(safetyBlocked = true, blockReason = inputSafety.reason)
            )
        }

        // === Phase 2: Procedural Memory Injection ===
        val proceduralContext = proceduralMemory.generateContextInjection(userMessage)

        // === Phase 3: Knowledge Retrieval with Governance Boost ===
        val retrievedMemories = retrieveWithGovernanceBoost(userMessage)

        // === Phase 4: Provenance-Aware Contradiction Resolution ===
        val resolvedMemories = resolveConflictsWithProvenance(retrievedMemories)

        // === Phase 5: Build Context & Generate Reply ===
        val contextParts = buildContext(resolvedMemories, proceduralContext)
        val reply = generateReply(userMessage, contextParts)

        // === Phase 6: Memory Extraction with Governance + Provenance ===
        val extractedMemories = extractAndStoreWithGovernance(userMessage, reply, senderName)

        // === Phase 7: Submit to Sleep-Time Agent ===
        scope.launch {
            sleepTimeAgent.submitConversation(
                ConversationSegment(
                    sessionId = currentSessionId,
                    messages = listOf(
                        Message(role = "user", content = userMessage, speaker = senderName),
                        Message(role = "assistant", content = reply)
                    ),
                    timestamp = System.currentTimeMillis()
                )
            )
        }

        // === Phase 8: Track Feedback Signal ===
        feedbackTracker.recordExchange(userMessage, reply)

        // Update chat history
        chatHistory.add(ChatTurn(role = "user", content = userMessage, speaker = senderName))
        chatHistory.add(ChatTurn(role = "assistant", content = reply))
        trimHistory()

        return ChatResponse(
            reply = reply,
            extractedMemories = extractedMemories,
            metadata = ChatMetadata(
                retrievedCount = resolvedMemories.size,
                governanceBoostApplied = resolvedMemories.any { it.governanceBoosted },
                provenanceConflictsResolved = resolvedMemories.count { it.conflictResolved }
            )
        )
    }

    /**
     * 带治理层加权的检索
     * 
     * Statutory 层（事实性记忆）获得 1.3x boost
     * Constitutional 层（核心身份）获得 1.5x boost
     * Operational 层（日常交互）保持原始分数
     * 
     * 这确保了"小明的女朋友叫小红"这类事实记忆
     * 在检索排序中优先于"小明今天吃了麻辣烫"
     */
    private suspend fun retrieveWithGovernanceBoost(query: String): List<RetrievedMemory> {
        val rawResults = reader.retrieve(query, topK = 15)
        
        return rawResults.map { memory ->
            val layer = governance.classifyLayer(memory.content, memory.metadata)
            val boostFactor = when (layer) {
                GovernanceLayer.CONSTITUTIONAL -> 1.5
                GovernanceLayer.STATUTORY -> 1.3
                GovernanceLayer.OPERATIONAL -> 1.0
            }
            
            val trustLevel = provenance.getProvenance(memory.id)?.trustLevel ?: TrustLevel.MEDIUM
            val trustBoost = when (trustLevel) {
                TrustLevel.HIGH -> 1.2
                TrustLevel.MEDIUM -> 1.0
                TrustLevel.LOW -> 0.8
            }
            
            RetrievedMemory(
                id = memory.id,
                content = memory.content,
                score = memory.score * boostFactor * trustBoost,
                layer = layer,
                trustLevel = trustLevel,
                governanceBoosted = boostFactor > 1.0,
                conflictResolved = false
            )
        }.sortedByDescending { it.score }.take(8)  // Context Economy: 最多8条
    }

    /**
     * 基于 Provenance 的矛盾解决
     * 
     * 当检索到的记忆之间存在矛盾时：
     * 1. 比较信任等级 → 高信任胜出
     * 2. 信任相同 → 比较时间新旧 → 新的胜出
     * 3. 都相同 → 保留两者，交给 LLM 判断
     */
    private suspend fun resolveConflictsWithProvenance(
        memories: List<RetrievedMemory>
    ): List<RetrievedMemory> {
        if (memories.size <= 1) return memories
        
        val resolved = mutableListOf<RetrievedMemory>()
        val processed = mutableSetOf<String>()
        
        for (i in memories.indices) {
            if (memories[i].id in processed) continue
            
            var current = memories[i]
            for (j in i + 1 until memories.size) {
                if (memories[j].id in processed) continue
                
                // 快速矛盾检测：同主题不同断言
                if (isConflicting(current.content, memories[j].content)) {
                    val comparison = provenance.compareProvenance(current.id, memories[j].id)
                    
                    when {
                        comparison.winnerTrust != null && comparison.winnerTrust == current.trustLevel -> {
                            // current wins
                            processed.add(memories[j].id)
                            current = current.copy(conflictResolved = true)
                            provenanceConflictsResolved++
                        }
                        comparison.winnerTrust != null && comparison.winnerTrust == memories[j].trustLevel -> {
                            // j wins
                            processed.add(current.id)
                            current = memories[j].copy(conflictResolved = true)
                            provenanceConflictsResolved++
                        }
                        else -> {
                            // TIE: keep both, let LLM decide
                            resolved.add(memories[j])
                        }
                    }
                }
            }
            
            if (current.id !in processed) {
                resolved.add(current)
            }
        }
        
        return resolved.sortedByDescending { it.score }
    }

    /**
     * 带治理和来源追踪的记忆提取与存储
     * 
     * 流程:
     * 1. LLM 提取知识点
     * 2. Safety 过滤
     * 3. Governance 分类 + 权限检查
     * 4. Gate 判断（重要性）
     * 5. Collision 检测
     * 6. Store 存储 + Provenance 记录
     */
    private suspend fun extractAndStoreWithGovernance(
        userMessage: String,
        aiReply: String,
        senderName: String
    ): List<ExtractedMemory> {
        // Step 1: LLM Extract
        val rawExtractions = extractKnowledge(userMessage, aiReply)
        val stored = mutableListOf<ExtractedMemory>()
        
        for (extraction in rawExtractions) {
            // Step 2: Safety Filter
            val safety = safetyFilter.validateForWrite(extraction.content)
            if (safety.action == SafetyAction.BLOCK) {
                blockedCount++
                continue
            }
            
            // Step 3: Governance Classification + Permission Check
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
            
            // Step 4: Gate (importance threshold)
            val gateResult = gate.evaluate(extraction.content)
            if (!gateResult.passed) {
                noiseCount++
                continue
            }
            
            // Step 5: Collision Detection (with provenance-aware resolution)
            val collisionResult = collision.check(extraction.content)
            if (collisionResult.hasConflict) {
                val existingProvenance = provenance.getProvenance(collisionResult.existingId!!)
                val newTrust = determineTrustLevel(senderName, extraction.source)
                
                if (existingProvenance != null && existingProvenance.trustLevel.ordinal > newTrust.ordinal) {
                    // Existing memory has higher trust, skip new one
                    continue
                }
                // New has equal or higher trust, proceed to overwrite
            }
            
            // Step 6: Store + Record Provenance
            val memoryId = store.save(extraction.content, extraction.metadata)
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
            
            extractedCount++
            stored.add(ExtractedMemory(
                id = memoryId,
                content = extraction.content,
                layer = layer,
                trustLevel = determineTrustLevel(senderName, extraction.source)
            ))
        }
        
        // Learn procedural patterns
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
     * 信任等级判定
     * 
     * HIGH: 用户直接陈述的事实（"我女朋友叫小红"）
     * MEDIUM: 对话推断的信息（从上下文推断的职业等）
     * LOW: 外部来源或不确定信息
     */
    private fun determineTrustLevel(senderName: String, source: String): TrustLevel {
        return when {
            senderName != "assistant" && source == "direct_statement" -> TrustLevel.HIGH
            senderName != "assistant" && source == "inference" -> TrustLevel.MEDIUM
            else -> TrustLevel.LOW
        }
    }

    /**
     * 构建上下文（Economy 原则）
     * 
     * 上下文预算分配：
     * - 系统 prompt + 人格: ~500 tokens
     * - Procedural 经验注入: ~200 tokens
     * - 检索到的记忆: ~800 tokens (max 8 条)
     * - 对话历史: ~1000 tokens (max 10 轮)
     * - 当前用户消息: 不限
     * 
     * 总预算 ≈ 2500 tokens，远低于 128K 上限，
     * 确保 LLM 有充足空间生成高质量回复
     */
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
            "$trustTag$layerTag ${memory.content}"
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
注意：标记 [✓] 的记忆来自用户直接陈述，可信度最高；标记 [?] 的信息需要谨慎使用。"""
    }

    private fun formatChatHistory(): String {
        return chatHistory.takeLast(maxHistoryRounds * 2).joinToString("\n") {
            "${it.role}: ${it.content}"
        }
    }

    /**
     * 生成安全拒绝回复
     */
    private fun generateSafeRejection(category: SafetyCategory?): String {
        return when (category) {
            SafetyCategory.PROMPT_INJECTION -> "嗯？你说的话有点奇怪呢，换个话题聊聊吧~"
            SafetyCategory.SENSITIVE_DATA -> "这类信息不太适合我记住哦，我们聊点别的吧~"
            SafetyCategory.DANGEROUS_URL -> "这个链接看起来不太安全呢，我先不点了~"
            else -> "抱歉，这个我不太方便回应，我们聊点别的好吗？"
        }
    }

    /**
     * 快速矛盾检测
     * 简化版：同主题 + 不同断言 = 可能矛盾
     */
    private suspend fun isConflicting(contentA: String, contentB: String): Boolean {
        // 实际项目中通过 LLM 判断，这里用简化逻辑
        val subjectA = extractSubject(contentA)
        val subjectB = extractSubject(contentB)
        return subjectA == subjectB && contentA != contentB && subjectA.isNotBlank()
    }

    private fun extractSubject(content: String): String {
        // 简化：提取"XX的YY"模式
        val pattern = Regex("(\\w+)的(\\w+)")
        return pattern.find(content)?.value ?: ""
    }

    /**
     * LLM 知识提取
     */
    private suspend fun extractKnowledge(userMessage: String, aiReply: String): List<RawExtraction> {
        val prompt = """分析以下对话，提取值得长期记住的知识点。
对于每个知识点，判断：
- content: 知识内容（一句话）
- source: "direct_statement"(用户直接说的) 或 "inference"(推断的)
- importance: 1-5 分

用户: $userMessage
AI: $aiReply

以 JSON 数组格式返回，只返回重要度 >= 3 的知识点。"""

        val response = llm.chat(listOf(
            mapOf("role" to "system", "content" to "你是一个精准的知识提取器"),
            mapOf("role" to "user", "content" to prompt)
        ))
        
        return parseExtractions(response)
    }

    private fun parseExtractions(llmResponse: String): List<RawExtraction> {
        return try {
            val json = Json { ignoreUnknownKeys = true }
            json.decodeFromString<List<RawExtraction>>(llmResponse)
        } catch (e: Exception) {
            emptyList()
        }
    }

    private suspend fun generateReply(userMessage: String, context: ContextParts): String {
        val messages = mutableListOf<Map<String, String>>()
        
        // System prompt with persona
        var systemContent = context.systemPrompt
        if (context.proceduralHints.isNotBlank()) {
            systemContent += "\n\n【交互经验提示】\n${context.proceduralHints}"
        }
        if (context.memoryContext.isNotBlank()) {
            systemContent += "\n\n【相关记忆】\n${context.memoryContext}"
        }
        messages.add(mapOf("role" to "system", "content" to systemContent))
        
        // Chat history
        for (turn in chatHistory.takeLast(maxHistoryRounds * 2)) {
            messages.add(mapOf("role" to turn.role, "content" to turn.content))
        }
        
        // Current message
        messages.add(mapOf("role" to "user", "content" to userMessage))
        
        return llm.chat(messages)
    }

    // === Session Management ===
    
    fun resetSession() {
        // Trigger sleep-time consolidation before reset
        scope.launch {
            sleepTimeAgent.triggerConsolidation()
        }
        chatHistory.clear()
        currentSessionId = UUID.randomUUID().toString().take(8)
        feedbackTracker.reset()
    }

    /**
     * Right-to-Forget: 用户主权实现
     * 用户可以要求删除特定范围的记忆
     */
    suspend fun exerciseRightToForget(
        userId: String,
        scope: ForgetScope
    ): ForgetResult {
        return governance.exerciseRightToForget(userId, scope)
    }

    /**
     * 解决挂起的矛盾（需用户确认的 Statutory 层变更）
     */
    suspend fun resolveContradiction(
        contradictionId: String,
        userChoice: UserChoice
    ): Boolean {
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
    
    fun getStats(): EngineStats {
        return EngineStats(
            extracted = extractedCount,
            noise = noiseCount,
            blocked = blockedCount,
            governanceBlocked = governanceBlockedCount,
            provenanceResolved = provenanceConflictsResolved,
            sessionId = currentSessionId,
            historySize = chatHistory.size
        )
    }

    fun shutdown() {
        scope.cancel()
    }

    private fun trimHistory() {
        while (chatHistory.size > maxHistoryRounds * 2) {
            chatHistory.removeFirst()
        }
    }
}

// === Data Classes ===

data class ChatTurn(val role: String, val content: String, val speaker: String? = null)

data class ChatResponse(
    val reply: String,
    val extractedMemories: List<ExtractedMemory>,
    val metadata: ChatMetadata
)

data class ChatMetadata(
    val retrievedCount: Int = 0,
    val governanceBoostApplied: Boolean = false,
    val provenanceConflictsResolved: Int = 0,
    val safetyBlocked: Boolean = false,
    val blockReason: String? = null
)

data class RetrievedMemory(
    val id: String,
    val content: String,
    val score: Double,
    val layer: GovernanceLayer,
    val trustLevel: TrustLevel,
    val governanceBoosted: Boolean,
    val conflictResolved: Boolean
)

data class ExtractedMemory(
    val id: String,
    val content: String,
    val layer: GovernanceLayer,
    val trustLevel: TrustLevel
)

data class RawExtraction(
    val content: String,
    val source: String,
    val importance: Int,
    val metadata: Map<String, String> = emptyMap()
)

data class ContextParts(
    val systemPrompt: String,
    val proceduralHints: String,
    val memoryContext: String,
    val chatHistory: String
)

data class EngineStats(
    val extracted: Int,
    val noise: Int,
    val blocked: Int,
    val governanceBlocked: Int,
    val provenanceResolved: Int,
    val sessionId: String,
    val historySize: Int
)

enum class ForgetScope { ALL, SESSION, TOPIC }
enum class UserChoice { KEEP_EXISTING, ACCEPT_NEW, KEEP_BOTH }
data class ForgetResult(val deletedCount: Int, val success: Boolean)

/**
 * 反馈信号追踪器
 * 
 * 隐式信号收集：
 * - 用户回复长度变化 → 参与度指标
 * - 连续对话轮数 → 满意度指标
 * - 话题切换频率 → 兴趣指标
 * - 特定否定词 → 负面反馈
 */
class FeedbackTracker {
    private val exchanges = mutableListOf<Exchange>()
    
    fun recordExchange(userMsg: String, aiReply: String) {
        exchanges.add(Exchange(
            userLength = userMsg.length,
            aiLength = aiReply.length,
            timestamp = System.currentTimeMillis(),
            hasNegation = userMsg.contains("不是") || userMsg.contains("不对") || 
                         userMsg.contains("错了") || userMsg.contains("没有")
        ))
    }
    
    fun getLatestSignal(): FeedbackSignal? {
        if (exchanges.size < 2) return null
        val recent = exchanges.takeLast(3)
        
        val avgLength = recent.map { it.userLength }.average()
        val hasNegation = recent.any { it.hasNegation }
        
        return FeedbackSignal(
            engagement = if (avgLength > 20) 0.8 else 0.4,
            satisfaction = if (!hasNegation) 0.7 else 0.3,
            type = if (hasNegation) "negative" else "positive"
        )
    }
    
    fun reset() { exchanges.clear() }
    
    private data class Exchange(
        val userLength: Int,
        val aiLength: Int,
        val timestamp: Long,
        val hasNegation: Boolean
    )
}
