package com.memory.sleeptime

import com.memory.llm.DeepSeekClient
import com.memory.store.KnowledgeStore
import com.memory.model.*
import kotlinx.coroutines.*
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.*
import java.text.SimpleDateFormat
import java.util.Date

/**
 * Sleep-Time Memory Agent
 * 
 * 灵感来源：Letta Sleep-Time Compute（§14.20.1, §14.21）
 * 
 * 核心理念：
 * - 对话结束后异步触发记忆整理，而非实时处理
 * - 独立的记忆管理 Agent，与主对话 Agent 解耦
 * - 后台完成记忆链接、巩固、衰减、演化
 * 
 * 设计原则（Context Constitution 启发）：
 * - 记忆是 Agent 存在的基础，不是附加功能
 * - 上下文是稀缺资源，需要主动管理
 * - Agent 通过管理自己的记忆来「学习」
 */
class SleepTimeAgent(
    private val llm: DeepSeekClient,
    private val store: KnowledgeStore,
    private val tag: String = "SleepTime"
) {
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    
    // 待处理的对话片段队列
    private val conversationQueue = Channel<ConversationSegment>(capacity = 64)
    
    // Sleep-Time 任务状态
    private val _status = MutableStateFlow(SleepTimeStatus.IDLE)
    val status: StateFlow<SleepTimeStatus> = _status.asStateFlow()
    
    // 巩固事件流（供监控/日志）
    private val _events = MutableSharedFlow<SleepTimeEvent>(replay = 0)
    val events: SharedFlow<SleepTimeEvent> = _events.asSharedFlow()
    
    private var processedSegments = 0
    private var consolidatedMemories = 0
    private var evolvedLinks = 0
    
    private fun log(msg: String) {
        val ts = SimpleDateFormat("HH:mm:ss.SSS").format(Date())
        println("[$ts][SleepTime·$tag] $msg")
    }

    /**
     * 启动 Sleep-Time Agent 后台循环
     * 持续监听对话片段队列，执行记忆管理任务
     */
    fun start() {
        scope.launch {
            log("Sleep-Time Agent 已启动，等待对话片段...")
            for (segment in conversationQueue) {
                try {
                    _status.value = SleepTimeStatus.PROCESSING
                    processSleepTimeTask(segment)
                    processedSegments++
                    _status.value = SleepTimeStatus.IDLE
                } catch (e: Exception) {
                    log("Sleep-Time 处理异常: ${e.message}")
                    _status.value = SleepTimeStatus.ERROR
                    delay(1000) // 错误后短暂等待
                    _status.value = SleepTimeStatus.IDLE
                }
            }
        }
        
        // 定期巩固任务（每30分钟）
        scope.launch {
            while (isActive) {
                delay(30 * 60 * 1000) // 30 minutes
                performPeriodicConsolidation()
            }
        }
    }

    /**
     * 提交对话片段到 Sleep-Time 队列
     * 非阻塞，立即返回
     */
    suspend fun submitConversation(segment: ConversationSegment) {
        conversationQueue.send(segment)
        log("对话片段已入队: ${segment.messages.size} 条消息")
    }

    /**
     * 核心 Sleep-Time 处理流程
     * 
     * 三阶段处理：
     * 1. 记忆链接演化（A-MEM 启发）
     * 2. 记忆巩固与压缩
     * 3. 衰减与清理
     */
    private suspend fun processSleepTimeTask(segment: ConversationSegment) {
        log("开始处理对话片段 [${segment.sessionId}]: ${segment.messages.size} 条消息")
        val startTime = System.currentTimeMillis()
        
        // Phase 1: 链接演化（A-MEM Zettelkasten 启发）
        val newLinks = performLinkEvolution(segment)
        evolvedLinks += newLinks
        log("  Phase 1 完成: 演化 $newLinks 条链接")
        
        // Phase 2: 记忆巩固
        val consolidated = performConsolidation(segment)
        consolidatedMemories += consolidated
        log("  Phase 2 完成: 巩固 $consolidated 条记忆")
        
        // Phase 3: 衰减检查
        val decayed = performDecayCheck()
        log("  Phase 3 完成: 衰减 $decayed 条记忆")
        
        val duration = System.currentTimeMillis() - startTime
        log("Sleep-Time 处理完成 (${duration}ms)")
        
        _events.emit(SleepTimeEvent(
            type = EventType.SEGMENT_PROCESSED,
            sessionId = segment.sessionId,
            newLinks = newLinks,
            consolidated = consolidated,
            decayed = decayed,
            durationMs = duration
        ))
    }

    /**
     * Phase 1: 链接演化
     * 
     * 灵感：A-MEM（§14.20.2）
     * 当新记忆加入时，检查是否应该与已有记忆建立/更新链接
     * 同时更新已有记忆的 Context 描述（记忆演化）
     */
    private suspend fun performLinkEvolution(segment: ConversationSegment): Int {
        val recentMemories = store.getRecentlyAdded(limit = 10)
        if (recentMemories.isEmpty()) return 0
        
        var linkCount = 0
        
        for (memory in recentMemories) {
            // 找到可能相关但未链接的记忆
            val candidates = store.search(memory.content, topK = 5)
                .filter { it.id != memory.id && !store.hasLink(memory.id, it.id) }
            
            if (candidates.isEmpty()) continue
            
            // 用 LLM 判断是否应该建立链接
            val linkPrompt = buildLinkEvaluationPrompt(memory, candidates)
            try {
                val result = llm.chatJson(linkPrompt, "评估记忆链接")
                val links = parseLinkResult(result, memory, candidates)
                links.forEach { (targetId, reason) ->
                    store.createLink(memory.id, targetId, reason)
                    linkCount++
                }
            } catch (e: Exception) {
                log("  链接评估失败: ${e.message}")
            }
        }
        
        return linkCount
    }

    /**
     * Phase 2: 记忆巩固
     * 
     * 灵感：Memory as Metabolism（§14.18.4）的 CONSOLIDATE 机制
     * 批量深度整合，类似睡眠巩固
     * 
     * 注意：保留原始事实（STONE 原则），摘要仅作导航层
     */
    private suspend fun performConsolidation(segment: ConversationSegment): Int {
        val allActive = store.getAllActive()
        if (allActive.size < 10) return 0 // 记忆太少不需要巩固
        
        // 找到高度相关但分散的记忆簇
        val clusters = findMemoryClusters(allActive)
        var consolidated = 0
        
        for (cluster in clusters) {
            if (cluster.size < 3) continue
            
            // 生成巩固摘要（仅作导航，不替代原始事实）
            val summaryPrompt = buildConsolidationPrompt(cluster)
            try {
                val summary = llm.chat(
                    "你是记忆整理助手。将以下相关记忆片段整合为一个清晰的概要。保留所有具体事实，不要丢失细节。",
                    summaryPrompt,
                    temperature = 0.3
                )
                
                // 创建导航节点（不删除原始事实）
                store.addNavigationNode(
                    summary = summary,
                    sourceIds = cluster.map { it.id },
                    type = NavigationType.CONSOLIDATION
                )
                consolidated++
            } catch (e: Exception) {
                log("  巩固失败: ${e.message}")
            }
        }
        
        return consolidated
    }

    /**
     * Phase 3: 衰减检查
     * 
     * 区分「软遗忘」（Ebbinghaus 衰减）和「硬遗忘」（主动失效）
     * Memora Benchmark（§14.15.2）启发：正确遗忘与正确记忆同等重要
     */
    private suspend fun performDecayCheck(): Int {
        val allActive = store.getAllActive()
        var decayedCount = 0
        
        for (memory in allActive) {
            // 软遗忘：长时间未访问 + 低重要度
            if (memory.daysSinceLastAccess > 30 && memory.importance < 0.3) {
                store.softDecay(memory.id)
                decayedCount++
                continue
            }
            
            // 硬遗忘检查：是否被后续矛盾标记为无效
            if (memory.contradictedBy.isNotEmpty() && memory.contradictionConfirmed) {
                store.hardInvalidate(memory.id)
                decayedCount++
            }
        }
        
        return decayedCount
    }

    /**
     * 定期巩固：独立于对话的周期性维护
     * 
     * 灵感：AUDIT 机制（Memory as Metabolism）
     * 定期压力测试高权重记忆：暂时移除并检测是否影响系统表现
     */
    private suspend fun performPeriodicConsolidation() {
        log("执行定期巩固...")
        _status.value = SleepTimeStatus.CONSOLIDATING
        
        try {
            // 1. 检查过期的时间敏感记忆
            val expiredCount = store.markExpiredMemories()
            log("  标记过期记忆: $expiredCount 条")
            
            // 2. 重新计算记忆重要度
            store.recalculateImportance()
            log("  重要度重算完成")
            
            // 3. 检查记忆完整性（Context Constitution: 避免目标漂移）
            val integrityIssues = checkMemoryIntegrity()
            if (integrityIssues > 0) {
                log("  发现 $integrityIssues 个完整性问题")
            }
            
        } catch (e: Exception) {
            log("定期巩固异常: ${e.message}")
        } finally {
            _status.value = SleepTimeStatus.IDLE
        }
    }

    /**
     * 记忆完整性审计
     * 
     * 灵感：Knowledge Objects（§14.15.4）的「目标漂移」警告
     * AI 不知道自己忘了什么 → 需要主动检查核心画像的完整性
     */
    private suspend fun checkMemoryIntegrity(): Int {
        val coreProfile = store.getCoreProfile()
        var issues = 0
        
        // 检查核心画像字段完整性
        val expectedFields = listOf("name", "occupation", "family", "preferences", "goals")
        for (field in expectedFields) {
            if (!coreProfile.containsKey(field)) {
                log("    完整性缺口: 核心画像缺少 '$field' 维度")
                issues++
            }
        }
        
        return issues
    }

    // === Helper Methods ===
    
    private fun buildLinkEvaluationPrompt(source: KnowledgeNode, candidates: List<KnowledgeNode>): String {
        return buildString {
            appendLine("评估以下新记忆与候选记忆之间是否存在有意义的关联。")
            appendLine()
            appendLine("新记忆: \"${source.content}\"")
            appendLine()
            appendLine("候选记忆:")
            candidates.forEachIndexed { i, c ->
                appendLine("  [$i] \"${c.content}\"")
            }
            appendLine()
            appendLine("返回JSON: {\"links\": [{\"index\": 0, \"reason\": \"关联原因\"}]} 或 {\"links\": []}")
        }
    }
    
    private fun parseLinkResult(
        result: kotlinx.serialization.json.JsonElement,
        source: KnowledgeNode,
        candidates: List<KnowledgeNode>
    ): List<Pair<String, String>> {
        val links = result.jsonObject["links"]?.jsonArray ?: return emptyList()
        return links.mapNotNull { link ->
            val obj = link.jsonObject
            val index = obj["index"]?.jsonPrimitive?.intOrNull ?: return@mapNotNull null
            val reason = obj["reason"]?.jsonPrimitive?.contentOrNull ?: return@mapNotNull null
            if (index in candidates.indices) {
                candidates[index].id to reason
            } else null
        }
    }
    
    private fun findMemoryClusters(memories: List<KnowledgeNode>): List<List<KnowledgeNode>> {
        // 简单的基于标签/关键词的聚类
        // TODO: 后续可升级为 embedding 聚类
        val clusters = mutableListOf<List<KnowledgeNode>>()
        val grouped = memories.groupBy { extractMainTopic(it.content) }
        grouped.values.filter { it.size >= 3 }.forEach { clusters.add(it) }
        return clusters
    }
    
    private fun extractMainTopic(content: String): String {
        // 简单的主题提取：取内容前10个字作为粗粒度分组依据
        return content.take(10)
    }
    
    private fun buildConsolidationPrompt(cluster: List<KnowledgeNode>): String {
        return cluster.joinToString("\n") { "- ${it.content}" }
    }

    fun getStats(): String {
        return "Sleep-Time | 已处理: ${processedSegments}段 | 巩固: ${consolidatedMemories}条 | 演化链接: ${evolvedLinks}条 | 状态: ${_status.value}"
    }

    fun shutdown() {
        conversationQueue.close()
        scope.cancel()
        log("Sleep-Time Agent 已关闭")
    }
}

// === Data Models ===

data class ConversationSegment(
    val sessionId: String,
    val messages: List<Message>,
    val timestamp: Long = System.currentTimeMillis()
)

data class Message(
    val role: String, // "user" or "assistant"
    val content: String,
    val senderName: String? = null
)

enum class SleepTimeStatus {
    IDLE,           // 等待中
    PROCESSING,     // 处理对话片段
    CONSOLIDATING,  // 定期巩固中
    ERROR           // 异常
}

data class SleepTimeEvent(
    val type: EventType,
    val sessionId: String = "",
    val newLinks: Int = 0,
    val consolidated: Int = 0,
    val decayed: Int = 0,
    val durationMs: Long = 0
)

enum class EventType {
    SEGMENT_PROCESSED,
    PERIODIC_CONSOLIDATION,
    INTEGRITY_CHECK
}

enum class NavigationType {
    CONSOLIDATION,  // 巩固摘要
    REFLECTION,     // 反思洞察
    TEMPORAL_CHAIN  // 时间链
}
