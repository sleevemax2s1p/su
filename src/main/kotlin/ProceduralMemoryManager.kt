package com.memory.procedural

import com.memory.llm.DeepSeekClient
import com.memory.store.KnowledgeStore
import kotlinx.serialization.json.*
import java.text.SimpleDateFormat
import java.util.Date

/**
 * 程序性记忆管理器
 * 
 * 灵感来源：
 * - SRA（Skill Retrieval Augmentation）§14.21.4
 * - Mem0 Skill Graph §14.20.3
 * - LinkedIn CMA Procedural Memory §14.18.2
 * - MALMAS Procedural Memory §14.17.4
 * 
 * 核心理念：
 * Agent 不仅记住事实（"用户叫小明"），还记住交互模式（"跟这个用户这样聊效果好"）
 * 
 * 两种程序性记忆：
 * 1. 关于用户的交互模式（被动检索）—— 用户喜欢什么样的回复风格
 * 2. Agent 自身的技能库（主动决策）—— 什么策略在什么场景下有效
 */
class ProceduralMemoryManager(
    private val llm: DeepSeekClient,
    private val store: KnowledgeStore
) {
    // 交互模式库
    private val interactionPatterns = mutableListOf<InteractionPattern>()
    
    // 策略效用记录（MemRL 简化版：记录策略是否有效）
    private val strategyUtility = mutableMapOf<String, StrategyRecord>()
    
    private fun log(msg: String) {
        val ts = SimpleDateFormat("HH:mm:ss.SSS").format(Date())
        println("[$ts][Procedural] $msg")
    }

    /**
     * 从对话交互中学习模式
     * 
     * 在 Sleep-Time 阶段调用（不阻塞主对话）
     * 分析哪些回复风格/策略收到了积极反馈
     */
    suspend fun learnFromInteraction(
        userMessage: String,
        agentReply: String,
        userFeedback: FeedbackSignal
    ) {
        // 只从积极反馈中学习模式（避免负面强化）
        if (userFeedback.sentiment < 0.3) {
            // 记录失败策略
            recordStrategyOutcome(
                context = extractContext(userMessage),
                strategy = extractStrategy(agentReply),
                success = false
            )
            return
        }
        
        if (userFeedback.sentiment > 0.7) {
            // 提取成功的交互模式
            val pattern = extractInteractionPattern(userMessage, agentReply, userFeedback)
            if (pattern != null) {
                addPattern(pattern)
                log("学到新交互模式: ${pattern.description}")
            }
            
            // 记录成功策略
            recordStrategyOutcome(
                context = extractContext(userMessage),
                strategy = extractStrategy(agentReply),
                success = true
            )
        }
    }

    /**
     * 根据当前上下文检索最相关的交互模式
     * 
     * SRA 启发：关键问题不是「能否检索到」，而是「何时加载」
     * 只在置信度足够高时才建议加载
     */
    fun retrievePatterns(currentContext: String, confidenceThreshold: Double = 0.7): List<PatternSuggestion> {
        if (interactionPatterns.isEmpty()) return emptyList()
        
        val suggestions = mutableListOf<PatternSuggestion>()
        
        for (pattern in interactionPatterns) {
            val relevance = calculateRelevance(currentContext, pattern)
            if (relevance >= confidenceThreshold) {
                suggestions.add(PatternSuggestion(
                    pattern = pattern,
                    confidence = relevance,
                    shouldLoad = relevance > 0.85 // 只有高置信度才主动加载
                ))
            }
        }
        
        return suggestions.sortedByDescending { it.confidence }.take(3)
    }

    /**
     * 获取当前最有效的策略
     * 
     * MemRL Q-value 的简化版：
     * 不用完整的 Bellman 方程，但用 success rate 作为效用估计
     */
    fun getBestStrategy(context: String): String? {
        val contextKey = extractContext(context)
        val record = strategyUtility[contextKey] ?: return null
        
        // 只返回成功率 > 60% 且使用次数 > 3 的策略
        return if (record.successRate > 0.6 && record.totalAttempts > 3) {
            record.bestStrategy
        } else null
    }

    /**
     * 生成程序性记忆的上下文注入文本
     * 
     * 注入到系统提示中，告诉 Agent「跟这个用户怎么聊效果好」
     */
    fun generateContextInjection(currentInput: String): String? {
        val patterns = retrievePatterns(currentInput)
        if (patterns.isEmpty()) return null
        
        val loadablePatterns = patterns.filter { it.shouldLoad }
        if (loadablePatterns.isEmpty()) return null
        
        return buildString {
            appendLine("（你从经验中学到的交互模式——自然运用，不要机械执行）")
            loadablePatterns.forEach { suggestion ->
                appendLine("- ${suggestion.pattern.description}")
            }
        }
    }

    // === Private Helpers ===
    
    private suspend fun extractInteractionPattern(
        userMessage: String,
        agentReply: String,
        feedback: FeedbackSignal
    ): InteractionPattern? {
        // 用 LLM 提取高层交互模式
        val prompt = """分析以下成功的对话交互，提取一条可复用的交互模式/策略。

用户说: "${userMessage.take(200)}"
助手回复: "${agentReply.take(200)}"
用户反馈: ${feedback.description}

提取一条简洁的交互模式（20字以内），描述「在什么情况下用什么策略有效」。
格式: {"pattern": "情境描述 → 有效策略", "context_type": "emotional/factual/creative/planning"}
如果没有可提取的通用模式，返回 {"pattern": null}"""
        
        return try {
            val result = llm.chatJson(prompt, "提取交互模式")
            val patternStr = result.jsonObject["pattern"]?.jsonPrimitive?.contentOrNull
            val contextType = result.jsonObject["context_type"]?.jsonPrimitive?.contentOrNull ?: "general"
            
            if (patternStr != null) {
                InteractionPattern(
                    description = patternStr,
                    contextType = contextType,
                    learnedAt = System.currentTimeMillis(),
                    successCount = 1,
                    totalUses = 1
                )
            } else null
        } catch (e: Exception) {
            log("模式提取失败: ${e.message}")
            null
        }
    }
    
    private fun addPattern(pattern: InteractionPattern) {
        // 检查是否与已有模式重复
        val existing = interactionPatterns.find { 
            it.description == pattern.description || it.contextType == pattern.contextType 
        }
        
        if (existing != null) {
            // 强化已有模式
            existing.successCount++
            existing.totalUses++
        } else {
            interactionPatterns.add(pattern)
            // 容量控制：最多保留 50 个模式
            if (interactionPatterns.size > 50) {
                // 移除最低效用的模式
                interactionPatterns.sortByDescending { it.successRate }
                interactionPatterns.removeAt(interactionPatterns.lastIndex)
            }
        }
    }
    
    private fun calculateRelevance(context: String, pattern: InteractionPattern): Double {
        // 简单的关键词匹配 + 类型匹配
        // TODO: 后续升级为 embedding 相似度
        val contextWords = context.toSet()
        val patternWords = pattern.description.toSet()
        val overlap = contextWords.intersect(patternWords).size.toDouble()
        return (overlap / maxOf(patternWords.size, 1)).coerceIn(0.0, 1.0)
    }
    
    private fun extractContext(input: String): String {
        // 粗粒度上下文分类
        return when {
            input.contains(Regex("难过|伤心|郁闷|烦|焦虑|压力")) -> "emotional_negative"
            input.contains(Regex("开心|高兴|好消息|棒|太好了")) -> "emotional_positive"
            input.contains(Regex("怎么|如何|帮我|教我")) -> "help_seeking"
            input.contains(Regex("计划|打算|准备|明天|下周")) -> "planning"
            else -> "general"
        }
    }
    
    private fun extractStrategy(reply: String): String {
        return when {
            reply.contains(Regex("哈哈|笑|逗")) -> "humor"
            reply.contains(Regex("我觉得|建议|不如")) -> "advice"
            reply.contains(Regex("抱抱|辛苦|不容易")) -> "empathy"
            reply.contains(Regex("等等|之前|记得")) -> "recall_and_connect"
            else -> "neutral"
        }
    }
    
    private fun recordStrategyOutcome(context: String, strategy: String, success: Boolean) {
        val key = context
        val record = strategyUtility.getOrPut(key) { 
            StrategyRecord(context = key) 
        }
        record.totalAttempts++
        if (success) {
            record.successCount++
            record.bestStrategy = strategy
        }
    }
    
    fun getStats(): String {
        return "程序性记忆 | 交互模式: ${interactionPatterns.size} | 策略记录: ${strategyUtility.size}"
    }
}

// === Data Models ===

data class InteractionPattern(
    val description: String,
    val contextType: String,
    val learnedAt: Long,
    var successCount: Int = 0,
    var totalUses: Int = 0
) {
    val successRate: Double get() = if (totalUses > 0) successCount.toDouble() / totalUses else 0.0
}

data class PatternSuggestion(
    val pattern: InteractionPattern,
    val confidence: Double,
    val shouldLoad: Boolean
)

data class FeedbackSignal(
    val sentiment: Double, // 0.0 = 极负面, 1.0 = 极正面
    val engagement: Double, // 0.0 = 无互动, 1.0 = 高互动
    val description: String = ""
)

data class StrategyRecord(
    val context: String,
    var totalAttempts: Int = 0,
    var successCount: Int = 0,
    var bestStrategy: String = ""
) {
    val successRate: Double get() = if (totalAttempts > 0) successCount.toDouble() / totalAttempts else 0.0
}
