package com.memory.extraction

import com.memory.policy.MemoryAction
import com.memory.governance.GovernanceLayer

/**
 * 上下文感知提取器 v2 (Context-Aware Extractor)
 * 
 * 基于 Cognis (Retrieve-then-Extract) + AtomMem (CRUD actions):
 * 
 * 输入: 当前消息 + 已检索的相关记忆
 * 输出: 一组 ExtractionAction (NEW/UPDATE/DELETE/SKIP)
 * 
 * 关键改进：
 * - v1 只能 NEW/UPDATE/SKIP
 * - v2 增加 DELETE (用户显式否定) 和 SUMMARIZE (信息压缩)
 * 
 * 否定检测 (DELETE triggers):
 * - "我不再...了" / "我已经不...了" / "其实我没有..."
 * - "那是以前的事了" / "不是...而是..."
 * - 显式修正: "我说错了" / "更正一下"
 * 
 * 信息压缩 (SUMMARIZE triggers):
 * - 同一实体有 3+ 条记忆
 * - 多条记忆可合并为一条更精确的
 */

// === 否定模式匹配 ===

object NegationDetector {
    
    // 显式否定模式
    private val negationPatterns = listOf(
        Regex("我(不再|已经不|没有再|不想再)(.+?)了"),
        Regex("其实我(没有|不是|并不)(.+)"),
        Regex("(那是以前|以前的事|过去了)"),
        Regex("我说错了|更正一下|纠正一下"),
        Regex("不是(.+?)而是(.+)"),
        Regex("我(已经|已|都)(搬|换|离|辞|分)"),
        Regex("我(不|没)(喜欢|想|要|在|住)(.+?)了"),
    )
    
    // 修正模式 (不是完全否定，而是更新)
    private val correctionPatterns = listOf(
        Regex("不是(.+?)而是(.+)"),
        Regex("(其实|实际上|准确来说)(.+?)是(.+)"),
        Regex("我(换|搬|转)到了(.+)"),
    )
    
    /**
     * 检测消息中的否定意图
     * 
     * @return 匹配到的否定类型和目标信息
     */
    fun detect(message: String, existingMemories: List<ExistingMemory>): List<NegationResult> {
        val results = mutableListOf<NegationResult>()
        
        for (pattern in negationPatterns) {
            val match = pattern.find(message) ?: continue
            
            // 找到与否定内容最相关的已有记忆
            val negatedContent = match.groupValues.drop(1).joinToString("")
            val targetMemory = findMostRelevantMemory(negatedContent, existingMemories)
            
            if (targetMemory != null) {
                val isCorrection = correctionPatterns.any { it.containsMatchIn(message) }
                results.add(NegationResult(
                    targetMemoryId = targetMemory.id,
                    targetContent = targetMemory.content,
                    negationType = if (isCorrection) NegationType.CORRECTION else NegationType.FULL_NEGATION,
                    confidence = calculateNegationConfidence(match, targetMemory),
                    extractedReplacement = if (isCorrection) extractReplacement(message, match) else null
                ))
            }
        }
        
        return results
    }
    
    private fun findMostRelevantMemory(
        negatedContent: String, 
        memories: List<ExistingMemory>
    ): ExistingMemory? {
        // 简单匹配：关键词重叠
        return memories.maxByOrNull { memory ->
            val overlap = negatedContent.toSet().intersect(memory.content.toSet()).size
            overlap.toDouble() / maxOf(negatedContent.length, memory.content.length, 1)
        }
    }
    
    private fun calculateNegationConfidence(match: MatchResult, memory: ExistingMemory): Double {
        // 基础置信度 0.7，如果有明确的否定词则更高
        var confidence = 0.7
        if (match.value.contains("不再") || match.value.contains("已经不")) confidence = 0.85
        if (match.value.contains("说错了") || match.value.contains("更正")) confidence = 0.95
        return confidence
    }
    
    private fun extractReplacement(message: String, match: MatchResult): String? {
        // 从修正模式中提取新信息
        for (pattern in correctionPatterns) {
            val corrMatch = pattern.find(message) ?: continue
            return corrMatch.groupValues.lastOrNull()
        }
        return null
    }
}

// === 摘要压缩检测 ===

object SummarizeDetector {
    
    /**
     * 检测是否应该触发摘要压缩
     * 
     * 条件：同一实体关联的记忆数量超过阈值
     */
    fun shouldSummarize(
        entityId: String,
        relatedMemories: List<ExistingMemory>,
        threshold: Int = 5
    ): SummarizeDecision? {
        if (relatedMemories.size < threshold) return null
        
        // 按时间排序，保留最新的 + 最重要的
        val sorted = relatedMemories.sortedByDescending { it.lastAccessTime }
        val toKeep = sorted.take(2)  // 最新两条保留
        val toMerge = sorted.drop(2)
        
        return SummarizeDecision(
            entityId = entityId,
            memoriesToMerge = toMerge.map { it.id },
            memoriesToKeep = toKeep.map { it.id },
            estimatedReduction = toMerge.size
        )
    }
}

// === 完整提取流程 (v2) ===

class ContextAwareExtractorV2(
    private val skipThreshold: Double = 0.85,
    private val updateThreshold: Double = 0.60,
    private val minConfidence: Double = 0.3
) {
    /**
     * 完整提取决策
     * 
     * 输出包含 CRUD 全部操作：
     * - NEW: 全新信息
     * - UPDATE: 部分匹配，需要更新
     * - DELETE: 用户显式否定
     * - SKIP: 重复或低置信
     * - SUMMARIZE: 信息过多需要压缩
     */
    fun extract(
        message: String,
        extractedFacts: List<ExtractedFact>,
        existingMemories: List<ExistingMemory>
    ): ExtractionPlan {
        val actions = mutableListOf<ExtractionAction>()
        
        // 1. 否定检测 (DELETE)
        val negations = NegationDetector.detect(message, existingMemories)
        for (neg in negations) {
            when (neg.negationType) {
                NegationType.FULL_NEGATION -> {
                    // 检查 governance — Constitutional 不可删除
                    val targetMem = existingMemories.find { it.id == neg.targetMemoryId }
                    if (targetMem?.layer != GovernanceLayer.CONSTITUTIONAL) {
                        actions.add(ExtractionAction(
                            action = MemoryAction.DELETE,
                            targetMemoryId = neg.targetMemoryId,
                            content = neg.targetContent,
                            confidence = neg.confidence,
                            reason = "User explicitly negated: ${neg.targetContent}"
                        ))
                    }
                }
                NegationType.CORRECTION -> {
                    // 修正 = DELETE old + STORE new
                    val targetMem = existingMemories.find { it.id == neg.targetMemoryId }
                    if (targetMem?.layer != GovernanceLayer.CONSTITUTIONAL) {
                        actions.add(ExtractionAction(
                            action = MemoryAction.UPDATE,
                            targetMemoryId = neg.targetMemoryId,
                            content = neg.extractedReplacement ?: message,
                            confidence = neg.confidence,
                            reason = "User corrected: ${neg.targetContent} → ${neg.extractedReplacement}"
                        ))
                    }
                }
            }
        }
        
        // 2. 常规提取 (NEW/UPDATE/SKIP)
        for (fact in extractedFacts) {
            // 跳过已被否定处理的
            if (negations.any { it.targetMemoryId == fact.matchingId }) continue
            
            val action = when {
                fact.matchScore >= skipThreshold -> ExtractionAction(
                    action = MemoryAction.SKIP,
                    targetMemoryId = fact.matchingId,
                    confidence = fact.matchScore,
                    reason = "Duplicate (${fact.matchScore})"
                )
                fact.matchScore >= updateThreshold && fact.matchingId != null -> ExtractionAction(
                    action = MemoryAction.UPDATE,
                    targetMemoryId = fact.matchingId,
                    content = fact.content,
                    confidence = fact.confidence,
                    reason = "Partial match (${fact.matchScore})"
                )
                fact.confidence >= minConfidence -> ExtractionAction(
                    action = MemoryAction.STORE,
                    content = fact.content,
                    confidence = fact.confidence,
                    reason = "New fact"
                )
                else -> ExtractionAction(
                    action = MemoryAction.SKIP,
                    confidence = 1.0 - fact.confidence,
                    reason = "Low confidence (${fact.confidence})"
                )
            }
            actions.add(action)
        }
        
        return ExtractionPlan(
            actions = actions,
            negationsDetected = negations.size,
            factsProcessed = extractedFacts.size
        )
    }
}

// === Data Classes ===

data class ExistingMemory(
    val id: String,
    val content: String,
    val layer: GovernanceLayer = GovernanceLayer.OPERATIONAL,
    val lastAccessTime: Long = System.currentTimeMillis()
)

data class ExtractedFact(
    val content: String,
    val confidence: Double,
    val matchingId: String? = null,
    val matchScore: Double = 0.0
)

enum class NegationType {
    FULL_NEGATION,   // 完全否定 (DELETE)
    CORRECTION       // 修正 (UPDATE)
}

data class NegationResult(
    val targetMemoryId: String,
    val targetContent: String,
    val negationType: NegationType,
    val confidence: Double,
    val extractedReplacement: String? = null
)

data class SummarizeDecision(
    val entityId: String,
    val memoriesToMerge: List<String>,
    val memoriesToKeep: List<String>,
    val estimatedReduction: Int
)

data class ExtractionAction(
    val action: MemoryAction,
    val targetMemoryId: String? = null,
    val content: String? = null,
    val confidence: Double = 1.0,
    val reason: String = ""
)

data class ExtractionPlan(
    val actions: List<ExtractionAction>,
    val negationsDetected: Int,
    val factsProcessed: Int
) {
    val storeCount get() = actions.count { it.action == MemoryAction.STORE }
    val updateCount get() = actions.count { it.action == MemoryAction.UPDATE }
    val deleteCount get() = actions.count { it.action == MemoryAction.DELETE }
    val skipCount get() = actions.count { it.action == MemoryAction.SKIP }
}
