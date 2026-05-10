package com.memory.context

import com.memory.retrieval.RankedResult
import com.memory.governance.GovernanceLayer
import com.memory.provenance.TrustLevel
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min

/**
 * 自适应上下文选择器 (Adaptive Context Selector)
 * 
 * 基于 MemAct (arXiv:2510.12635) 的核心发现：
 * "MemAct-RL-14B 匹配 16x 更大模型精度，同时上下文长度减少 51%"
 * 
 * 关键洞察：不是检索越多注入越好。过多的记忆会引入噪声，
 * 导致 LLM 注意力分散、生成质量下降。应该 adaptive 地决定：
 * 1. 注入几条记忆 (dynamic K)
 * 2. 哪些记忆值得注入 (relevance threshold)
 * 3. 如何排序注入顺序 (priority ordering)
 * 
 * 设计哲学（正交维度）：
 * - 上下文选择是查询时的视图函数
 * - 不改变底层存储/检索结果
 * - 根据查询复杂度、记忆质量、token 预算动态调整
 * 
 * 与 v8 的区别：
 * - v8: 固定 take(8)
 * - v9: 基于 score gap + query complexity + budget 动态决定
 */
class AdaptiveContextSelector(
    private val maxTokenBudget: Int = 1200,        // 记忆部分最大 token 预算
    private val minRelevanceScore: Double = 0.005,  // 最低相关性阈值
    private val scoreGapThreshold: Double = 0.3,    // 分数断崖检测阈值
    private val avgTokensPerMemory: Int = 50,       // 平均每条记忆的 token 数
    private val constitutionalAlwaysInclude: Boolean = true  // Constitutional 记忆始终注入
) {
    
    /**
     * 主选择入口
     * 
     * @param candidates 检索排序后的候选记忆 (已按 relevance 降序)
     * @param queryComplexity 查询复杂度 (由外部评估器提供)
     * @param conversationContext 当前对话上下文信息
     * @return 选择结果：要注入的记忆列表 + 元信息
     */
    fun select(
        candidates: List<MemoryCandidate>,
        queryComplexity: QueryComplexity = QueryComplexity.MEDIUM,
        conversationContext: ConversationContext = ConversationContext()
    ): SelectionResult {
        if (candidates.isEmpty()) {
            return SelectionResult(selected = emptyList(), reason = "No candidates")
        }
        
        // Step 1: 确定动态 K (基于 token 预算和查询复杂度)
        val budgetK = maxTokenBudget / avgTokensPerMemory
        val complexityK = when (queryComplexity) {
            QueryComplexity.SIMPLE -> 3      // 简单问题不需要太多上下文
            QueryComplexity.MEDIUM -> 6
            QueryComplexity.COMPLEX -> 10    // 复杂问题需要更多背景
            QueryComplexity.MULTI_HOP -> 12  // 多跳推理需要完整链路
        }
        val dynamicK = min(budgetK, complexityK)
        
        // Step 2: Score gap 检测 — 找到 relevance 断崖
        val gapCutoff = detectScoreGap(candidates)
        
        // Step 3: 最低阈值过滤
        val thresholdCutoff = candidates.indexOfLast { it.score >= minRelevanceScore } + 1
        
        // Step 4: 综合确定最终 K
        val finalK = min(dynamicK, min(gapCutoff, thresholdCutoff))
        
        // Step 5: 选择 + Constitutional 保底
        val selected = mutableListOf<MemoryCandidate>()
        val reasons = mutableListOf<String>()
        
        // 先加入 Constitutional（无论 score 如何）
        if (constitutionalAlwaysInclude) {
            val constitutionals = candidates.filter { it.layer == GovernanceLayer.CONSTITUTIONAL }
            selected.addAll(constitutionals)
            if (constitutionals.isNotEmpty()) {
                reasons.add("${constitutionals.size} constitutional memories always included")
            }
        }
        
        // 再加入 top-K 非 Constitutional
        val nonConst = candidates.filter { it.layer != GovernanceLayer.CONSTITUTIONAL }
        val remaining = finalK - selected.size
        if (remaining > 0) {
            selected.addAll(nonConst.take(remaining))
        }
        
        // Step 6: 优先级排序 (注入顺序影响 LLM 注意力)
        val ordered = priorityOrder(selected, conversationContext)
        
        return SelectionResult(
            selected = ordered,
            totalCandidates = candidates.size,
            selectedCount = ordered.size,
            dynamicK = dynamicK,
            scoreGapK = gapCutoff,
            thresholdK = thresholdCutoff,
            finalK = finalK,
            estimatedTokens = ordered.size * avgTokensPerMemory,
            reason = buildReason(ordered, reasons, candidates.size)
        )
    }
    
    /**
     * Score Gap 检测
     * 
     * 找到检索分数的"断崖"——当相邻结果的分数差超过平均差的 N 倍时,
     * 说明后面的结果相关性急剧下降，不应注入。
     * 
     * 例如: [0.95, 0.92, 0.88, 0.40, 0.35, 0.30]
     *                              ↑ gap = 0.48, 断崖在此
     */
    private fun detectScoreGap(candidates: List<MemoryCandidate>): Int {
        if (candidates.size <= 1) return candidates.size
        
        val scores = candidates.map { it.score }
        val diffs = mutableListOf<Double>()
        
        for (i in 1 until scores.size) {
            diffs.add(scores[i - 1] - scores[i])
        }
        
        if (diffs.isEmpty()) return candidates.size
        
        val avgDiff = diffs.average()
        val threshold = avgDiff * (1.0 + scoreGapThreshold)
        
        // 找第一个超过阈值的 gap
        for (i in diffs.indices) {
            if (diffs[i] > threshold && diffs[i] > 0.1) {  // 绝对值也要达标
                return i + 1  // 断崖前的数量
            }
        }
        
        return candidates.size  // 没有明显断崖
    }
    
    /**
     * 优先级排序 (注入 prompt 的顺序)
     * 
     * 研究表明 LLM 对 prompt 中间位置的信息注意力最弱 (Lost in the Middle)。
     * 策略: 最重要的放最前和最后，中等重要的放中间。
     */
    private fun priorityOrder(
        memories: List<MemoryCandidate>,
        context: ConversationContext
    ): List<MemoryCandidate> {
        if (memories.size <= 3) return memories
        
        // 按重要性排序
        val sorted = memories.sortedByDescending { candidate ->
            var priority = candidate.score
            
            // Constitutional boost
            if (candidate.layer == GovernanceLayer.CONSTITUTIONAL) priority += 0.5
            
            // High trust boost
            if (candidate.trustLevel == TrustLevel.HIGH) priority += 0.1
            
            // Recency boost (最近对话提到的优先)
            if (context.recentTopics.any { topic -> candidate.content.contains(topic) }) {
                priority += 0.2
            }
            
            priority
        }
        
        // "Sandwich" ordering: 重要的在首尾，次重要的在中间
        val result = mutableListOf<MemoryCandidate>()
        val n = sorted.size
        
        // 前半放奇数位 (最重要的)
        for (i in 0 until n step 2) {
            result.add(sorted[i])
        }
        // 后半放偶数位 (次重要的，逆序)
        val middle = mutableListOf<MemoryCandidate>()
        for (i in 1 until n step 2) {
            middle.add(sorted[i])
        }
        result.addAll(middle.reversed())
        
        return result
    }
    
    private fun buildReason(
        selected: List<MemoryCandidate>,
        extraReasons: List<String>,
        totalCandidates: Int
    ): String {
        val parts = mutableListOf<String>()
        parts.add("Selected ${selected.size}/$totalCandidates candidates")
        parts.addAll(extraReasons)
        return parts.joinToString("; ")
    }
    
    /**
     * 查询复杂度评估 (简化版)
     * 
     * 实际项目中可以用 LLM 评估或基于特征的分类器
     */
    companion object {
        fun assessQueryComplexity(query: String): QueryComplexity {
            // 多跳关键词
            val multiHopSignals = listOf("的", "在哪", "谁的", "工作的公司", "住在")
            val multiHopCount = multiHopSignals.count { it in query }
            
            // 复杂度指标
            val length = query.length
            val hasQuestion = query.contains("?") || query.contains("？") || 
                             query.contains("吗") || query.contains("呢")
            
            return when {
                multiHopCount >= 2 -> QueryComplexity.MULTI_HOP
                length > 50 || (hasQuestion && multiHopCount >= 1) -> QueryComplexity.COMPLEX
                length > 20 || hasQuestion -> QueryComplexity.MEDIUM
                else -> QueryComplexity.SIMPLE
            }
        }
    }
}

// === Data Classes ===

enum class QueryComplexity {
    SIMPLE,     // "今天天气好" → 最多3条
    MEDIUM,     // "你还记得上次吗" → 最多6条
    COMPLEX,    // "上次我们讨论的那个方案怎么样了" → 最多10条
    MULTI_HOP   // "我女朋友工作的公司在哪个城市" → 最多12条
}

data class MemoryCandidate(
    val id: String,
    val content: String,
    val score: Double,
    val layer: GovernanceLayer,
    val trustLevel: TrustLevel,
    val tokenEstimate: Int = 50,
    val retrievalSource: String = "hybrid"
)

data class ConversationContext(
    val recentTopics: List<String> = emptyList(),
    val turnCount: Int = 0,
    val userMood: String? = null
)

data class SelectionResult(
    val selected: List<MemoryCandidate>,
    val totalCandidates: Int = 0,
    val selectedCount: Int = 0,
    val dynamicK: Int = 0,
    val scoreGapK: Int = 0,
    val thresholdK: Int = 0,
    val finalK: Int = 0,
    val estimatedTokens: Int = 0,
    val reason: String = ""
) {
    val compressionRatio: Double
        get() = if (totalCandidates > 0) {
            1.0 - (selectedCount.toDouble() / totalCandidates)
        } else 0.0
    
    override fun toString(): String = """
        |=== Context Selection ===
        |Selected: $selectedCount / $totalCandidates (compression: ${String.format("%.0f%%", compressionRatio * 100)})
        |Dynamic K: $dynamicK, Gap K: $scoreGapK, Threshold K: $thresholdK → Final K: $finalK
        |Estimated tokens: $estimatedTokens / 1200 budget
        |Reason: $reason
    """.trimMargin()
}
