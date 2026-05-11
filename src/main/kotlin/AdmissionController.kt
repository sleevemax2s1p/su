package com.memory.admission

import com.memory.store.MemoryEntry
import com.memory.store.MemorySource

/**
 * Adaptive Memory Admission Controller (受 A-MAC, arXiv:2603.04549 启发)
 *
 * 在 ADD 之前增加质量门控：
 * - 不是所有对话内容都值得记忆
 * - 5维度评分：relevance, novelty, importance, actionability, user-specificity
 * - 与 ADD-only 兼容：决定"是否ADD"，不影响"ADD了不改"
 *
 * 为什么需要准入控制：
 * 1. 用户说"嗯"、"好的"、"哈哈" → 不需要记忆
 * 2. 重复信息（"我住在北京"已存过100次）→ 降低准入概率
 * 3. 高价值信息（生日、过敏原、重要偏好）→ 一定要记
 *
 * 效果预期（参考 A-MAC）：
 * - LoCoMo 延迟降低 ~31%
 * - 信噪比提升 → 检索准确率提升
 * - 存储占用降低 40-60%
 */
class AdmissionController(
    // 准入阈值：综合分低于此值不入库
    private val admissionThreshold: Double = 0.3,
    // 各维度权重
    private val relevanceWeight: Double = 0.2,
    private val noveltyWeight: Double = 0.25,
    private val importanceWeight: Double = 0.3,
    private val actionabilityWeight: Double = 0.1,
    private val specificityWeight: Double = 0.15,
    // 高价值类别（强制准入）
    private val highValuePatterns: List<Regex> = defaultHighValuePatterns()
) {
    
    /**
     * 评估一条候选记忆是否应该被 ADD
     *
     * @param content 待评估的内容
     * @param source 来源类型
     * @param existingMemories 已有记忆（用于 novelty 计算）
     * @return AdmissionDecision
     */
    fun evaluate(
        content: String,
        source: MemorySource,
        existingMemories: List<MemoryEntry> = emptyList(),
        conversationContext: String = ""
    ): AdmissionDecision {
        // 强制准入：高价值内容（过敏原、重要日期、身份信息等）
        if (isHighValue(content)) {
            return AdmissionDecision(
                admitted = true,
                score = 1.0,
                reason = "high_value_pattern",
                scores = AdmissionScores(1.0, 1.0, 1.0, 1.0, 1.0)
            )
        }
        
        // 强制拒绝：明显无信息量的内容
        if (isLowSignal(content)) {
            return AdmissionDecision(
                admitted = false,
                score = 0.0,
                reason = "low_signal",
                scores = AdmissionScores(0.0, 0.0, 0.0, 0.0, 0.0)
            )
        }
        
        // 5维度评分
        val scores = AdmissionScores(
            relevance = computeRelevance(content, conversationContext),
            novelty = computeNovelty(content, existingMemories),
            importance = computeImportance(content),
            actionability = computeActionability(content),
            specificity = computeSpecificity(content, source)
        )
        
        // 加权综合分
        val finalScore = (
            scores.relevance * relevanceWeight +
            scores.novelty * noveltyWeight +
            scores.importance * importanceWeight +
            scores.actionability * actionabilityWeight +
            scores.specificity * specificityWeight
        )
        
        return AdmissionDecision(
            admitted = finalScore >= admissionThreshold,
            score = finalScore,
            reason = if (finalScore >= admissionThreshold) "admitted" else "below_threshold",
            scores = scores
        )
    }
    
    /**
     * 批量评估（对话结束后一次性评估所有提取的事实）
     */
    fun evaluateBatch(
        candidates: List<AdmissionCandidate>,
        existingMemories: List<MemoryEntry> = emptyList()
    ): List<AdmissionDecision> {
        return candidates.map { c ->
            evaluate(c.content, c.source, existingMemories, c.context)
        }
    }
    
    // === 各维度评分函数 ===
    
    /**
     * Relevance: 内容与对话上下文的相关性
     * 高相关 = 值得记忆
     */
    private fun computeRelevance(content: String, context: String): Double {
        if (context.isEmpty()) return 0.5 // neutral without context
        
        val contentChars = content.toSet()
        val contextChars = context.toSet()
        val overlap = contentChars.intersect(contextChars).size
        val total = contentChars.union(contextChars).size
        
        return if (total == 0) 0.0 else (overlap.toDouble() / total).coerceIn(0.0, 1.0)
    }
    
    /**
     * Novelty: 相对于已有记忆的新颖程度
     * 如果已有高度相似的记忆 → 低 novelty
     */
    private fun computeNovelty(content: String, existing: List<MemoryEntry>): Double {
        if (existing.isEmpty()) return 1.0 // 全新用户，一切都是新的
        
        // 计算与最相似已有记忆的距离
        var maxSimilarity = 0.0
        for (mem in existing) {
            val sim = charJaccard(content, mem.content)
            if (sim > maxSimilarity) maxSimilarity = sim
        }
        
        // novelty = 1 - max_similarity
        return (1.0 - maxSimilarity).coerceIn(0.0, 1.0)
    }
    
    /**
     * Importance: 内容的重要程度（基于模式匹配）
     */
    private fun computeImportance(content: String): Double {
        var score = 0.3 // baseline
        
        // 重要信号
        if (content.length > 20) score += 0.1 // 较长的陈述通常更有信息量
        if (containsPersonalInfo(content)) score += 0.3
        if (containsPreference(content)) score += 0.2
        if (containsEvent(content)) score += 0.15
        if (containsRelationship(content)) score += 0.2
        
        return score.coerceIn(0.0, 1.0)
    }
    
    /**
     * Actionability: 是否可用于未来行动
     * "明天要面试" → 高（可以提前关心）
     * "今天天气不错" → 低（无后续行动）
     */
    private fun computeActionability(content: String): Double {
        var score = 0.2 // baseline
        
        if (containsFutureEvent(content)) score += 0.5
        if (containsNeed(content)) score += 0.3
        if (containsGoal(content)) score += 0.3
        
        return score.coerceIn(0.0, 1.0)
    }
    
    /**
     * User-specificity: 对用户的特异性
     * "我过敏" → 高特异性
     * "今天星期一" → 低特异性（通用知识）
     */
    private fun computeSpecificity(content: String, source: MemorySource): Double {
        var score = when (source) {
            MemorySource.USER -> 0.5   // 用户说的通常比较特异
            MemorySource.AGENT -> 0.3  // Agent 说的可能是总结
            MemorySource.SYSTEM -> 0.2 // 系统事件通常通用
        }
        
        // 包含第一人称 → 更特异
        if (Regex("[我]").containsMatchIn(content)) score += 0.2
        // 包含具体名词/地点 → 更特异
        if (Regex("[\\u4e00-\\u9fa5]{2,4}(?:市|区|街|路|公司|大学|医院)").containsMatchIn(content)) score += 0.15
        // 包含具体数字/日期 → 更特异
        if (Regex("\\d{4}|\\d+月|\\d+号|\\d+日|\\d+岁").containsMatchIn(content)) score += 0.15
        
        return score.coerceIn(0.0, 1.0)
    }
    
    // === Pattern Detection ===
    
    private fun isHighValue(content: String): Boolean {
        return highValuePatterns.any { it.containsMatchIn(content) }
    }
    
    private fun isLowSignal(content: String): Boolean {
        // 内容太短（<3字）且无实质信息
        if (content.length <= 2) return true
        
        // 纯语气词/确认
        val lowSignalPatterns = listOf(
            Regex("^(嗯|哦|好的?|行|ok|OK|是的?|对|没有?|不是|哈+|嘿|呵|啊|噢|呃)$"),
            Regex("^(谢谢|感谢|好吧|随便|都行|无所谓)$"),
            Regex("^(\\?|？|!|！|\\.|。)+$")
        )
        
        return lowSignalPatterns.any { it.matches(content.trim()) }
    }
    
    private fun containsPersonalInfo(content: String): Boolean {
        return Regex("(叫|名字|姓|年龄|岁|生日|生于|出生|电话|邮箱|地址|住|家|老家)").containsMatchIn(content)
    }
    
    private fun containsPreference(content: String): Boolean {
        return Regex("(喜欢|讨厌|爱好|偏好|最爱|不喜欢|受不了|过敏|忌口)").containsMatchIn(content)
    }
    
    private fun containsEvent(content: String): Boolean {
        return Regex("(面试|会议|约会|旅行|搬家|毕业|入职|结婚|分手|手术)").containsMatchIn(content)
    }
    
    private fun containsRelationship(content: String): Boolean {
        return Regex("(朋友|同事|老板|女友|男友|老婆|老公|爸|妈|哥|姐|弟|妹|儿子|女儿)").containsMatchIn(content)
    }
    
    private fun containsFutureEvent(content: String): Boolean {
        return Regex("(明天|下周|下个月|之后|打算|计划|准备|要去|想去|约了)").containsMatchIn(content)
    }
    
    private fun containsNeed(content: String): Boolean {
        return Regex("(需要|想要|希望|能不能|可以帮|帮我|提醒我)").containsMatchIn(content)
    }
    
    private fun containsGoal(content: String): Boolean {
        return Regex("(目标|希望能|想成为|努力|奋斗|梦想)").containsMatchIn(content)
    }
    
    private fun charJaccard(a: String, b: String): Double {
        val sa = a.toSet()
        val sb = b.toSet()
        val intersection = sa.intersect(sb).size
        val union = sa.union(sb).size
        return if (union == 0) 0.0 else intersection.toDouble() / union
    }
    
    companion object {
        fun defaultHighValuePatterns(): List<Regex> = listOf(
            Regex("过敏|忌口|禁忌"),              // 健康安全
            Regex("生日.*(\\d+月|\\d+号)"),       // 重要日期
            Regex("(密码|账号|手机号)"),           // 敏感信息（需加密存储）
            Regex("(紧急联系人|ICE)"),            // 紧急信息
            Regex("(怀孕|手术|住院)")             // 重大生活事件
        )
    }
}

// === Data Classes ===

data class AdmissionDecision(
    val admitted: Boolean,
    val score: Double,
    val reason: String,
    val scores: AdmissionScores
)

data class AdmissionScores(
    val relevance: Double,
    val novelty: Double,
    val importance: Double,
    val actionability: Double,
    val specificity: Double
)

data class AdmissionCandidate(
    val content: String,
    val source: MemorySource,
    val context: String = ""
)
