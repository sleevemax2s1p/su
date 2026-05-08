package com.memory.gate

import com.memory.store.KnowledgeStore
import com.memory.llm.DeepSeekClient

/**
 * 苏大姐 v7.1 — 知识准入控制器 (A-MAC 五维模型实现)
 * 
 * 灵感来源: ICLR 2026 MemAgent Workshop - A-MAC (Adaptive Memory Admission Control)
 * 
 * 五维评分:
 * - Utility (U): 对未来任务的潜在有用性
 * - Confidence (C): 事实可靠性 (防幻觉)
 * - Novelty (N): 相对已有记忆的新颖度
 * - Recency (R): 时间新近性
 * - Type Prior (T): 内容类型先验权重
 * 
 * 相比旧版 KnowledgeGate (单纯 importance 评分):
 * - 新增 Novelty 检测 (解决重复存储问题)
 * - 新增 Confidence 检测 (防止幻觉入库)
 * - 保留 Utility 评估 (对应旧的 importance)
 * - 统一为可学习的加权组合
 */
class AdaptiveMemoryGate(
    private val store: KnowledgeStore,
    private val llm: DeepSeekClient? = null
) {
    // 各维度权重 (可通过反馈学习调整)
    private var weights = DimensionWeights(
        utility = 0.30,
        confidence = 0.20,
        novelty = 0.25,
        recency = 0.10,
        typePrior = 0.15
    )
    
    // 准入阈值 (低于此值拒绝入库)
    private var admissionThreshold = 0.45
    
    // Novelty 阈值 (相似度超过此值视为冗余)
    private val redundancyThreshold = 0.85

    /**
     * 五维评估: 决定一条候选记忆是否应进入长期存储
     * 
     * @param candidate 候选记忆内容
     * @param candidateEmbedding 候选记忆的 embedding (可选, 若无则内部计算)
     * @param existingEmbeddings 已有记忆的 embedding 列表 (用于 novelty 计算)
     * @param source 来源信息 (用于 confidence 判断)
     * @return AdmissionDecision 包含各维度分数和最终决策
     */
    suspend fun evaluate(
        candidate: String,
        candidateEmbedding: FloatArray? = null,
        existingEmbeddings: List<FloatArray> = emptyList(),
        source: MemorySource = MemorySource.CONVERSATION,
        timestamp: Long = System.currentTimeMillis()
    ): AdmissionDecision {
        
        // === Dimension 1: Utility ===
        val utilityScore = evaluateUtility(candidate)
        
        // === Dimension 2: Confidence ===
        val confidenceScore = evaluateConfidence(candidate, source)
        
        // === Dimension 3: Novelty ===
        val noveltyScore = evaluateNovelty(candidateEmbedding, existingEmbeddings)
        
        // === Dimension 4: Recency ===
        val recencyScore = evaluateRecency(timestamp)
        
        // === Dimension 5: Type Prior ===
        val typePriorScore = evaluateTypePrior(candidate)
        
        // Weighted combination
        val finalScore = weights.utility * utilityScore +
                        weights.confidence * confidenceScore +
                        weights.novelty * noveltyScore +
                        weights.recency * recencyScore +
                        weights.typePrior * typePriorScore
        
        // 快速拒绝: 如果 novelty 极低 (近乎重复), 直接拒绝
        val isRedundant = noveltyScore < 0.15
        val isTrivial = utilityScore < 0.3 && typePriorScore < 0.75
        val isUnreliable = confidenceScore < 0.3
        
        val admitted = !isRedundant && !isTrivial && !isUnreliable && finalScore >= admissionThreshold
        
        return AdmissionDecision(
            admitted = admitted,
            finalScore = finalScore,
            scores = DimensionScores(
                utility = utilityScore,
                confidence = confidenceScore,
                novelty = noveltyScore,
                recency = recencyScore,
                typePrior = typePriorScore
            ),
            rejectionReason = when {
                isTrivial -> RejectionReason.TRIVIAL
                isUnreliable -> RejectionReason.LOW_CONFIDENCE
                isRedundant -> RejectionReason.REDUNDANT
                finalScore < admissionThreshold -> RejectionReason.BELOW_THRESHOLD
                else -> null
            }
        )
    }

    /**
     * Utility: 对未来对话的潜在有用性
     * 
     * 快速规则 (不需要 LLM):
     * - 包含人名+属性 → 高 (0.9)
     * - 包含时间节点+事件 → 高 (0.85)
     * - 包含偏好/习惯 → 高 (0.8)
     * - 纯情绪表达 → 中 (0.5)
     * - 过于泛化的内容 → 低 (0.3)
     */
    private fun evaluateUtility(content: String): Double {
        val patterns = listOf(
            // 人物+属性: "小明的女朋友叫小红"
            Regex("(\\w+)的(名字|女朋友|男朋友|老婆|老公|工作|公司|学校|年龄|生日|爱好|宠物)") to 0.9,
            // 具体事件+时间: "下周一要面试"
            Regex("(明天|后天|下周|下个月|今天|昨天|上周).{0,10}(要|会|打算|计划|准备)") to 0.85,
            // 偏好: "喜欢/不喜欢/讨厌/最爱"
            Regex("(喜欢|不喜欢|讨厌|最爱|偏好|习惯).{1,20}") to 0.8,
            // 重要状态变化: "换了工作/分手了/搬家了"
            Regex("(换了|辞了|分手|离婚|搬|毕业|入职|结婚|怀孕|生了)") to 0.85,
            // 纯情绪: "好累/开心/难过"
            Regex("^(好累|好开心|好难过|心情不好|无聊|烦死了)$") to 0.5,
        )
        
        for ((pattern, score) in patterns) {
            if (pattern.containsMatchIn(content)) return score
        }
        
        // 默认: 基于长度和信息密度的启发式
        return when {
            content.length < 5 -> 0.1   // 太短, 信息量不足
            content.length > 100 -> 0.6  // 长内容通常有信息量
            else -> 0.5                  // 中等
        }
    }

    /**
     * Confidence: 事实可靠性
     * 
     * 来源越可靠, confidence 越高:
     * - 用户直接陈述 → 0.95
     * - 对话推断 → 0.7
     * - AI 生成摘要 → 0.6
     * - 外部来源 → 0.5
     * 
     * 额外降分:
     * - 包含不确定词("可能", "大概", "好像") → -0.2
     * - 包含矛盾信号("但是", "不过", "也不一定") → -0.1
     */
    private fun evaluateConfidence(content: String, source: MemorySource): Double {
        val baseConfidence = when (source) {
            MemorySource.USER_DIRECT -> 0.95
            MemorySource.CONVERSATION -> 0.7
            MemorySource.AI_SUMMARY -> 0.6
            MemorySource.EXTERNAL -> 0.5
        }
        
        var penalty = 0.0
        val uncertaintyWords = listOf("可能", "大概", "好像", "似乎", "也许", "不确定", "approximately", "maybe", "perhaps")
        val contradictionWords = listOf("但是", "不过", "也不一定", "不太确定", "however", "but")
        
        if (uncertaintyWords.any { content.contains(it) }) penalty += 0.2
        if (contradictionWords.any { content.contains(it) }) penalty += 0.1
        
        return (baseConfidence - penalty).coerceIn(0.1, 1.0)
    }

    /**
     * Novelty: 相对已有记忆的新颖度
     * 
     * 计算方式: 1 - max(cosine_similarity(candidate, existing_memories))
     * 
     * 如果没有已有记忆的 embedding → 默认新颖度为 0.8 (大概率是新信息)
     * 如果最大相似度 > redundancyThreshold → 几乎是重复, novelty ≈ 0
     */
    private fun evaluateNovelty(
        candidateEmbedding: FloatArray?,
        existingEmbeddings: List<FloatArray>
    ): Double {
        if (candidateEmbedding == null || existingEmbeddings.isEmpty()) {
            return 0.8  // 没有比较基准, 假设为新信息
        }
        
        val maxSimilarity = existingEmbeddings.maxOf { existing ->
            cosineSimilarity(candidateEmbedding, existing)
        }
        
        val novelty = 1.0 - maxSimilarity
        
        // 如果高度相似但不完全相同, 可能是更新而非重复
        // 这里留一个小窗口: similarity 在 0.80-0.85 之间视为"更新"而非"冗余"
        return if (maxSimilarity in 0.80..redundancyThreshold) {
            novelty * 1.5  // 轻微 boost, 可能是有价值的更新
        } else {
            novelty
        }
    }

    /**
     * Recency: 时间新近性
     * 
     * 使用指数衰减: score = exp(-λ * hours_ago)
     * λ = 0.01 → 72小时后衰减到约 50%
     */
    private fun evaluateRecency(timestamp: Long): Double {
        val hoursAgo = (System.currentTimeMillis() - timestamp) / (1000.0 * 3600)
        val lambda = 0.01
        return Math.exp(-lambda * hoursAgo).coerceIn(0.1, 1.0)
    }

    /**
     * Type Prior: 内容类型先验
     * 
     * 研究表明这是 A-MAC 中最强的信号。
     * 不同类型的信息有不同的长期价值:
     * - 身份/关系信息 → 最高 (永久有效)
     * - 习惯/偏好 → 高 (长期有效)
     * - 计划/时间事件 → 中高 (短期高价值)
     * - 情绪状态 → 中 (临时性)
     * - 闲聊/寒暄 → 低 (几乎无长期价值)
     */
    private fun evaluateTypePrior(content: String): Double {
        return when {
            // 身份/关系: 最高优先级
            content.matches(Regex(".*(名字|叫|是我的|女朋友|男朋友|老婆|老公|爸|妈|儿子|女儿|朋友|同事).*")) -> 0.95
            // 偏好/习惯
            content.matches(Regex(".*(喜欢|讨厌|习惯|总是|从不|每天|always|never|prefer).*")) -> 0.85
            // 计划/事件
            content.matches(Regex(".*(要去|打算|计划|准备|will|plan|tomorrow|下周|明天).*")) -> 0.75
            // 事实/知识
            content.matches(Regex(".*(在.*工作|住在|毕业于|学的是|works at|lives in).*")) -> 0.9
            // 情绪表达
            content.matches(Regex(".*(开心|难过|焦虑|压力|兴奋|angry|sad|happy|stressed).*")) -> 0.5
            // 默认
            else -> 0.4
        }
    }

    /**
     * 余弦相似度计算
     */
    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Double {
        if (a.size != b.size) return 0.0
        var dotProduct = 0.0
        var normA = 0.0
        var normB = 0.0
        for (i in a.indices) {
            dotProduct += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        val denominator = Math.sqrt(normA) * Math.sqrt(normB)
        return if (denominator == 0.0) 0.0 else dotProduct / denominator
    }

    /**
     * 通过反馈更新权重 (简化版在线学习)
     * 
     * 当用户表示"你怎么忘了"(false negative) 或"你记错了"(false positive) 时,
     * 调整相应维度的权重。
     */
    fun updateWeightsFromFeedback(feedback: AdmissionFeedback) {
        val learningRate = 0.05
        
        when (feedback.type) {
            FeedbackType.FALSE_NEGATIVE -> {
                // 应该记住但没记住 → 降低准入阈值, 提高 utility 权重
                admissionThreshold = (admissionThreshold - learningRate).coerceIn(0.3, 0.7)
                weights = weights.copy(utility = (weights.utility + learningRate).coerceAtMost(0.5))
                normalizeWeights()
            }
            FeedbackType.FALSE_POSITIVE -> {
                // 记住了不该记的 → 提高准入阈值
                admissionThreshold = (admissionThreshold + learningRate).coerceIn(0.3, 0.7)
            }
            FeedbackType.REDUNDANT_STORED -> {
                // 存了重复的 → 提高 novelty 权重
                weights = weights.copy(novelty = (weights.novelty + learningRate).coerceAtMost(0.4))
                normalizeWeights()
            }
        }
    }

    private fun normalizeWeights() {
        val total = weights.utility + weights.confidence + weights.novelty + 
                   weights.recency + weights.typePrior
        weights = DimensionWeights(
            utility = weights.utility / total,
            confidence = weights.confidence / total,
            novelty = weights.novelty / total,
            recency = weights.recency / total,
            typePrior = weights.typePrior / total
        )
    }

    fun getStats(): GateStats {
        return GateStats(
            weights = weights,
            threshold = admissionThreshold,
            redundancyThreshold = redundancyThreshold
        )
    }
}

// === Data Classes ===

data class DimensionWeights(
    val utility: Double,
    val confidence: Double,
    val novelty: Double,
    val recency: Double,
    val typePrior: Double
)

data class DimensionScores(
    val utility: Double,
    val confidence: Double,
    val novelty: Double,
    val recency: Double,
    val typePrior: Double
)

data class AdmissionDecision(
    val admitted: Boolean,
    val finalScore: Double,
    val scores: DimensionScores,
    val rejectionReason: RejectionReason?
)

enum class RejectionReason {
    REDUNDANT,          // novelty 过低 (重复信息)
    TRIVIAL,            // 信息密度过低
    BELOW_THRESHOLD,    // 综合分数不够
    LOW_CONFIDENCE,     // 信息不可靠
    BLOCKED_BY_SAFETY   // 安全过滤器拦截
}

enum class MemorySource {
    USER_DIRECT,    // 用户直接陈述
    CONVERSATION,   // 对话推断
    AI_SUMMARY,     // AI 生成摘要
    EXTERNAL        // 外部来源
}

data class AdmissionFeedback(
    val type: FeedbackType,
    val memoryContent: String? = null,
    val context: String? = null
)

enum class FeedbackType {
    FALSE_NEGATIVE,     // 该记没记
    FALSE_POSITIVE,     // 不该记却记了
    REDUNDANT_STORED    // 存了重复的
}

data class GateStats(
    val weights: DimensionWeights,
    val threshold: Double,
    val redundancyThreshold: Double
)
