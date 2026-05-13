package com.memory.retrieval

/**
 * Temporal Reasoning Module (受 Mem0 v3 Temporal Reasoning 5月13日发布启发)
 *
 * 核心问题：
 * 传统 recency = f(now - memory_time) 固定以"现在"为参考点
 * 但用户查询可能指向不同时间：
 * - "我以前住在哪里" → 应该偏好旧记忆
 * - "下周有什么安排" → 应该偏好未来事件
 * - "去年发生了什么" → 应该偏好去年的记忆
 *
 * 解决方案：
 * 1. 解析查询中的时间表达式 → 生成 time anchor
 * 2. 用 |memory_time - anchor| 替代 |now - memory_time|
 * 3. 支持 reference_date 参数（可复现测试）
 *
 * 这是"查询时的视图函数"的完美实例：
 * 同一批存储数据，通过不同 time anchor 产生不同的时间视图
 */

/**
 * 时间推理器
 */
class TemporalReasoner(
    private val defaultHalfLifeDays: Double = 30.0,
    private val minScore: Double = 0.1
) {
    // === 时间模式 ===
    
    // 过去导向：应该偏好更旧的记忆
    private val pastPatterns = listOf(
        TemporalPattern(Regex("以前|之前|过去|曾经|原来"), AnchorType.PAST_VAGUE, 0.0),
        TemporalPattern(Regex("上周"), AnchorType.PAST_RELATIVE, -7.0),
        TemporalPattern(Regex("上个月"), AnchorType.PAST_RELATIVE, -30.0),
        TemporalPattern(Regex("去年"), AnchorType.PAST_RELATIVE, -365.0),
        TemporalPattern(Regex("前天"), AnchorType.PAST_RELATIVE, -2.0),
        TemporalPattern(Regex("昨天"), AnchorType.PAST_RELATIVE, -1.0),
        TemporalPattern(Regex("(\\d+)天前"), AnchorType.PAST_RELATIVE, 0.0), // dynamic
        TemporalPattern(Regex("(\\d+)个?月前"), AnchorType.PAST_RELATIVE, 0.0), // dynamic
        TemporalPattern(Regex("(\\d+)年前"), AnchorType.PAST_RELATIVE, 0.0),  // dynamic
        TemporalPattern(Regex("小时候|小的时候|年轻的时候"), AnchorType.PAST_DISTANT, 0.0),
    )
    
    // 现在导向：偏好最新（默认行为）
    private val presentPatterns = listOf(
        TemporalPattern(Regex("现在|目前|当前|此刻"), AnchorType.PRESENT, 0.0),
        TemporalPattern(Regex("最近|近期|这几天"), AnchorType.RECENT, -3.0), // last few days
        TemporalPattern(Regex("今天"), AnchorType.PRESENT, 0.0),
    )
    
    // 未来导向：偏好未来事件记忆
    private val futurePatterns = listOf(
        TemporalPattern(Regex("明天"), AnchorType.FUTURE_RELATIVE, 1.0),
        TemporalPattern(Regex("后天"), AnchorType.FUTURE_RELATIVE, 2.0),
        TemporalPattern(Regex("下周"), AnchorType.FUTURE_RELATIVE, 7.0),
        TemporalPattern(Regex("下个月"), AnchorType.FUTURE_RELATIVE, 30.0),
        TemporalPattern(Regex("明年"), AnchorType.FUTURE_RELATIVE, 365.0),
        TemporalPattern(Regex("(\\d+)天后"), AnchorType.FUTURE_RELATIVE, 0.0), // dynamic
        TemporalPattern(Regex("即将|马上|快要|将要"), AnchorType.FUTURE_NEAR, 1.0),
        TemporalPattern(Regex("以后|将来|未来"), AnchorType.FUTURE_VAGUE, 30.0),
    )
    
    /**
     * 分析查询，返回 TemporalContext
     */
    fun analyze(
        query: String,
        referenceDate: Long = System.currentTimeMillis()
    ): TemporalContext {
        val detectedAnchors = mutableListOf<TemporalAnchor>()
        
        // Check past patterns
        for (pattern in pastPatterns) {
            val match = pattern.regex.find(query) ?: continue
            val offsetDays = when {
                pattern.anchorType == AnchorType.PAST_VAGUE -> -90.0  // vague past ≈ 3 months ago
                pattern.anchorType == AnchorType.PAST_DISTANT -> -3650.0 // distant past
                pattern.offsetDays != 0.0 -> pattern.offsetDays
                else -> extractDynamicOffset(match, negative = true)
            }
            detectedAnchors.add(TemporalAnchor(
                anchorType = pattern.anchorType,
                anchorTimeMs = referenceDate + (offsetDays * 86400 * 1000).toLong(),
                confidence = 0.8,
                matchedText = match.value
            ))
        }
        
        // Check present patterns
        for (pattern in presentPatterns) {
            val match = pattern.regex.find(query) ?: continue
            val offsetDays = pattern.offsetDays
            detectedAnchors.add(TemporalAnchor(
                anchorType = pattern.anchorType,
                anchorTimeMs = referenceDate + (offsetDays * 86400 * 1000).toLong(),
                confidence = 0.9,
                matchedText = match.value
            ))
        }
        
        // Check future patterns
        for (pattern in futurePatterns) {
            val match = pattern.regex.find(query) ?: continue
            val offsetDays = when {
                pattern.offsetDays != 0.0 -> pattern.offsetDays
                else -> extractDynamicOffset(match, negative = false)
            }
            detectedAnchors.add(TemporalAnchor(
                anchorType = pattern.anchorType,
                anchorTimeMs = referenceDate + (offsetDays * 86400 * 1000).toLong(),
                confidence = 0.8,
                matchedText = match.value
            ))
        }
        
        // Select primary anchor (highest confidence, or default to NOW)
        val primaryAnchor = detectedAnchors.maxByOrNull { it.confidence }
        
        return TemporalContext(
            hasTemporalIntent = detectedAnchors.isNotEmpty(),
            primaryAnchor = primaryAnchor,
            allAnchors = detectedAnchors,
            referenceDate = referenceDate,
            temporalDirection = when (primaryAnchor?.anchorType) {
                AnchorType.PAST_VAGUE, AnchorType.PAST_RELATIVE, AnchorType.PAST_DISTANT -> TemporalDirection.PAST
                AnchorType.FUTURE_RELATIVE, AnchorType.FUTURE_NEAR, AnchorType.FUTURE_VAGUE -> TemporalDirection.FUTURE
                AnchorType.PRESENT, AnchorType.RECENT -> TemporalDirection.PRESENT
                null -> TemporalDirection.PRESENT
            }
        )
    }
    
    /**
     * 基于 temporal context 计算 recency score
     * 
     * 替代传统的 computeRecencyScore(memoryTimestamp, currentTime)
     * 新公式: recency = f(|memory_time - anchor|)
     */
    fun computeTemporalScore(
        memoryTimestamp: Long,
        temporalContext: TemporalContext
    ): Double {
        val anchor = temporalContext.primaryAnchor?.anchorTimeMs ?: temporalContext.referenceDate
        
        // Distance from anchor (absolute)
        val distanceMs = Math.abs(memoryTimestamp - anchor)
        val distanceDays = distanceMs.toDouble() / (86400.0 * 1000.0)
        
        // Exponential decay from anchor
        val decay = Math.exp(-0.693 * distanceDays / defaultHalfLifeDays)
        
        // Direction bonus: if query is about the past, slightly prefer older memories
        val directionBonus = when (temporalContext.temporalDirection) {
            TemporalDirection.PAST -> {
                // For past queries, memories BEFORE the anchor get a small bonus
                if (memoryTimestamp <= (temporalContext.primaryAnchor?.anchorTimeMs ?: Long.MAX_VALUE)) 0.1 else 0.0
            }
            TemporalDirection.FUTURE -> {
                // For future queries, memories AFTER now get a bonus
                if (memoryTimestamp > temporalContext.referenceDate) 0.1 else 0.0
            }
            TemporalDirection.PRESENT -> 0.0
        }
        
        return (decay + directionBonus).coerceIn(minScore, 1.0)
    }
    
    // === Helpers ===
    
    private fun extractDynamicOffset(match: MatchResult, negative: Boolean): Double {
        val numStr = match.groupValues.getOrNull(1) ?: return 0.0
        val num = numStr.toDoubleOrNull() ?: return 0.0
        
        val multiplier = when {
            match.value.contains("年") -> 365.0
            match.value.contains("月") -> 30.0
            else -> 1.0 // days
        }
        
        return if (negative) -num * multiplier else num * multiplier
    }
}

// === Data Classes ===

enum class AnchorType {
    PAST_VAGUE,      // "以前", "之前"
    PAST_RELATIVE,   // "上周", "3天前"
    PAST_DISTANT,    // "小时候"
    PRESENT,         // "现在", "今天"
    RECENT,          // "最近", "这几天"
    FUTURE_RELATIVE, // "明天", "下周"
    FUTURE_NEAR,     // "即将", "马上"
    FUTURE_VAGUE     // "以后", "将来"
}

enum class TemporalDirection {
    PAST,
    PRESENT,
    FUTURE
}

data class TemporalPattern(
    val regex: Regex,
    val anchorType: AnchorType,
    val offsetDays: Double
)

data class TemporalAnchor(
    val anchorType: AnchorType,
    val anchorTimeMs: Long,
    val confidence: Double,
    val matchedText: String
)

data class TemporalContext(
    val hasTemporalIntent: Boolean,
    val primaryAnchor: TemporalAnchor?,
    val allAnchors: List<TemporalAnchor>,
    val referenceDate: Long,
    val temporalDirection: TemporalDirection
)
