package com.memory.temporal

/**
 * ValidityWindow — 事实有效期推断 (MinnsDB + StructMem 启发)
 *
 * 核心洞察：不同类型的事实有不同的"自然有效期"
 * - "我对花生过敏" → 永久有效
 * - "我在减肥" → 数周到数月
 * - "我明天要面试" → 1-2天
 * - "我现在很累" → 数小时
 *
 * 设计哲学对齐：
 * - 有效期是"信息本体"的一个属性维度，不是外加的标签
 * - ADD-only 不变：过期的 fact 不删除，在 retrieval 时自然降权
 * - 与 TemporalReasoner 配合：validity window 影响 temporal score
 *
 * 使用方式：
 * 1. 存储时：inferValidity(content) → ValidityInfo
 * 2. 检索时：isLikelyValid(memory, queryTime) → boolean/score
 * 3. 不强制过滤，而是作为 signal 之一
 */
class ValidityWindow(
    // Default validity for unclassified facts
    private val defaultValidityDays: Double = 365.0,
    // Grace period: even "expired" facts don't go to 0
    private val graceFactor: Double = 0.3
) {
    
    /**
     * 推断事实的有效期类别
     */
    fun inferValidity(content: String, entities: List<String> = emptyList()): ValidityInfo {
        // Check permanent patterns first
        for ((pattern, category) in PERMANENT_PATTERNS) {
            if (pattern.containsMatchIn(content)) {
                return ValidityInfo(
                    category = category,
                    estimatedDays = -1.0, // -1 = permanent
                    confidence = 0.9,
                    reason = "Matched permanent pattern: ${pattern.pattern}"
                )
            }
        }
        
        // Check ephemeral patterns
        for ((pattern, days, category) in EPHEMERAL_PATTERNS) {
            if (pattern.containsMatchIn(content)) {
                return ValidityInfo(
                    category = category,
                    estimatedDays = days,
                    confidence = 0.7,
                    reason = "Matched ephemeral pattern: ${pattern.pattern}"
                )
            }
        }
        
        // Check medium-term patterns
        for ((pattern, days, category) in MEDIUM_PATTERNS) {
            if (pattern.containsMatchIn(content)) {
                return ValidityInfo(
                    category = category,
                    estimatedDays = days,
                    confidence = 0.6,
                    reason = "Matched medium-term pattern: ${pattern.pattern}"
                )
            }
        }
        
        // Default: long-term but not permanent
        return ValidityInfo(
            category = ValidityCategory.LONG_TERM,
            estimatedDays = defaultValidityDays,
            confidence = 0.3,
            reason = "Default (no pattern matched)"
        )
    }
    
    /**
     * 计算事实在给定查询时间的有效性分数
     *
     * @return [graceFactor, 1.0] — 即使"过期"也不降为0
     */
    fun computeValidityScore(
        factTimestamp: Long,
        queryTime: Long,
        validityInfo: ValidityInfo
    ): Double {
        // Permanent facts always valid
        if (validityInfo.estimatedDays < 0) return 1.0
        
        val ageMs = queryTime - factTimestamp
        val ageDays = ageMs.toDouble() / DAY_MS
        
        // Within validity window → full score
        if (ageDays <= validityInfo.estimatedDays) return 1.0
        
        // Beyond window → decay toward grace factor
        val overageDays = ageDays - validityInfo.estimatedDays
        val decayRate = 0.1 // gentle decay beyond window
        val decay = Math.exp(-decayRate * overageDays)
        
        // Interpolate between grace and 1.0
        return graceFactor + (1.0 - graceFactor) * decay
    }
    
    /**
     * 简单判断：是否可能仍然有效
     */
    fun isLikelyValid(
        factTimestamp: Long,
        queryTime: Long,
        validityInfo: ValidityInfo,
        threshold: Double = 0.5
    ): Boolean {
        return computeValidityScore(factTimestamp, queryTime, validityInfo) >= threshold
    }
    
    companion object {
        private const val DAY_MS = 86400000L
        
        // 永久有效的模式
        val PERMANENT_PATTERNS = listOf(
            Regex("过敏|不耐受|恐惧症|phobia") to ValidityCategory.PERMANENT,
            Regex("血型|生日|出生|姓名|身份证") to ValidityCategory.PERMANENT,
            Regex("母语|国籍|民族|宗教") to ValidityCategory.PERMANENT,
            Regex("一直|永远|从小|天生") to ValidityCategory.PERMANENT,
            Regex("学历|毕业|学位") to ValidityCategory.PERMANENT,
        )
        
        // 短期有效的模式（数小时到数天）
        val EPHEMERAL_PATTERNS = listOf(
            Triple(Regex("现在|此刻|刚才|刚刚"), 0.5, ValidityCategory.EPHEMERAL),    // hours
            Triple(Regex("今天|今晚|今早"), 1.0, ValidityCategory.EPHEMERAL),          // 1 day
            Triple(Regex("明天|后天"), 2.0, ValidityCategory.EPHEMERAL),               // 2 days
            Triple(Regex("这周|本周"), 7.0, ValidityCategory.SHORT_TERM),              // 1 week
            Triple(Regex("心情|感觉|情绪|状态"), 1.0, ValidityCategory.EPHEMERAL),     // emotional states
            Triple(Regex("饿|累|困|渴|疼"), 0.25, ValidityCategory.EPHEMERAL),         // physical states
        )
        
        // 中期有效的模式（数周到数月）
        val MEDIUM_PATTERNS = listOf(
            Triple(Regex("减肥|健身|练习|节食"), 60.0, ValidityCategory.MEDIUM_TERM),
            Triple(Regex("在学|正在看|在读"), 90.0, ValidityCategory.MEDIUM_TERM),
            Triple(Regex("项目|工作任务|deadline"), 30.0, ValidityCategory.MEDIUM_TERM),
            Triple(Regex("住在|租房|合租"), 180.0, ValidityCategory.MEDIUM_TERM),
            Triple(Regex("男朋友|女朋友|对象|约会"), 90.0, ValidityCategory.MEDIUM_TERM),
            Triple(Regex("工作|上班|公司"), 365.0, ValidityCategory.LONG_TERM),
        )
    }
}

// === Data Classes ===

enum class ValidityCategory {
    PERMANENT,      // 永久：过敏、血型、生日
    LONG_TERM,      // 长期：工作、居住地
    MEDIUM_TERM,    // 中期：项目、健身计划
    SHORT_TERM,     // 短期：本周安排
    EPHEMERAL       // 瞬时：当前情绪、身体状态
}

data class ValidityInfo(
    val category: ValidityCategory,
    val estimatedDays: Double,  // -1 = permanent
    val confidence: Double,     // 推断置信度
    val reason: String
)
