package com.memory.safety

/**
 * PinnedMemoryGuard — 安全关键记忆的豁免机制
 *
 * 解决的问题：
 * 某些记忆对用户安全至关重要（过敏、用药禁忌、紧急联系人），
 * 不应该被任何降权信号（时间衰减、低频访问、低置信度）影响。
 *
 * 设计原则：
 * - "Pin" 是一种元数据标记，不改变底层数据（ADD-only 不变）
 * - Pin 在 ingestion 阶段自动检测，也可手动标记
 * - Retrieval 时 pinned memory 有 minimum floor score（保底分）
 * - Pin 不等于"永远排第一"——只保证不被降到 noise 水平
 *
 * 与 ValidityWindow 的关系：
 * - ValidityWindow PERMANENT → 自动成为 pin 候选
 * - 但 pin 还包括非永久但安全关键的信息（如"明天手术不能吃东西"）
 *
 * 与 AdmissionController 的关系：
 * - AdmissionController fast-path（过敏/健康）→ 自动 pin
 * - Pin 是 fast-path 的"检索侧对偶"：fast-path 保证入库，pin 保证检出
 *
 * 灵感来源：
 * - Mem0 Memory Decay edge case testing（安全关键记忆的衰减问题）
 * - ICLR 2026 MemAgents Workshop 共识：safety-critical info 需要特殊处理
 *
 * 使用方式：
 * 在 ChatEngine v11 的 applyV11Signals() 后增加一步：
 * if (pinnedGuard.isPinned(mem.id)) { score = max(score, floorScore) }
 */
class PinnedMemoryGuard(
    // Minimum score for pinned memories (prevents decay to noise level)
    private val floorScore: Double = 0.5,
    // Auto-pin patterns (safety-critical)
    private val autoPinPatterns: List<PinPattern> = DEFAULT_PIN_PATTERNS
) {
    // Manually pinned memory IDs
    private val pinnedIds: MutableSet<String> = mutableSetOf()
    
    // Auto-pin cache (computed at ingestion time)
    private val autoPinCache: MutableMap<String, PinReason> = mutableMapOf()
    
    /**
     * 在 ingestion 时自动检测是否应该 pin
     * 
     * @return PinDecision with reason (or NOT_PINNED)
     */
    fun evaluateForPin(memoryId: String, content: String, entities: List<String> = emptyList()): PinDecision {
        for (pattern in autoPinPatterns) {
            if (pattern.regex.containsMatchIn(content)) {
                autoPinCache[memoryId] = pattern.reason
                return PinDecision(
                    pinned = true,
                    reason = pattern.reason,
                    category = pattern.category,
                    explanation = "Auto-pinned: matched '${pattern.regex.pattern}'"
                )
            }
        }
        
        return PinDecision(pinned = false, reason = PinReason.NOT_PINNED, 
                           category = PinCategory.NONE, explanation = "No pin pattern matched")
    }
    
    /**
     * 手动 pin（用户明确说"记住这个很重要"）
     */
    fun manualPin(memoryId: String, reason: PinReason = PinReason.USER_EXPLICIT) {
        pinnedIds.add(memoryId)
        autoPinCache[memoryId] = reason
    }
    
    /**
     * 手动 unpin
     */
    fun unpin(memoryId: String) {
        pinnedIds.remove(memoryId)
        autoPinCache.remove(memoryId)
    }
    
    /**
     * 检查是否 pinned
     */
    fun isPinned(memoryId: String): Boolean {
        return memoryId in pinnedIds || memoryId in autoPinCache
    }
    
    /**
     * 获取 pin 原因
     */
    fun getPinReason(memoryId: String): PinReason? {
        return autoPinCache[memoryId]
    }
    
    /**
     * 在 retrieval 阶段应用 pin 保护
     * 
     * 核心逻辑：如果记忆被 pin，其 final score 不低于 floorScore
     * 这不改变排序中的其他记忆，只保证 pinned 不沉底
     */
    fun applyPinProtection(memoryId: String, currentScore: Double): Double {
        if (!isPinned(memoryId)) return currentScore
        return maxOf(currentScore, floorScore)
    }
    
    /**
     * 批量应用 pin 保护
     */
    fun applyPinProtectionBatch(scoredMemories: List<Pair<String, Double>>): List<Pair<String, Double>> {
        return scoredMemories.map { (id, score) -> id to applyPinProtection(id, score) }
    }
    
    /**
     * 获取所有 pinned memories
     */
    fun getAllPinned(): Map<String, PinReason> {
        return HashMap(autoPinCache)
    }
    
    /**
     * 统计
     */
    fun getStats(): PinStats {
        val byReason = autoPinCache.values.groupBy { it }.mapValues { it.value.size }
        return PinStats(
            totalPinned = autoPinCache.size,
            manualPins = pinnedIds.size,
            autoPins = autoPinCache.size - pinnedIds.size,
            byReason = byReason
        )
    }
    
    companion object {
        val DEFAULT_PIN_PATTERNS = listOf(
            // Health & Safety
            PinPattern(Regex("过敏|不耐受|禁忌"), PinReason.HEALTH_ALLERGY, PinCategory.HEALTH),
            PinPattern(Regex("药物|用药|服药|处方"), PinReason.HEALTH_MEDICATION, PinCategory.HEALTH),
            PinPattern(Regex("手术|住院|急诊"), PinReason.HEALTH_PROCEDURE, PinCategory.HEALTH),
            PinPattern(Regex("糖尿病|高血压|心脏病|哮喘"), PinReason.HEALTH_CONDITION, PinCategory.HEALTH),
            
            // Emergency contacts & safety
            PinPattern(Regex("紧急联系|急救|SOS"), PinReason.EMERGENCY_CONTACT, PinCategory.SAFETY),
            PinPattern(Regex("不能吃|不能喝|不能碰"), PinReason.HEALTH_RESTRICTION, PinCategory.HEALTH),
            
            // Critical personal info
            PinPattern(Regex("密码|账号|银行"), PinReason.SENSITIVE_CREDENTIAL, PinCategory.SECURITY),
            
            // User-expressed importance
            PinPattern(Regex("很重要|千万别忘|一定要记住|务必"), PinReason.USER_EMPHASIS, PinCategory.USER_MARKED),
        )
    }
}

// === Data Classes ===

data class PinPattern(
    val regex: Regex,
    val reason: PinReason,
    val category: PinCategory
)

enum class PinReason {
    NOT_PINNED,
    HEALTH_ALLERGY,
    HEALTH_MEDICATION,
    HEALTH_PROCEDURE,
    HEALTH_CONDITION,
    HEALTH_RESTRICTION,
    EMERGENCY_CONTACT,
    SENSITIVE_CREDENTIAL,
    USER_EMPHASIS,
    USER_EXPLICIT       // Manual pin by user
}

enum class PinCategory {
    NONE,
    HEALTH,
    SAFETY,
    SECURITY,
    USER_MARKED
}

data class PinDecision(
    val pinned: Boolean,
    val reason: PinReason,
    val category: PinCategory,
    val explanation: String
)

data class PinStats(
    val totalPinned: Int,
    val manualPins: Int,
    val autoPins: Int,
    val byReason: Map<PinReason, Int>
)
