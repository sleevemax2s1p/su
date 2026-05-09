package com.memory.forgetting

import com.memory.governance.GovernanceLayer
import com.memory.provenance.TrustLevel
import kotlinx.coroutines.*
import java.util.concurrent.ConcurrentHashMap
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max

/**
 * 选择性遗忘引擎 (Selective Forgetting Engine)
 * 
 * 基于 MemoryAgentBench (ICLR 2026) 的第四维能力要求：
 * 系统需要能自动遗忘过时、错误、低价值的信息。
 * 
 * 设计哲学（正交维度思维）：
 * 遗忘不是删除，而是 retention_score 降到阈值以下后的自然失活。
 * retention_score = f(recency, access_frequency, trust, governance_layer)
 * 这是查询时的视图函数——记忆仍在，只是不再被检索到。
 * 
 * 三层遗忘策略：
 * 1. Passive Decay (被动衰减) — 随时间自然衰减
 * 2. Active Invalidation (主动失效) — 被新信息 supersede 后标记
 * 3. Contradiction Eviction (矛盾驱逐) — 低信任+有矛盾→加速遗忘
 * 
 * 保护规则（Governance 集成）：
 * - Constitutional 层记忆：永不自动遗忘（retention = 1.0 永久）
 * - Statutory 层记忆：衰减速度 0.5x（用户事实，长期有效）
 * - Operational 层记忆：正常衰减（日常对话）
 */
class SelectiveForgettingEngine(
    private val decayHalfLife: Long = 7 * 24 * 3600 * 1000L,  // 7天半衰期 (operational)
    private val statutoryHalfLife: Long = 90 * 24 * 3600 * 1000L,  // 90天半衰期 (statutory)
    private val retentionThreshold: Double = 0.1,  // 低于此值视为"遗忘"
    private val accessBoostFactor: Double = 1.5,  // 每次访问提升的系数
    private val maxRetentionScore: Double = 1.0,
    private val consolidationIntervalMs: Long = 3600 * 1000L  // 每小时运行一次
) {
    // 每条记忆的遗忘状态
    private val memoryStates = ConcurrentHashMap<String, MemoryRetentionState>()
    
    // 被遗忘的记忆 ID 集合（用于检索时过滤）
    private val forgottenMemories = ConcurrentHashMap.newKeySet<String>()
    
    // 统计
    private var totalForgotten = 0
    private var totalRevived = 0
    private var consolidationRuns = 0
    
    /**
     * 注册新记忆
     * 在存储记忆时调用，初始化 retention 状态
     */
    fun registerMemory(
        memoryId: String,
        layer: GovernanceLayer,
        trustLevel: TrustLevel,
        createdAt: Long = System.currentTimeMillis()
    ) {
        memoryStates[memoryId] = MemoryRetentionState(
            memoryId = memoryId,
            layer = layer,
            trustLevel = trustLevel,
            createdAt = createdAt,
            lastAccessedAt = createdAt,
            accessCount = 0,
            retentionScore = maxRetentionScore,
            supersededBy = null,
            contradictionCount = 0
        )
    }
    
    /**
     * 记录记忆被访问（检索命中时调用）
     * 
     * 每次访问会提升 retention（"越用越记得"）
     * 这是 Ebbinghaus 遗忘曲线的逆过程：间隔重复强化记忆
     */
    fun recordAccess(memoryId: String) {
        memoryStates[memoryId]?.let { state ->
            val now = System.currentTimeMillis()
            memoryStates[memoryId] = state.copy(
                lastAccessedAt = now,
                accessCount = state.accessCount + 1,
                retentionScore = minOf(
                    maxRetentionScore,
                    state.retentionScore * accessBoostFactor
                )
            )
            
            // 如果之前被遗忘了但现在被重新访问（通过全文搜索等），可以复活
            if (memoryId in forgottenMemories) {
                forgottenMemories.remove(memoryId)
                totalRevived++
            }
        }
    }
    
    /**
     * 标记记忆被 supersede（被新信息取代）
     * 
     * 被取代的记忆 retention 立即减半，并加速衰减
     */
    fun markSuperseded(memoryId: String, supersededBy: String) {
        memoryStates[memoryId]?.let { state ->
            memoryStates[memoryId] = state.copy(
                supersededBy = supersededBy,
                retentionScore = state.retentionScore * 0.5  // 立即减半
            )
        }
    }
    
    /**
     * 记录矛盾检测
     * 
     * 每次与其他记忆产生矛盾时计数+1
     * 矛盾多的记忆更容易被遗忘
     */
    fun recordContradiction(memoryId: String) {
        memoryStates[memoryId]?.let { state ->
            memoryStates[memoryId] = state.copy(
                contradictionCount = state.contradictionCount + 1
            )
        }
    }
    
    /**
     * 计算当前 retention score
     * 
     * 公式: retention = base_decay * access_boost * trust_factor * contradiction_penalty
     * 
     * base_decay = exp(-λt), λ = ln(2) / halfLife
     * access_boost = 1 + log(1 + accessCount) * 0.3
     * trust_factor = {HIGH: 1.2, MEDIUM: 1.0, LOW: 0.7}
     * contradiction_penalty = 1.0 / (1 + contradictionCount * 0.3)
     */
    fun computeRetention(memoryId: String, now: Long = System.currentTimeMillis()): Double {
        val state = memoryStates[memoryId] ?: return 0.0
        
        // Constitutional = 永久保留
        if (state.layer == GovernanceLayer.CONSTITUTIONAL) return 1.0
        
        // 选择半衰期
        val halfLife = when (state.layer) {
            GovernanceLayer.STATUTORY -> statutoryHalfLife
            GovernanceLayer.OPERATIONAL -> decayHalfLife
            else -> decayHalfLife
        }
        
        // 时间衰减（指数衰减）
        val timeSinceLastAccess = now - state.lastAccessedAt
        val lambda = ln(2.0) / halfLife
        val baseDecay = exp(-lambda * timeSinceLastAccess)
        
        // 访问频率加成 (对数尺度，避免线性爆炸)
        val accessBoost = 1.0 + ln(1.0 + state.accessCount) * 0.3
        
        // 信任度因子
        val trustFactor = when (state.trustLevel) {
            TrustLevel.HIGH -> 1.2
            TrustLevel.MEDIUM -> 1.0
            TrustLevel.LOW -> 0.7
        }
        
        // 矛盾惩罚
        val contradictionPenalty = 1.0 / (1.0 + state.contradictionCount * 0.3)
        
        // Supersede 惩罚
        val supersedePenalty = if (state.supersededBy != null) 0.3 else 1.0
        
        val retention = baseDecay * accessBoost * trustFactor * contradictionPenalty * supersedePenalty
        return minOf(maxRetentionScore, maxOf(0.0, retention))
    }
    
    /**
     * 执行遗忘整合 (Consolidation)
     * 
     * 定期运行，扫描所有记忆，将 retention 低于阈值的标记为遗忘
     * 类似于大脑睡眠时的记忆整合过程
     */
    fun runConsolidation(now: Long = System.currentTimeMillis()): ConsolidationResult {
        consolidationRuns++
        var newlyForgotten = 0
        var retained = 0
        var protectedByGovernance = 0
        
        for ((memoryId, state) in memoryStates) {
            // Constitutional 永远保护
            if (state.layer == GovernanceLayer.CONSTITUTIONAL) {
                protectedByGovernance++
                continue
            }
            
            val retention = computeRetention(memoryId, now)
            
            // 更新 retention score
            memoryStates[memoryId] = state.copy(retentionScore = retention)
            
            if (retention < retentionThreshold && memoryId !in forgottenMemories) {
                forgottenMemories.add(memoryId)
                newlyForgotten++
                totalForgotten++
            }
            
            if (retention >= retentionThreshold) {
                retained++
            }
        }
        
        return ConsolidationResult(
            totalMemories = memoryStates.size,
            retained = retained,
            newlyForgotten = newlyForgotten,
            totalForgotten = forgottenMemories.size,
            protectedByGovernance = protectedByGovernance,
            runNumber = consolidationRuns
        )
    }
    
    /**
     * 检查记忆是否已被遗忘
     * 在检索时调用，过滤掉已遗忘的记忆
     */
    fun isForgotten(memoryId: String): Boolean {
        return memoryId in forgottenMemories
    }
    
    /**
     * 获取需要遗忘的记忆列表
     * 提供给检索层用于过滤
     */
    fun getForgottenMemoryIds(): Set<String> {
        return forgottenMemories.toSet()
    }
    
    /**
     * 获取统计信息
     */
    fun getStats(): ForgettingStats {
        return ForgettingStats(
            totalTracked = memoryStates.size,
            currentlyForgotten = forgottenMemories.size,
            totalEverForgotten = totalForgotten,
            totalRevived = totalRevived,
            consolidationRuns = consolidationRuns,
            avgRetention = memoryStates.values
                .filter { it.layer != GovernanceLayer.CONSTITUTIONAL }
                .map { it.retentionScore }
                .average()
                .takeIf { !it.isNaN() } ?: 0.0
        )
    }
    
    /**
     * 用户主动要求遗忘（Right-to-Forget 的底层实现）
     */
    fun forceForget(memoryId: String): Boolean {
        if (memoryId in memoryStates) {
            forgottenMemories.add(memoryId)
            memoryStates[memoryId]?.let {
                memoryStates[memoryId] = it.copy(retentionScore = 0.0)
            }
            totalForgotten++
            return true
        }
        return false
    }
}

// === Data Classes ===

data class MemoryRetentionState(
    val memoryId: String,
    val layer: GovernanceLayer,
    val trustLevel: TrustLevel,
    val createdAt: Long,
    val lastAccessedAt: Long,
    val accessCount: Int,
    val retentionScore: Double,
    val supersededBy: String?,
    val contradictionCount: Int
)

data class ConsolidationResult(
    val totalMemories: Int,
    val retained: Int,
    val newlyForgotten: Int,
    val totalForgotten: Int,
    val protectedByGovernance: Int,
    val runNumber: Int
) {
    override fun toString(): String = """
        |=== Consolidation Run #$runNumber ===
        |Total memories: $totalMemories
        |Retained: $retained
        |Newly forgotten: $newlyForgotten
        |Total forgotten: $totalForgotten
        |Protected (Constitutional): $protectedByGovernance
    """.trimMargin()
}

data class ForgettingStats(
    val totalTracked: Int,
    val currentlyForgotten: Int,
    val totalEverForgotten: Int,
    val totalRevived: Int,
    val consolidationRuns: Int,
    val avgRetention: Double
)
