package com.memory.retrieval

import com.memory.store.MemoryEntry
import java.util.concurrent.ConcurrentHashMap
import kotlin.math.ln
import kotlin.math.max

/**
 * AccessFrequencyTracker — Mem0 Memory Decay 启发
 * 
 * 跟踪每条记忆被检索命中的次数，用于 search-time ranking bias：
 * - 频繁被检索到的记忆 → 更可能是"核心记忆"，给予 boost
 * - 从未被命中的记忆 → 可能是噪音或过时信息
 * 
 * 设计原则：
 * - 不修改原始记忆（ADD-only 不变）
 * - 访问计数是"索引级元数据"，不是"数据本身"
 * - boost 是对数函数，避免马太效应（高频记忆无限膨胀）
 * - 可以 per-project 配置衰减策略
 * 
 * 公式: frequency_boost = 1.0 + log_factor * ln(1 + access_count)
 * - access_count=0 → boost=1.0 (neutral)
 * - access_count=1 → boost≈1.07
 * - access_count=5 → boost≈1.13
 * - access_count=20 → boost≈1.22
 * - access_count=100 → boost≈1.33
 * 
 * 对数增长确保：
 * 1. 首次命中 → 小 boost（避免 one-hit wonder）
 * 2. 持续命中 → 渐增但有上限
 * 3. 不会出现某条记忆因历史命中过多而永远压制新记忆
 */
class AccessFrequencyTracker(
    // Logarithmic boost factor (higher = more weight to frequency)
    private val logFactor: Double = 0.1,
    // Maximum boost cap (prevents runaway)
    private val maxBoost: Double = 1.5,
    // Minimum hit count to start boosting (noise filter)
    private val minHitsForBoost: Int = 1
) {
    // memory_id → access count
    private val accessCounts: ConcurrentHashMap<String, Int> = ConcurrentHashMap()
    
    /**
     * 记录一次检索命中
     * 在 MultiSignalRetriever.rank() 返回结果后，对 top-K 结果调用此方法
     */
    fun recordAccess(memoryId: String) {
        accessCounts.merge(memoryId, 1) { old, _ -> old + 1 }
    }
    
    /**
     * 批量记录（一次检索的 top-K 结果）
     */
    fun recordAccessBatch(memoryIds: List<String>) {
        memoryIds.forEach { recordAccess(it) }
    }
    
    /**
     * 计算频率 boost
     * @return multiplier ∈ [1.0, maxBoost]
     */
    fun computeBoost(memoryId: String): Double {
        val count = accessCounts.getOrDefault(memoryId, 0)
        if (count < minHitsForBoost) return 1.0
        
        // Logarithmic growth: 1.0 + factor * ln(1 + count)
        val boost = 1.0 + logFactor * ln((1 + count).toDouble())
        return boost.coerceIn(1.0, maxBoost)
    }
    
    /**
     * 获取访问计数
     */
    fun getAccessCount(memoryId: String): Int {
        return accessCounts.getOrDefault(memoryId, 0)
    }
    
    /**
     * 获取所有追踪的记忆的统计
     */
    fun getStats(): FrequencyStats {
        if (accessCounts.isEmpty()) {
            return FrequencyStats(0, 0, 0.0, 0, emptyList())
        }
        
        val counts = accessCounts.values.toList()
        return FrequencyStats(
            trackedMemories = accessCounts.size,
            totalAccesses = counts.sum(),
            averageAccesses = counts.average(),
            maxAccesses = counts.maxOrNull() ?: 0,
            topMemories = accessCounts.entries
                .sortedByDescending { it.value }
                .take(10)
                .map { TopMemory(it.key, it.value, computeBoost(it.key)) }
        )
    }
    
    /**
     * 重置（用于测试或 project 切换）
     */
    fun reset() {
        accessCounts.clear()
    }
    
    /**
     * 导出状态（持久化用）
     */
    fun export(): Map<String, Int> {
        return HashMap(accessCounts)
    }
    
    /**
     * 导入状态（恢复用）
     */
    fun import(data: Map<String, Int>) {
        accessCounts.clear()
        accessCounts.putAll(data)
    }
}

// === Data Classes ===

data class FrequencyStats(
    val trackedMemories: Int,
    val totalAccesses: Int,
    val averageAccesses: Double,
    val maxAccesses: Int,
    val topMemories: List<TopMemory>
)

data class TopMemory(
    val memoryId: String,
    val accessCount: Int,
    val currentBoost: Double
)
