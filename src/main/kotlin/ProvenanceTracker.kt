package com.memory.store

/**
 * ProvenanceTracker — 记忆溯源系统 (MemORAI 启发)
 *
 * 每条记忆不仅有内容，还有完整的"出生证明"：
 * - 来自哪个 turn（turn_id）
 * - 来自哪个 session
 * - 原始消息是什么（context）
 * - 提取时的置信度
 * - 是否经过 admission（通过/拒绝/fast-path）
 *
 * 用途：
 * 1. "你怎么知道我喜欢猫的？" → 溯源到具体对话
 * 2. 置信度低的记忆在 retrieval 时可以降权
 * 3. 同源记忆可以聚合（同一条消息提取的多个 fact）
 * 4. Debug：追踪记忆从 raw message → stored fact 的完整链路
 *
 * 设计哲学对齐：
 * - 溯源信息是"元数据维度"，不是记忆本身
 * - 存储时附加，查询时可选使用（视图函数可以忽略它）
 * - ADD-only：溯源记录也不可变
 */
class ProvenanceTracker {
    
    // memory_id → provenance record
    private val records: MutableMap<String, ProvenanceRecord> = mutableMapOf()
    
    /**
     * 记录一条记忆的溯源信息
     */
    fun record(
        memoryId: String,
        turnId: String,
        sessionId: String,
        userId: String,
        originalMessage: String,
        extractionConfidence: Double = 1.0,
        admissionDecision: AdmissionOutcome = AdmissionOutcome.ADMITTED,
        extractionMethod: String = "fact_extractor_v1",
        timestamp: Long = System.currentTimeMillis()
    ): ProvenanceRecord {
        val record = ProvenanceRecord(
            memoryId = memoryId,
            turnId = turnId,
            sessionId = sessionId,
            userId = userId,
            originalMessage = originalMessage,
            extractionConfidence = extractionConfidence,
            admissionDecision = admissionDecision,
            extractionMethod = extractionMethod,
            createdAt = timestamp
        )
        records[memoryId] = record
        return record
    }
    
    /**
     * 查询记忆来源
     */
    fun getProvenance(memoryId: String): ProvenanceRecord? {
        return records[memoryId]
    }
    
    /**
     * 查询某个 turn 产生的所有记忆
     */
    fun getByTurn(turnId: String): List<ProvenanceRecord> {
        return records.values.filter { it.turnId == turnId }
    }
    
    /**
     * 查询某个 session 产生的所有记忆
     */
    fun getBySession(sessionId: String): List<ProvenanceRecord> {
        return records.values.filter { it.sessionId == sessionId }
    }
    
    /**
     * 获取同源记忆（来自同一条原始消息的多个 facts）
     */
    fun getSiblings(memoryId: String): List<ProvenanceRecord> {
        val record = records[memoryId] ?: return emptyList()
        return records.values.filter { 
            it.turnId == record.turnId && it.memoryId != memoryId 
        }
    }
    
    /**
     * 置信度过滤：返回低于阈值的记忆（可能需要验证或降权）
     */
    fun getLowConfidence(threshold: Double = 0.5): List<ProvenanceRecord> {
        return records.values.filter { it.extractionConfidence < threshold }
    }
    
    /**
     * 统计
     */
    fun getStats(): ProvenanceStats {
        val allRecords = records.values.toList()
        if (allRecords.isEmpty()) {
            return ProvenanceStats(0, 0, 0, 0.0, emptyMap())
        }
        
        val sessions = allRecords.map { it.sessionId }.distinct().size
        val turns = allRecords.map { it.turnId }.distinct().size
        val avgConfidence = allRecords.map { it.extractionConfidence }.average()
        val byOutcome = allRecords.groupBy { it.admissionDecision }
            .mapValues { it.value.size }
        
        return ProvenanceStats(
            totalMemories = allRecords.size,
            totalSessions = sessions,
            totalTurns = turns,
            avgConfidence = avgConfidence,
            admissionBreakdown = byOutcome
        )
    }
    
    /**
     * 计算溯源相关的检索信号
     * 
     * 高置信度 + 被验证过 → boost
     * 低置信度 + 未验证 → slight penalty
     */
    fun computeProvenanceSignal(memoryId: String): Double {
        val record = records[memoryId] ?: return 1.0 // no provenance = neutral
        
        var signal = record.extractionConfidence
        
        // Fast-path admission (high-value keywords) gets slight boost
        if (record.admissionDecision == AdmissionOutcome.FAST_PATH) {
            signal = (signal * 1.1).coerceAtMost(1.0)
        }
        
        return signal.coerceIn(0.3, 1.0) // never fully suppress
    }
}

// === Data Classes ===

data class ProvenanceRecord(
    val memoryId: String,
    val turnId: String,
    val sessionId: String,
    val userId: String,
    val originalMessage: String,
    val extractionConfidence: Double,
    val admissionDecision: AdmissionOutcome,
    val extractionMethod: String,
    val createdAt: Long
)

enum class AdmissionOutcome {
    ADMITTED,       // Normal admission through scoring
    FAST_PATH,      // High-value fast path (allergy, health, etc.)
    REJECTED,       // Below threshold
    FILTERED        // Caught by pre-filter (filler, too short)
}

data class ProvenanceStats(
    val totalMemories: Int,
    val totalSessions: Int,
    val totalTurns: Int,
    val avgConfidence: Double,
    val admissionBreakdown: Map<AdmissionOutcome, Int>
)
