package com.memory.provenance

import java.text.SimpleDateFormat
import java.util.Date
import java.util.UUID

/**
 * 记忆溯源系统 (Memory Provenance)
 * 
 * 灵感来源：
 * - Context Engineering 五大标准之 Provenance (§14.22.3)
 * - Animesis CMA 的 Governance Primitives: write ownership (§14.22.1)
 * - Letta Git-backed Memory: 变更历史 (§14.22.2)
 * 
 * 核心理念：
 * 每条记忆不仅有内容，还有「它从哪来、什么时候来的、谁说的」。
 * 
 * 解决的问题：
 * 1. 矛盾解决时，知道两条冲突记忆各自的来源和时间
 * 2. 记忆审计：追踪某条记忆是何时被创建/修改/删除的
 * 3. 信任校准：第一手信息（用户说的）vs 推断信息（Agent 总结的）可信度不同
 * 4. 遗忘权执行：精确定位并删除特定来源的记忆
 */
class MemoryProvenance {
    
    // 变更日志（简化版 git log）
    private val changelog = mutableListOf<ChangeRecord>()
    
    // 溯源索引：memoryId -> ProvenanceInfo
    private val provenanceIndex = mutableMapOf<String, ProvenanceInfo>()
    
    private fun log(msg: String) {
        val ts = SimpleDateFormat("HH:mm:ss.SSS").format(Date())
        println("[$ts][Provenance] $msg")
    }

    /**
     * 为新记忆创建溯源信息
     * 
     * 在记忆写入时调用，附加元数据
     */
    fun createProvenance(
        memoryId: String,
        content: String,
        source: ProvenanceSource
    ): ProvenanceInfo {
        val provenance = ProvenanceInfo(
            memoryId = memoryId,
            createdAt = System.currentTimeMillis(),
            source = source,
            trustLevel = calculateTrustLevel(source),
            changeHistory = mutableListOf(
                ChangeRecord(
                    id = UUID.randomUUID().toString().take(8),
                    memoryId = memoryId,
                    operation = ChangeOperation.CREATE,
                    timestamp = System.currentTimeMillis(),
                    initiator = source.initiator,
                    sessionId = source.sessionId,
                    contentSnapshot = content.take(100)
                )
            )
        )
        
        provenanceIndex[memoryId] = provenance
        log("创建溯源: $memoryId | source=${source.type} | trust=${provenance.trustLevel}")
        return provenance
    }

    /**
     * 记录记忆修改事件
     */
    fun recordModification(
        memoryId: String,
        operation: ChangeOperation,
        initiator: String,
        sessionId: String,
        reason: String = "",
        newContent: String? = null
    ) {
        val record = ChangeRecord(
            id = UUID.randomUUID().toString().take(8),
            memoryId = memoryId,
            operation = operation,
            timestamp = System.currentTimeMillis(),
            initiator = initiator,
            sessionId = sessionId,
            reason = reason,
            contentSnapshot = newContent?.take(100)
        )
        
        changelog.add(record)
        provenanceIndex[memoryId]?.changeHistory?.add(record)
        log("记录变更: $memoryId | op=$operation | by=$initiator | reason=${reason.take(50)}")
    }

    /**
     * 查询某条记忆的完整溯源信息
     */
    fun getProvenance(memoryId: String): ProvenanceInfo? {
        return provenanceIndex[memoryId]
    }

    /**
     * 获取两条冲突记忆的溯源对比
     * 
     * 用于矛盾解决：谁更可信？谁更新？
     */
    fun compareProvenance(memoryIdA: String, memoryIdB: String): ProvenanceComparison {
        val a = provenanceIndex[memoryIdA]
        val b = provenanceIndex[memoryIdB]
        
        if (a == null || b == null) {
            return ProvenanceComparison(
                winner = null,
                reason = "One or both memories have no provenance info"
            )
        }
        
        // 决策逻辑：
        // 1. 用户直接说的 > Agent 推断的
        // 2. 同类型来源时，更新的 > 更旧的
        // 3. 被多次强化的 > 只出现一次的
        
        val decision = when {
            a.trustLevel > b.trustLevel -> "A" 
            b.trustLevel > a.trustLevel -> "B"
            a.createdAt > b.createdAt -> "A"  // 同信任度，更新的优先
            b.createdAt > a.createdAt -> "B"
            else -> null  // 无法自动决定
        }
        
        val reason = when (decision) {
            "A" -> "Memory A has higher trust (${a.source.type}) or is more recent"
            "B" -> "Memory B has higher trust (${b.source.type}) or is more recent"
            else -> "Cannot determine automatically, requires user input"
        }
        
        return ProvenanceComparison(
            winner = decision,
            provenanceA = a,
            provenanceB = b,
            reason = reason
        )
    }

    /**
     * 按来源筛选记忆
     * 
     * 用于遗忘权执行：找到某个 session/用户产生的所有记忆
     */
    fun findBySource(sessionId: String? = null, senderName: String? = null): List<String> {
        return provenanceIndex.entries
            .filter { (_, info) ->
                (sessionId == null || info.source.sessionId == sessionId) &&
                (senderName == null || info.source.senderName == senderName)
            }
            .map { it.key }
    }

    /**
     * 获取最近 N 条变更记录（审计用）
     */
    fun getRecentChanges(limit: Int = 20): List<ChangeRecord> {
        return changelog.takeLast(limit)
    }

    /**
     * 计算记忆的信任度
     * 
     * 基于来源类型：
     * - 用户直接陈述 > 用户对话推断 > Agent 自动总结 > 系统生成
     */
    private fun calculateTrustLevel(source: ProvenanceSource): TrustLevel {
        return when (source.type) {
            SourceType.USER_DIRECT_STATEMENT -> TrustLevel.HIGH
            SourceType.USER_CONVERSATION_INFERRED -> TrustLevel.MEDIUM_HIGH
            SourceType.AGENT_EXTRACTION -> TrustLevel.MEDIUM
            SourceType.AGENT_CONSOLIDATION -> TrustLevel.MEDIUM_LOW
            SourceType.SYSTEM_GENERATED -> TrustLevel.LOW
        }
    }

    fun getStats(): String {
        return "溯源 | 已索引: ${provenanceIndex.size} 条 | 变更记录: ${changelog.size} 条"
    }
}

// === Data Models ===

data class ProvenanceInfo(
    val memoryId: String,
    val createdAt: Long,
    val source: ProvenanceSource,
    val trustLevel: TrustLevel,
    val changeHistory: MutableList<ChangeRecord> = mutableListOf()
) {
    val modificationCount: Int get() = changeHistory.size - 1  // 减去初始 CREATE
    val lastModifiedAt: Long get() = changeHistory.lastOrNull()?.timestamp ?: createdAt
}

data class ProvenanceSource(
    val type: SourceType,
    val sessionId: String,
    val senderName: String? = null,
    val initiator: String,  // "user" | "agent" | "system"
    val extractionMethod: String = "llm_extraction"  // 提取方法
)

data class ChangeRecord(
    val id: String,
    val memoryId: String,
    val operation: ChangeOperation,
    val timestamp: Long,
    val initiator: String,
    val sessionId: String,
    val reason: String = "",
    val contentSnapshot: String? = null
)

data class ProvenanceComparison(
    val winner: String?,  // "A", "B", or null
    val provenanceA: ProvenanceInfo? = null,
    val provenanceB: ProvenanceInfo? = null,
    val reason: String
)

enum class SourceType {
    USER_DIRECT_STATEMENT,      // 用户明确说的："我叫小明"
    USER_CONVERSATION_INFERRED, // 从对话推断的：用户提到"我们公司"→ 推断他在某公司工作
    AGENT_EXTRACTION,           // Agent LLM 提取的
    AGENT_CONSOLIDATION,        // Sleep-Time 巩固产生的
    SYSTEM_GENERATED            // 系统自动生成（如时间衰减标记）
}

enum class TrustLevel(val score: Double) {
    HIGH(0.95),
    MEDIUM_HIGH(0.8),
    MEDIUM(0.65),
    MEDIUM_LOW(0.5),
    LOW(0.3)
}

enum class ChangeOperation {
    CREATE,      // 创建
    UPDATE,      // 更新/强化
    MERGE,       // 合并（巩固）
    INVALIDATE,  // 标记无效（矛盾解决）
    DELETE,       // 删除
    DECAY        // 衰减
}
