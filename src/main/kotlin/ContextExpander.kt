package com.memory.retrieval

import com.memory.store.MemoryEntry
import com.memory.store.AppendOnlyStore

/**
 * Context Expander (受 MemMachine 启发)
 *
 * 核心洞察：单条记忆片段往往不够——需要"上下文窗口"
 * 
 * 问题场景：
 * - 用户说"我搬到了上海" → 但为什么搬？前几条记忆可能提到"北京房价太高"
 * - 用户说"项目终于完成了" → 什么项目？同 session 的前文有答案
 * - 检索到"他很伤心" → 为什么伤心？时间邻居能解释
 *
 * 策略：
 * 1. Nucleus: 检索命中的核心记忆
 * 2. Session Neighbors: 同一 session 内的前后 N 条
 * 3. Temporal Neighbors: 时间上相近的记忆（跨 session）
 * 4. Entity Chain: 同 entity 的时间链（展示演变过程）
 *
 * 预算控制：
 * - 扩展后的 context 有 token 预算限制
 * - nucleus 优先，neighbors 按距离衰减权重
 * - 超预算时裁剪最远的 neighbors
 *
 * 与 ADD-only 的协同：
 * 因为从不删除记忆，temporal/session neighbors 永远完整可用
 * CRUD 模型中 UPDATE/DELETE 会破坏上下文连续性
 */
class ContextExpander(
    // 每个 nucleus 向前/后扩展的 session neighbor 数量
    private val sessionWindowBefore: Int = 2,
    private val sessionWindowAfter: Int = 1,
    // 时间邻居：前后 N 小时内的记忆
    private val temporalWindowHours: Double = 2.0,
    // Entity chain: 最多展示 N 条同 entity 的历史
    private val entityChainLimit: Int = 3,
    // 最终 context 的最大条目数
    private val maxContextEntries: Int = 12,
    // 去重：同一条记忆不重复出现
    private val deduplicate: Boolean = true
) {
    
    /**
     * 扩展检索结果的上下文
     *
     * @param nuclei 核心检索命中（已排序的 top-K）
     * @param store 完整记忆库（用于查找 neighbors）
     * @param userId 用户 ID
     * @param expansionMode 扩展模式
     * @return 扩展后的 context 列表（有序：nucleus 在前，neighbors 在后）
     */
    fun expand(
        nuclei: List<RankedMemory>,
        store: AppendOnlyStore,
        userId: String,
        expansionMode: ExpansionMode = ExpansionMode.SESSION_AND_TEMPORAL
    ): ExpandedContext {
        if (nuclei.isEmpty()) {
            return ExpandedContext(emptyList(), ExpansionStats())
        }
        
        val allMemories = store.getAll(userId).sortedBy { it.timestamp }
        val memoryById = allMemories.associateBy { it.id }
        val memoryByIndex = allMemories.mapIndexed { i, m -> m.id to i }.toMap()
        
        // Session groups: sessionId → sorted list of entries
        val sessionGroups = allMemories.groupBy { it.sessionId ?: "no_session_${it.id}" }
            .mapValues { (_, v) -> v.sortedBy { it.timestamp } }
        
        val seen = mutableSetOf<String>()
        val contextEntries = mutableListOf<ContextEntry>()
        var stats = ExpansionStats()
        
        // Phase 1: Add nuclei (highest priority)
        for (nucleus in nuclei) {
            if (seen.contains(nucleus.id)) continue
            seen.add(nucleus.id)
            contextEntries.add(ContextEntry(
                memory = nucleus.entry ?: continue,
                role = ContextRole.NUCLEUS,
                score = nucleus.score,
                distanceFromNucleus = 0
            ))
        }
        stats = stats.copy(nucleiCount = contextEntries.size)
        
        // Phase 2: Session neighbors
        if (expansionMode in listOf(ExpansionMode.SESSION_ONLY, ExpansionMode.SESSION_AND_TEMPORAL, ExpansionMode.FULL)) {
            for (nucleus in nuclei) {
                val entry = nucleus.entry ?: continue
                val sessionKey = entry.sessionId ?: "no_session_${entry.id}"
                val sessionList = sessionGroups[sessionKey] ?: continue
                val posInSession = sessionList.indexOfFirst { it.id == entry.id }
                if (posInSession < 0) continue
                
                // Before neighbors
                for (offset in 1..sessionWindowBefore) {
                    val idx = posInSession - offset
                    if (idx < 0) break
                    val neighbor = sessionList[idx]
                    if (deduplicate && neighbor.id in seen) continue
                    seen.add(neighbor.id)
                    contextEntries.add(ContextEntry(
                        memory = neighbor,
                        role = ContextRole.SESSION_NEIGHBOR,
                        score = nucleus.score * (0.8 - offset * 0.15),
                        distanceFromNucleus = -offset
                    ))
                }
                
                // After neighbors
                for (offset in 1..sessionWindowAfter) {
                    val idx = posInSession + offset
                    if (idx >= sessionList.size) break
                    val neighbor = sessionList[idx]
                    if (deduplicate && neighbor.id in seen) continue
                    seen.add(neighbor.id)
                    contextEntries.add(ContextEntry(
                        memory = neighbor,
                        role = ContextRole.SESSION_NEIGHBOR,
                        score = nucleus.score * (0.7 - offset * 0.15),
                        distanceFromNucleus = offset
                    ))
                }
            }
            stats = stats.copy(sessionNeighbors = contextEntries.count { it.role == ContextRole.SESSION_NEIGHBOR })
        }
        
        // Phase 3: Temporal neighbors (cross-session)
        if (expansionMode in listOf(ExpansionMode.TEMPORAL_ONLY, ExpansionMode.SESSION_AND_TEMPORAL, ExpansionMode.FULL)) {
            val windowMs = (temporalWindowHours * 3600 * 1000).toLong()
            
            for (nucleus in nuclei) {
                val entry = nucleus.entry ?: continue
                val startTime = entry.timestamp - windowMs
                val endTime = entry.timestamp + windowMs
                
                for (mem in allMemories) {
                    if (mem.id in seen) continue
                    if (mem.timestamp in startTime..endTime && mem.id != entry.id) {
                        seen.add(mem.id)
                        val timeDist = Math.abs(mem.timestamp - entry.timestamp).toDouble() / windowMs
                        contextEntries.add(ContextEntry(
                            memory = mem,
                            role = ContextRole.TEMPORAL_NEIGHBOR,
                            score = nucleus.score * (0.6 * (1.0 - timeDist)),
                            distanceFromNucleus = ((mem.timestamp - entry.timestamp) / 60000).toInt()
                        ))
                    }
                }
            }
            stats = stats.copy(temporalNeighbors = contextEntries.count { it.role == ContextRole.TEMPORAL_NEIGHBOR })
        }
        
        // Phase 4: Entity chain
        if (expansionMode == ExpansionMode.FULL) {
            val nucleusEntities = nuclei.flatMap { it.entry?.entities ?: emptyList() }.toSet()
            
            for (entity in nucleusEntities) {
                val entityMemories = store.getByEntity(entity)
                    .filter { it.id !in seen }
                    .sortedByDescending { it.timestamp }
                    .take(entityChainLimit)
                
                for (mem in entityMemories) {
                    seen.add(mem.id)
                    contextEntries.add(ContextEntry(
                        memory = mem,
                        role = ContextRole.ENTITY_CHAIN,
                        score = 0.3, // lower priority
                        distanceFromNucleus = -1 // not applicable
                    ))
                }
            }
            stats = stats.copy(entityChainEntries = contextEntries.count { it.role == ContextRole.ENTITY_CHAIN })
        }
        
        // Phase 5: Budget enforcement — sort by score, truncate
        val sorted = contextEntries
            .sortedWith(compareBy<ContextEntry> { 
                when (it.role) {
                    ContextRole.NUCLEUS -> 0
                    ContextRole.SESSION_NEIGHBOR -> 1
                    ContextRole.TEMPORAL_NEIGHBOR -> 2
                    ContextRole.ENTITY_CHAIN -> 3
                }
            }.thenByDescending { it.score })
        
        val final = sorted.take(maxContextEntries)
        stats = stats.copy(
            totalBeforeTruncation = contextEntries.size,
            totalAfterTruncation = final.size,
            truncated = contextEntries.size > maxContextEntries
        )
        
        return ExpandedContext(entries = final, stats = stats)
    }
    
    /**
     * 格式化为 LLM 可读的 context string
     */
    fun formatForLLM(expanded: ExpandedContext): String {
        if (expanded.entries.isEmpty()) return ""
        
        val sb = StringBuilder()
        sb.appendLine("=== 相关记忆 ===")
        
        var currentRole: ContextRole? = null
        for (entry in expanded.entries) {
            if (entry.role != currentRole) {
                currentRole = entry.role
                when (currentRole) {
                    ContextRole.NUCLEUS -> sb.appendLine("\n[核心记忆]")
                    ContextRole.SESSION_NEIGHBOR -> sb.appendLine("\n[对话上下文]")
                    ContextRole.TEMPORAL_NEIGHBOR -> sb.appendLine("\n[时间相近]")
                    ContextRole.ENTITY_CHAIN -> sb.appendLine("\n[相关历史]")
                }
            }
            
            val timeLabel = formatTimestamp(entry.memory.timestamp)
            sb.appendLine("- $timeLabel: ${entry.memory.content}")
        }
        
        return sb.toString()
    }
    
    private fun formatTimestamp(ts: Long): String {
        val now = System.currentTimeMillis()
        val ageMs = now - ts
        val ageHours = ageMs / (1000 * 3600)
        val ageDays = ageMs / (1000 * 3600 * 24)
        
        return when {
            ageHours < 1 -> "刚才"
            ageHours < 24 -> "${ageHours}小时前"
            ageDays < 7 -> "${ageDays}天前"
            ageDays < 30 -> "${ageDays / 7}周前"
            else -> "${ageDays / 30}个月前"
        }
    }
}

// === Data Classes ===

enum class ExpansionMode {
    NUCLEUS_ONLY,        // 不扩展，只返回 nuclei
    SESSION_ONLY,        // 只扩展 session neighbors
    TEMPORAL_ONLY,       // 只扩展 temporal neighbors
    SESSION_AND_TEMPORAL,// session + temporal（默认）
    FULL                 // session + temporal + entity chain
}

enum class ContextRole {
    NUCLEUS,             // 核心检索命中
    SESSION_NEIGHBOR,    // 同 session 的前后文
    TEMPORAL_NEIGHBOR,   // 时间相近的跨 session 记忆
    ENTITY_CHAIN         // 同 entity 的历史演变
}

data class ContextEntry(
    val memory: MemoryEntry,
    val role: ContextRole,
    val score: Double,
    val distanceFromNucleus: Int  // 负=之前, 0=nucleus, 正=之后, -1=不适用
)

data class ExpandedContext(
    val entries: List<ContextEntry>,
    val stats: ExpansionStats
) {
    val nuclei get() = entries.filter { it.role == ContextRole.NUCLEUS }
    val neighbors get() = entries.filter { it.role != ContextRole.NUCLEUS }
    val size get() = entries.size
}

data class ExpansionStats(
    val nucleiCount: Int = 0,
    val sessionNeighbors: Int = 0,
    val temporalNeighbors: Int = 0,
    val entityChainEntries: Int = 0,
    val totalBeforeTruncation: Int = 0,
    val totalAfterTruncation: Int = 0,
    val truncated: Boolean = false
)
