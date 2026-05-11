package com.memory.store

import java.util.UUID
import java.util.concurrent.ConcurrentHashMap

/**
 * Append-Only Memory Store (受 Mem0 v3 启发)
 *
 * 核心设计原则：
 * - 存储层只做 ADD（永不覆盖、永不删除）
 * - 矛盾/更新/遗忘 全部在 检索时 通过 ranking 解决
 * - 信息是正交维度（本体 × 时间 × 权重），不是分类
 * - 所有上层组织形式（摘要/分类/重要度）都是查询时的视图函数
 *
 * 为什么 ADD-only：
 * 1. Mem0 v3: LoCoMo 71.4 → 91.6 (+20.2)，仅通过移除 UPDATE/DELETE
 * 2. 信息不丢失 → "住在纽约" 和 "搬到旧金山" 共存，temporal context 区分
 * 3. 审计友好：完整历史可追溯
 * 4. 与 "查询时视图函数" 哲学一致：存储是事实流，视图是检索策略
 *
 * 架构演进：
 * v8/v9: CRUD (Store/Update/Delete/Skip) → 需要决策何时更新/删除
 * v10:    ADD-only + Retrieval-time ranking → 决策完全移到检索阶段
 */
class AppendOnlyStore(
    private val embeddingDim: Int = 1536
) {
    // 核心存储：不可变的 append-only log
    private val memories = ConcurrentHashMap<String, MemoryEntry>()
    
    // Entity index: 轻量级 entity linking (替代完整 graph database)
    private val entityIndex = ConcurrentHashMap<String, MutableSet<String>>() // entity → memoryIds
    
    // Temporal index: 时间戳索引
    private val temporalIndex = sortedMapOf<Long, MutableList<String>>() // timestamp → memoryIds
    
    /**
     * ADD — 唯一的写操作
     *
     * 不做去重、不做覆盖。相同信息多次 ADD 也是合法的
     * （检索时 dedup/ranking 自然处理）
     */
    fun add(
        content: String,
        source: MemorySource,
        entities: List<String> = emptyList(),
        timestamp: Long = System.currentTimeMillis(),
        sessionId: String? = null,
        userId: String? = null,
        metadata: Map<String, Any> = emptyMap()
    ): MemoryEntry {
        val id = UUID.randomUUID().toString()
        val entry = MemoryEntry(
            id = id,
            content = content,
            source = source,
            entities = entities,
            timestamp = timestamp,
            sessionId = sessionId,
            userId = userId,
            metadata = metadata,
            createdAt = System.currentTimeMillis()
        )
        
        // Append to log
        memories[id] = entry
        
        // Update entity index
        for (entity in entities) {
            entityIndex.getOrPut(entity) { mutableSetOf() }.add(id)
        }
        
        // Update temporal index
        synchronized(temporalIndex) {
            temporalIndex.getOrPut(timestamp) { mutableListOf() }.add(id)
        }
        
        return entry
    }
    
    /**
     * 批量 ADD（对话结束后一次性写入）
     */
    fun addBatch(entries: List<AddRequest>): List<MemoryEntry> {
        return entries.map { req ->
            add(
                content = req.content,
                source = req.source,
                entities = req.entities,
                timestamp = req.timestamp,
                sessionId = req.sessionId,
                userId = req.userId,
                metadata = req.metadata
            )
        }
    }
    
    /**
     * 全量读取（供检索层使用）
     */
    fun getAll(userId: String? = null): List<MemoryEntry> {
        return if (userId != null) {
            memories.values.filter { it.userId == userId }
        } else {
            memories.values.toList()
        }
    }
    
    /**
     * 通过 entity 查找相关记忆（entity linking）
     */
    fun getByEntity(entity: String): List<MemoryEntry> {
        val ids = entityIndex[entity] ?: return emptyList()
        return ids.mapNotNull { memories[it] }
    }
    
    /**
     * 时间范围查询
     */
    fun getByTimeRange(start: Long, end: Long): List<MemoryEntry> {
        val ids = mutableListOf<String>()
        synchronized(temporalIndex) {
            temporalIndex.subMap(start, end).values.forEach { ids.addAll(it) }
        }
        return ids.mapNotNull { memories[it] }
    }
    
    /**
     * 统计
     */
    fun count(userId: String? = null): Int {
        return if (userId != null) {
            memories.values.count { it.userId == userId }
        } else {
            memories.size
        }
    }
    
    fun entityCount(): Int = entityIndex.size
}

// === Data Classes ===

/**
 * 记忆条目 — 不可变
 * 一旦写入，永不修改
 */
data class MemoryEntry(
    val id: String,
    val content: String,
    val source: MemorySource,
    val entities: List<String>,
    val timestamp: Long,
    val sessionId: String?,
    val userId: String?,
    val metadata: Map<String, Any>,
    val createdAt: Long
)

/**
 * 记忆来源
 * - USER: 用户说的话中提取的事实
 * - AGENT: AI 说的话中提取的事实（Mem0 v3: first-class）
 * - SYSTEM: 系统事件（登录、操作等）
 */
enum class MemorySource {
    USER,
    AGENT,
    SYSTEM
}

data class AddRequest(
    val content: String,
    val source: MemorySource,
    val entities: List<String> = emptyList(),
    val timestamp: Long = System.currentTimeMillis(),
    val sessionId: String? = null,
    val userId: String? = null,
    val metadata: Map<String, Any> = emptyMap()
)
