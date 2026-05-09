package com.memory.graph

import kotlinx.serialization.Serializable

/**
 * 实体关系图 (Entity-Relation Graph Memory)
 * 
 * 基于 Mem0g (ECAI 2025) 验证的 Graph Memory 模式：
 * - Vector memory: "用户提到了 Python" (单跳)
 * - Graph memory: "用户 → 使用 → Python → 用于 → 数据管道 → 使用 → pandas" (多跳)
 * 
 * 设计哲学（正交维度）：
 * - Graph 是信息的一种视图/组织形式
 * - 不改变底层存储（信息仍是 flat 的）
 * - 通过关系链接在查询时提供多跳推理能力
 * 
 * 与现有系统集成：
 * - 实体提取在 Context-Aware Extraction (v8) 中进行
 * - 关系存储在本模块
 * - 查询时通过 BFS/DFS 遍历提供额外上下文
 * 
 * Benchmark 参考:
 * - Mem0g: 68.4% accuracy vs Mem0 66.9% (+1.5%) on multi-hop
 * - Latency: 2.59s p95 vs 1.44s (可接受的 tradeoff)
 */
class EntityRelationGraph {
    
    // 实体节点: id → EntityNode
    private val entities = mutableMapOf<String, EntityNode>()
    
    // 关系边: sourceId → [(relation, targetId)]
    private val outEdges = mutableMapOf<String, MutableList<RelationEdge>>()
    private val inEdges = mutableMapOf<String, MutableList<RelationEdge>>()
    
    // 实体名 → 实体ID 的索引 (支持模糊查找)
    private val nameIndex = mutableMapOf<String, MutableSet<String>>()
    
    // 统计
    private var totalEntities = 0
    private var totalRelations = 0
    private var totalConflictsDetected = 0
    
    /**
     * 添加或更新实体
     */
    fun upsertEntity(
        name: String,
        type: EntityType,
        attributes: Map<String, String> = emptyMap(),
        sourceMemoryId: String? = null
    ): String {
        val normalized = name.trim().lowercase()
        
        // 查找已有实体
        val existingIds = nameIndex[normalized]
        if (existingIds != null && existingIds.isNotEmpty()) {
            val existingId = existingIds.first()
            val existing = entities[existingId]!!
            // Merge attributes
            val merged = existing.attributes + attributes
            entities[existingId] = existing.copy(
                attributes = merged,
                lastUpdated = System.currentTimeMillis(),
                sourceMemoryIds = existing.sourceMemoryIds + listOfNotNull(sourceMemoryId)
            )
            return existingId
        }
        
        // 创建新实体
        val id = "ent_${++totalEntities}"
        entities[id] = EntityNode(
            id = id,
            name = name,
            normalizedName = normalized,
            type = type,
            attributes = attributes,
            createdAt = System.currentTimeMillis(),
            lastUpdated = System.currentTimeMillis(),
            sourceMemoryIds = listOfNotNull(sourceMemoryId)
        )
        nameIndex.getOrPut(normalized) { mutableSetOf() }.add(id)
        
        // 也索引别名/部分名
        if (normalized.length > 2) {
            // 为中文名添加全名索引
            nameIndex.getOrPut(normalized) { mutableSetOf() }.add(id)
        }
        
        return id
    }
    
    /**
     * 添加关系
     * 
     * 冲突检测：如果已存在同主体同关系但不同目标，标记为冲突
     * 例如：用户→住在→北京 vs 用户→住在→上海 → conflict!
     */
    fun addRelation(
        sourceId: String,
        relation: String,
        targetId: String,
        confidence: Double = 1.0,
        sourceMemoryId: String? = null
    ): RelationResult {
        if (sourceId !in entities || targetId !in entities) {
            return RelationResult(success = false, reason = "Entity not found")
        }
        
        // 冲突检测: 同源+同关系+不同目标 = 潜在冲突
        val existingEdges = outEdges[sourceId] ?: emptyList()
        val conflicting = existingEdges.filter { 
            it.relation == relation && it.targetId != targetId 
        }
        
        if (conflicting.isNotEmpty()) {
            totalConflictsDetected++
            // 标记旧关系为 superseded
            for (edge in conflicting) {
                edge.superseded = true
                edge.supersededBy = "rel_${totalRelations + 1}"
            }
        }
        
        val edge = RelationEdge(
            id = "rel_${++totalRelations}",
            sourceId = sourceId,
            relation = relation,
            targetId = targetId,
            confidence = confidence,
            createdAt = System.currentTimeMillis(),
            sourceMemoryId = sourceMemoryId,
            superseded = false,
            supersededBy = null
        )
        
        outEdges.getOrPut(sourceId) { mutableListOf() }.add(edge)
        inEdges.getOrPut(targetId) { mutableListOf() }.add(edge)
        
        return RelationResult(
            success = true,
            edgeId = edge.id,
            conflictsDetected = conflicting.size,
            conflictingEdgeIds = conflicting.map { it.id }
        )
    }
    
    /**
     * 多跳图查询 (BFS)
     * 
     * 从起始实体出发，沿关系链最多走 maxHops 步
     * 返回路径上所有相关实体和关系
     * 
     * 用于回答多跳问题：
     * "用户的女朋友在哪里工作？" → 用户→女朋友→小红→工作→XX公司
     */
    fun multiHopQuery(
        startEntityId: String,
        maxHops: Int = 3,
        relationFilter: Set<String>? = null,
        includeSuperseded: Boolean = false
    ): GraphQueryResult {
        val visited = mutableSetOf<String>()
        val paths = mutableListOf<GraphPath>()
        
        data class QueueItem(
            val entityId: String,
            val path: List<PathStep>,
            val depth: Int
        )
        
        val queue = ArrayDeque<QueueItem>()
        queue.add(QueueItem(startEntityId, emptyList(), 0))
        visited.add(startEntityId)
        
        while (queue.isNotEmpty()) {
            val (currentId, currentPath, depth) = queue.removeFirst()
            
            if (depth >= maxHops) continue
            
            val edges = outEdges[currentId] ?: continue
            for (edge in edges) {
                if (!includeSuperseded && edge.superseded) continue
                if (relationFilter != null && edge.relation !in relationFilter) continue
                
                val nextPath = currentPath + PathStep(
                    fromEntity = entities[currentId]!!.name,
                    relation = edge.relation,
                    toEntity = entities[edge.targetId]!!.name,
                    confidence = edge.confidence
                )
                
                paths.add(GraphPath(steps = nextPath, totalConfidence = nextPath.map { it.confidence }.reduce { a, b -> a * b }))
                
                if (edge.targetId !in visited) {
                    visited.add(edge.targetId)
                    queue.add(QueueItem(edge.targetId, nextPath, depth + 1))
                }
            }
        }
        
        return GraphQueryResult(
            startEntity = entities[startEntityId]?.name ?: "unknown",
            paths = paths.sortedByDescending { it.totalConfidence },
            entitiesVisited = visited.size,
            maxDepthReached = paths.maxOfOrNull { it.steps.size } ?: 0
        )
    }
    
    /**
     * 按名称查找实体
     */
    fun findEntityByName(name: String): EntityNode? {
        val normalized = name.trim().lowercase()
        val ids = nameIndex[normalized] ?: return null
        return ids.firstOrNull()?.let { entities[it] }
    }
    
    /**
     * 获取实体的所有关系（出+入）
     */
    fun getEntityRelations(entityId: String): EntityRelations {
        val outgoing = (outEdges[entityId] ?: emptyList()).filter { !it.superseded }
        val incoming = (inEdges[entityId] ?: emptyList()).filter { !it.superseded }
        
        return EntityRelations(
            entity = entities[entityId],
            outgoing = outgoing.map { edge ->
                RelationSummary(edge.relation, entities[edge.targetId]!!.name, edge.confidence)
            },
            incoming = incoming.map { edge ->
                RelationSummary(edge.relation, entities[edge.sourceId]!!.name, edge.confidence, isIncoming = true)
            }
        )
    }
    
    /**
     * 生成实体摘要（用于注入到 prompt 中）
     * 
     * 类似 Hindsight 的 Entity Summaries:
     * "小红: 用户的女朋友, 住在北京, 在腾讯工作"
     */
    fun generateEntitySummary(entityId: String): String {
        val entity = entities[entityId] ?: return ""
        val relations = getEntityRelations(entityId)
        
        val parts = mutableListOf<String>()
        parts.add("${entity.name} (${entity.type.displayName})")
        
        for (rel in relations.outgoing) {
            parts.add("${rel.relation}${rel.target}")
        }
        for (rel in relations.incoming) {
            parts.add("被${rel.target}${rel.relation}")
        }
        
        return parts.joinToString(", ")
    }
    
    /**
     * 从对话中批量提取实体和关系
     * 
     * 输入: LLM 提取的三元组列表 [(subject, relation, object)]
     * 输出: 构建的图结构
     */
    fun batchIngest(
        triples: List<Triple<String, String, String>>,
        entityTypes: Map<String, EntityType> = emptyMap(),
        sourceMemoryId: String? = null
    ): IngestResult {
        var entitiesCreated = 0
        var relationsCreated = 0
        var conflicts = 0
        
        for ((subject, relation, obj) in triples) {
            val subjectType = entityTypes[subject] ?: EntityType.UNKNOWN
            val objectType = entityTypes[obj] ?: EntityType.UNKNOWN
            
            val subjectId = upsertEntity(subject, subjectType, sourceMemoryId = sourceMemoryId)
            val objectId = upsertEntity(obj, objectType, sourceMemoryId = sourceMemoryId)
            
            if (entities[subjectId]!!.sourceMemoryIds.size == 1) entitiesCreated++
            if (entities[objectId]!!.sourceMemoryIds.size == 1) entitiesCreated++
            
            val result = addRelation(subjectId, relation, objectId, sourceMemoryId = sourceMemoryId)
            if (result.success) relationsCreated++
            conflicts += result.conflictsDetected
        }
        
        return IngestResult(
            entitiesCreated = entitiesCreated,
            relationsCreated = relationsCreated,
            conflictsDetected = conflicts,
            totalEntities = entities.size,
            totalRelations = outEdges.values.sumOf { it.size }
        )
    }
    
    fun getStats(): GraphStats {
        return GraphStats(
            totalEntities = entities.size,
            totalRelations = outEdges.values.sumOf { it.size },
            totalConflicts = totalConflictsDetected,
            activeRelations = outEdges.values.sumOf { edges -> edges.count { !it.superseded } },
            supersededRelations = outEdges.values.sumOf { edges -> edges.count { it.superseded } }
        )
    }
}

// === Enums & Data Classes ===

enum class EntityType(val displayName: String) {
    PERSON("人物"),
    LOCATION("地点"),
    ORGANIZATION("组织"),
    EVENT("事件"),
    CONCEPT("概念"),
    OBJECT("物品"),
    TIME("时间"),
    UNKNOWN("未知")
}

data class EntityNode(
    val id: String,
    val name: String,
    val normalizedName: String,
    val type: EntityType,
    val attributes: Map<String, String>,
    val createdAt: Long,
    val lastUpdated: Long,
    val sourceMemoryIds: List<String>
)

data class RelationEdge(
    val id: String,
    val sourceId: String,
    val relation: String,
    val targetId: String,
    val confidence: Double,
    val createdAt: Long,
    val sourceMemoryId: String?,
    var superseded: Boolean,
    var supersededBy: String?
)

data class RelationResult(
    val success: Boolean,
    val edgeId: String? = null,
    val reason: String? = null,
    val conflictsDetected: Int = 0,
    val conflictingEdgeIds: List<String> = emptyList()
)

data class PathStep(
    val fromEntity: String,
    val relation: String,
    val toEntity: String,
    val confidence: Double
)

data class GraphPath(
    val steps: List<PathStep>,
    val totalConfidence: Double
) {
    override fun toString(): String {
        if (steps.isEmpty()) return "(empty)"
        val chain = steps.first().fromEntity + steps.joinToString("") { " →[${it.relation}]→ ${it.toEntity}" }
        return "$chain (conf: ${String.format("%.2f", totalConfidence)})"
    }
}

data class GraphQueryResult(
    val startEntity: String,
    val paths: List<GraphPath>,
    val entitiesVisited: Int,
    val maxDepthReached: Int
)

data class RelationSummary(
    val relation: String,
    val target: String,
    val confidence: Double,
    val isIncoming: Boolean = false
)

data class EntityRelations(
    val entity: EntityNode?,
    val outgoing: List<RelationSummary>,
    val incoming: List<RelationSummary>
)

data class IngestResult(
    val entitiesCreated: Int,
    val relationsCreated: Int,
    val conflictsDetected: Int,
    val totalEntities: Int,
    val totalRelations: Int
)

data class GraphStats(
    val totalEntities: Int,
    val totalRelations: Int,
    val totalConflicts: Int,
    val activeRelations: Int,
    val supersededRelations: Int
)
