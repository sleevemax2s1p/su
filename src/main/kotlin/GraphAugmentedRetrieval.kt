package com.memory.retrieval

import com.memory.graph.EntityRelationGraph
import com.memory.graph.EntityType
import com.memory.graph.GraphPath
import com.memory.graph.GraphQueryResult

/**
 * 图增强检索 (Graph-Augmented Retrieval)
 * 
 * 将 HybridRetriever 和 EntityRelationGraph 统一为一个检索入口。
 * 
 * 检索策略：
 * 1. 先用 Hybrid Retrieval (Vector+BM25) 获取直接相关记忆
 * 2. 从检索结果中提取命中的实体
 * 3. 对这些实体执行 Graph Multi-hop Query
 * 4. 将 Graph 路径转为补充上下文注入到结果中
 * 
 * 何时启用 Graph：
 * - 问题涉及关系推理 ("小红在哪工作？", "用户公司的地址？")
 * - 问题涉及实体属性 ("用户的女朋友叫什么？")
 * - 检索结果中出现已知实体名
 * 
 * 何时不启用 Graph (节省延迟)：
 * - 简单情感表达 ("今天心情好")
 * - 无实体名词的查询
 * - 纯近期对话查询
 * 
 * 性能预期 (基于 Mem0g benchmark):
 * - 单跳问题: Graph 几乎无增益 (Vector 已够好)
 * - 多跳问题: +5~10% accuracy
 * - 延迟开销: +50~100ms (in-memory graph, 无外部调用)
 */
class GraphAugmentedRetrieval(
    private val hybridRetriever: HybridRetriever,
    private val graph: EntityRelationGraph,
    private val maxGraphHops: Int = 2,
    private val maxGraphPaths: Int = 5,
    private val graphBoostFactor: Double = 1.15  // Graph 命中的额外加权
) {
    
    /**
     * 统一检索入口
     * 
     * @param query 用户查询
     * @param vectorResults 向量检索候选
     * @param bm25Results BM25 检索候选
     * @param enableGraph 是否启用图增强
     * @return 融合结果 + Graph 补充上下文
     */
    fun retrieve(
        query: String,
        vectorResults: List<RetrievalCandidate>,
        bm25Results: List<RetrievalCandidate>,
        enableGraph: Boolean = true
    ): AugmentedRetrievalResult {
        // Step 1: 基础 Hybrid Retrieval
        val baseResults = hybridRetriever.retrieve(query, vectorResults, bm25Results)
        
        if (!enableGraph) {
            return AugmentedRetrievalResult(
                rankedResults = baseResults,
                graphContext = emptyList(),
                graphEnabled = false
            )
        }
        
        // Step 2: 从查询和结果中提取实体
        val mentionedEntities = extractEntityMentions(query, baseResults)
        
        if (mentionedEntities.isEmpty()) {
            return AugmentedRetrievalResult(
                rankedResults = baseResults,
                graphContext = emptyList(),
                graphEnabled = true,
                graphTriggered = false
            )
        }
        
        // Step 3: Multi-hop Graph Query
        val graphPaths = mutableListOf<GraphPath>()
        for (entityId in mentionedEntities) {
            val queryResult = graph.multiHopQuery(
                startEntityId = entityId,
                maxHops = maxGraphHops
            )
            graphPaths.addAll(queryResult.paths.take(maxGraphPaths))
        }
        
        // Step 4: Graph boost — 如果某个检索结果包含 Graph 中的实体，加权
        val boostedResults = baseResults.map { result ->
            val containsGraphEntity = mentionedEntities.any { entityId ->
                val entity = graph.findEntityByName(extractEntityName(entityId))
                entity != null && result.content.contains(entity.name)
            }
            
            if (containsGraphEntity) {
                result.copy(rrfScore = result.rrfScore * graphBoostFactor)
            } else {
                result
            }
        }.sortedByDescending { it.rrfScore }
        
        // Step 5: 生成 Graph 上下文描述
        val graphContext = graphPaths
            .distinctBy { it.toString() }
            .take(maxGraphPaths)
            .map { path ->
                GraphContextEntry(
                    path = path.toString(),
                    confidence = path.totalConfidence,
                    hops = path.steps.size
                )
            }
        
        return AugmentedRetrievalResult(
            rankedResults = boostedResults,
            graphContext = graphContext,
            graphEnabled = true,
            graphTriggered = true,
            entitiesFound = mentionedEntities.size
        )
    }
    
    /**
     * 从查询和检索结果中提取实体 mention
     * 
     * 策略：在 query 中查找已知实体名
     */
    private fun extractEntityMentions(
        query: String,
        results: List<RankedResult>
    ): List<String> {
        val found = mutableListOf<String>()
        
        // 在 query 中查找已知实体
        // (实际生产中应该用 NER, 这里用简单的名称匹配)
        val allText = query + " " + results.take(3).joinToString(" ") { it.content }
        
        // Graph 的 findEntityByName 已支持 name index 查找
        // 这里遍历 graph 中的所有实体做 mention detection
        val entityNames = graph.getAllEntityNames()
        for (name in entityNames) {
            if (name.length >= 2 && name in allText) {
                val entityId = graph.findEntityIdByName(name)
                if (entityId != null) {
                    found.add(entityId)
                }
            }
        }
        
        return found.distinct()
    }
    
    private fun extractEntityName(entityId: String): String {
        return graph.getEntityName(entityId) ?: ""
    }
    
    /**
     * 判断是否应该启用 Graph (启发式规则)
     * 
     * 可以被调用方用来决定 enableGraph 参数
     */
    fun shouldEnableGraph(query: String): Boolean {
        // 关系查询关键词
        val relationKeywords = listOf(
            "谁", "哪里", "什么时候", "在哪", "叫什么",
            "的", "和", "跟", "关系", "认识",
            "工作", "住", "公司", "朋友", "家人"
        )
        
        // 简单情感/闲聊不启用
        val casualKeywords = listOf(
            "嗯", "好的", "哈哈", "心情", "感觉", "天气",
            "你好", "再见", "谢谢"
        )
        
        val hasRelation = relationKeywords.any { it in query }
        val isCasual = casualKeywords.any { it in query } && query.length < 10
        
        return hasRelation && !isCasual
    }
}

// === Extended Graph interface methods (needed by this module) ===
// These should be added to EntityRelationGraph.kt

fun EntityRelationGraph.getAllEntityNames(): List<String> {
    return getStats().let { stats ->
        // This is a simplified accessor; in production, graph exposes a name list
        emptyList() // Placeholder - actual implementation traverses nameIndex
    }
}

fun EntityRelationGraph.findEntityIdByName(name: String): String? {
    return findEntityByName(name)?.id
}

fun EntityRelationGraph.getEntityName(entityId: String): String? {
    // Accessor for entity name by ID
    return null // Placeholder
}

// === Data Classes ===

data class AugmentedRetrievalResult(
    val rankedResults: List<RankedResult>,
    val graphContext: List<GraphContextEntry>,
    val graphEnabled: Boolean,
    val graphTriggered: Boolean = false,
    val entitiesFound: Int = 0
) {
    /**
     * 生成可注入 prompt 的 graph 上下文字符串
     */
    fun buildGraphContextString(): String {
        if (graphContext.isEmpty()) return ""
        
        return buildString {
            appendLine("【关系图谱】")
            for (entry in graphContext) {
                appendLine("• ${entry.path}")
            }
        }
    }
}

data class GraphContextEntry(
    val path: String,
    val confidence: Double,
    val hops: Int
)
