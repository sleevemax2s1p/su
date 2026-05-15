package com.memory.embedding

/**
 * EmbeddingProvider — 向量嵌入抽象层
 *
 * 解决的问题：
 * 当前 MultiSignalRetriever 中 semantic score 来自外部候选（vector search 结果），
 * 但测试中使用 char-overlap 导致排序不准确。需要一个清晰的抽象层让：
 * 1. 测试使用可预测的 mock embeddings
 * 2. 生产使用真实 embedding model
 * 3. 接口统一，组件无需关心底层实现
 *
 * 设计原则：
 * - Interface + Strategy Pattern（可插拔）
 * - Batch embedding 支持（减少 API 调用）
 * - 缓存层（避免重复计算）
 * - 相似度计算统一（cosine similarity）
 *
 * 生产选项：
 * - OpenAI text-embedding-3-small (1536d)
 * - Cohere embed-multilingual-v3 (1024d)
 * - 本地 BGE-M3 / GTE-Large-zh
 * - 火山引擎 doubao-embedding
 */

// === Core Interface ===

interface EmbeddingProvider {
    /**
     * 将文本编码为向量
     */
    fun embed(text: String): FloatArray
    
    /**
     * 批量编码
     */
    fun embedBatch(texts: List<String>): List<FloatArray>
    
    /**
     * 向量维度
     */
    val dimension: Int
    
    /**
     * Provider 名称（用于 logging/debug）
     */
    val name: String
}

// === Similarity Calculator ===

object SimilarityCalculator {
    /**
     * Cosine similarity between two vectors
     */
    fun cosine(a: FloatArray, b: FloatArray): Double {
        require(a.size == b.size) { "Vectors must have same dimension" }
        
        var dotProduct = 0.0
        var normA = 0.0
        var normB = 0.0
        
        for (i in a.indices) {
            dotProduct += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        
        val denominator = Math.sqrt(normA) * Math.sqrt(normB)
        return if (denominator == 0.0) 0.0 else dotProduct / denominator
    }
    
    /**
     * Find top-K most similar vectors
     */
    fun topK(
        query: FloatArray,
        candidates: List<Pair<String, FloatArray>>,  // id → vector
        k: Int = 10
    ): List<Pair<String, Double>> {
        return candidates
            .map { (id, vec) -> id to cosine(query, vec) }
            .sortedByDescending { it.second }
            .take(k)
    }
}

// === Implementations ===

/**
 * CharOverlapProvider — 测试用（当前行为的正式封装）
 * 
 * 不是真正的 embedding，而是基于字符重叠的相似度
 * 优点：确定性、快速、无依赖
 * 缺点：不理解语义
 */
class CharOverlapProvider : EmbeddingProvider {
    override val dimension: Int = 0  // Not a real vector
    override val name: String = "char-overlap"
    
    override fun embed(text: String): FloatArray {
        // Not applicable — this provider works via direct similarity
        return FloatArray(0)
    }
    
    override fun embedBatch(texts: List<String>): List<FloatArray> {
        return texts.map { embed(it) }
    }
    
    /**
     * Direct similarity computation (bypasses embed/cosine)
     */
    fun computeSimilarity(query: String, content: String): Double {
        val qChars = query.toSet()
        val cChars = content.toSet()
        val intersection = qChars.intersect(cChars).size
        val union = qChars.union(cChars).size
        return if (union == 0) 0.0 else intersection.toDouble() / union
    }
}

/**
 * SemanticRuleProvider — 基于规则的语义相似度（测试用增强版）
 * 
 * 比 CharOverlap 更智能：
 * - 考虑关键词匹配
 * - 考虑同义词
 * - 考虑动作意图（"搬到X" 与 "住在哪" 的语义关联）
 * 
 * 用于测试中需要更准确排序的场景
 */
class SemanticRuleProvider : EmbeddingProvider {
    override val dimension: Int = 64  // Pseudo-dimension
    override val name: String = "semantic-rule"
    
    // Intent → keyword mappings for query understanding
    private val intentKeywords = mapOf(
        "location_current" to listOf("住", "在哪", "现在", "搬", "居住", "地址"),
        "location_past" to listOf("以前", "之前", "住过", "老家", "曾经"),
        "preference" to listOf("喜欢", "爱好", "偏好", "喜好", "最爱"),
        "health" to listOf("过敏", "病", "健康", "身体", "药"),
        "work" to listOf("工作", "上班", "公司", "项目", "职业"),
        "relationship" to listOf("朋友", "家人", "男朋友", "女朋友", "对象"),
    )
    
    // Action verbs that indicate state changes
    private val stateChangeVerbs = listOf("搬到", "换了", "开始", "不再", "改为")
    
    override fun embed(text: String): FloatArray {
        // Produce a pseudo-embedding based on intent detection
        val vec = FloatArray(dimension)
        var idx = 0
        
        for ((intent, keywords) in intentKeywords) {
            val score = keywords.count { kw -> text.contains(kw) }.toFloat() / keywords.size
            if (idx < dimension) vec[idx] = score
            idx++
        }
        
        // State change signal
        val hasStateChange = stateChangeVerbs.any { text.contains(it) }
        if (idx < dimension) vec[idx] = if (hasStateChange) 1.0f else 0.0f
        
        return vec
    }
    
    override fun embedBatch(texts: List<String>): List<FloatArray> {
        return texts.map { embed(it) }
    }
    
    /**
     * Rule-based semantic similarity
     * More accurate than char-overlap for memory retrieval scenarios
     */
    fun computeSimilarity(query: String, content: String): Double {
        var score = 0.0
        var signals = 0
        
        // 1. Intent alignment
        for ((_, keywords) in intentKeywords) {
            val queryHits = keywords.count { query.contains(it) }
            val contentHits = keywords.count { content.contains(it) }
            if (queryHits > 0 && contentHits > 0) {
                score += 0.3
                signals++
            }
        }
        
        // 2. Entity overlap
        val queryEntities = extractEntities(query)
        val contentEntities = extractEntities(content)
        val entityOverlap = queryEntities.intersect(contentEntities).size
        if (entityOverlap > 0) {
            score += 0.2 * entityOverlap
            signals++
        }
        
        // 3. Action-state relevance
        // "搬到X" answers "住在哪"
        if (query.contains("住") && stateChangeVerbs.any { content.contains(it) && content.contains("到") }) {
            score += 0.25
            signals++
        }
        
        // 4. Direct keyword match
        val queryBigrams = query.windowed(2).filter { it[0].code > 0x4E00 }.toSet()
        val contentBigrams = content.windowed(2).filter { it[0].code > 0x4E00 }.toSet()
        val bigramOverlap = queryBigrams.intersect(contentBigrams).size
        if (queryBigrams.isNotEmpty()) {
            score += 0.15 * bigramOverlap / queryBigrams.size
        }
        
        return score.coerceIn(0.0, 1.0)
    }
    
    private fun extractEntities(text: String): Set<String> {
        val entities = mutableSetOf<String>()
        val knownEntities = listOf("北京", "上海", "深圳", "广州", "杭州", "花生", "猫", "狗")
        for (e in knownEntities) {
            if (text.contains(e)) entities.add(e)
        }
        return entities
    }
}

/**
 * CachedEmbeddingProvider — 带缓存的 wrapper
 * 
 * 避免重复计算相同文本的 embedding
 */
class CachedEmbeddingProvider(
    private val delegate: EmbeddingProvider,
    private val maxCacheSize: Int = 10000
) : EmbeddingProvider {
    private val cache: LinkedHashMap<String, FloatArray> = object : LinkedHashMap<String, FloatArray>(
        maxCacheSize, 0.75f, true  // access-order for LRU
    ) {
        override fun removeEldestEntry(eldest: Map.Entry<String, FloatArray>): Boolean {
            return size > maxCacheSize
        }
    }
    
    override val dimension: Int = delegate.dimension
    override val name: String = "cached(${delegate.name})"
    
    override fun embed(text: String): FloatArray {
        return cache.getOrPut(text) { delegate.embed(text) }
    }
    
    override fun embedBatch(texts: List<String>): List<FloatArray> {
        return texts.map { embed(it) }
    }
    
    fun cacheSize(): Int = cache.size
    fun clearCache() = cache.clear()
}

/**
 * VectorIndex — 简化的向量索引
 * 
 * 在生产中会用 FAISS/Milvus/Qdrant，这里提供接口抽象
 */
class VectorIndex(
    private val provider: EmbeddingProvider
) {
    private val vectors: MutableMap<String, FloatArray> = mutableMapOf()
    
    /**
     * 添加文档到索引
     */
    fun add(id: String, text: String) {
        vectors[id] = provider.embed(text)
    }
    
    /**
     * 批量添加
     */
    fun addBatch(items: List<Pair<String, String>>) {
        val texts = items.map { it.second }
        val embeddings = provider.embedBatch(texts)
        items.forEachIndexed { idx, (id, _) ->
            vectors[id] = embeddings[idx]
        }
    }
    
    /**
     * 搜索最相似的 top-K
     */
    fun search(query: String, k: Int = 10): List<Pair<String, Double>> {
        val queryVec = provider.embed(query)
        return SimilarityCalculator.topK(queryVec, vectors.toList(), k)
    }
    
    /**
     * 索引大小
     */
    fun size(): Int = vectors.size
    
    /**
     * 删除（用于测试，生产中 ADD-only 不需要）
     */
    fun remove(id: String) {
        vectors.remove(id)
    }
}
