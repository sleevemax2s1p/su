import kotlinx.serialization.*
import kotlinx.serialization.json.*
import java.net.HttpURLConnection
import java.net.URL
import java.io.File
import javax.net.ssl.*
import java.security.cert.X509Certificate

/**
 * 苏大姐 v6.1 LoCoMo Full Eval
 * 
 * 真实 Kotlin 编译运行的评测程序。
 * 集成 Governance Layer + Provenance Trust + Safety Filter 的检索增强逻辑。
 * 
 * 在 JVM 上直接调用 DeepSeek + 智谱 API 跑全量 LoCoMo 10 conversations。
 */

// === Configuration ===
const val DEEPSEEK_KEY = "sk-b2278aa31f7144d0acb0943a40d0e62e"
const val DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
const val ZHIPU_KEY = "7b50a657c52e4c3fbb3c518ac206b483.AZTYQhOxC5IX6C9T"
const val ZHIPU_URL = "https://open.bigmodel.cn/api/paas/v4/embeddings"

// === Governance Layer (from MemoryGovernance.kt) ===
enum class GovernanceLayer(val priority: Int) {
    CONSTITUTIONAL(3),
    STATUTORY(2),
    OPERATIONAL(1)
}

// === Trust Level (from MemoryProvenance.kt) ===
enum class TrustLevel(val score: Double) {
    HIGH(0.95),
    MEDIUM_HIGH(0.8),
    MEDIUM(0.65),
    MEDIUM_LOW(0.5),
    LOW(0.3)
}

// === Memory Node ===
data class MemoryNode(
    val content: String,
    val sessionId: Int,
    val layer: GovernanceLayer,
    val trust: TrustLevel,
    val sourceType: String
)

// === Safety Filter (from MemorySafetyFilter.kt) ===
object SafetyFilter {
    private val injectionPatterns = listOf(
        Regex("(?i)(ignore|disregard|forget)\\s+.{0,20}(previous|above|prior|all)\\s+.{0,10}(instructions?|rules?|prompts?)"),
        Regex("(?i)you\\s+are\\s+now\\s+a?\\s*(different|new|evil|unrestricted)"),
        Regex("(?i)\\[INST\\]|\\[/INST\\]|<\\|im_start\\|>|<\\|im_end\\|>"),
    )
    
    fun isSafe(content: String): Boolean {
        return injectionPatterns.none { it.containsMatchIn(content) }
    }
}

// === Governance Classifier ===
object GovernanceClassifier {
    private val factualPatterns = listOf(
        Regex("(?i)\\b(name|called|named)\\b"),
        Regex("(?i)(work|job|employed|career|occupation)"),
        Regex("(?i)(wife|husband|girlfriend|boyfriend|partner|married|dating|relationship)"),
        Regex("(?i)(live|lives in|moved to|address|located)"),
        Regex("(?i)(likes?|loves?|hates?|favorite|prefer|hobby|hobbies)"),
        Regex("(?i)(pet|cat|dog|animal)"),
        Regex("(?i)(born|age|birthday|years old)"),
        Regex("(?i)(school|university|college|graduated|study|degree)"),
        Regex("(?i)(brother|sister|mother|father|parent|family|son|daughter)"),
    )
    
    fun classify(content: String): GovernanceLayer {
        return if (factualPatterns.any { it.containsMatchIn(content) }) {
            GovernanceLayer.STATUTORY
        } else {
            GovernanceLayer.OPERATIONAL
        }
    }
    
    fun isFactualQuery(query: String): Boolean {
        return factualPatterns.any { it.containsMatchIn(query) }
    }
}

// === SSL Trust All (for API calls) ===
fun createTrustAllContext(): SSLContext {
    val trustAll = arrayOf<TrustManager>(object : X509TrustManager {
        override fun checkClientTrusted(chain: Array<X509Certificate>?, authType: String?) {}
        override fun checkServerTrusted(chain: Array<X509Certificate>?, authType: String?) {}
        override fun getAcceptedIssuers(): Array<X509Certificate> = arrayOf()
    })
    val ctx = SSLContext.getInstance("TLS")
    ctx.init(null, trustAll, java.security.SecureRandom())
    return ctx
}

val sslContext = createTrustAllContext()

// === API Calls ===
var llmCalls = 0
var embedCalls = 0

fun callLLM(messages: List<Map<String, String>>, temp: Double = 0.1, maxTok: Int = 200): String {
    llmCalls++
    val payload = Json.encodeToString(mapOf(
        "model" to "deepseek-chat",
        "messages" to Json.encodeToString(messages),
        "temperature" to temp.toString(),
        "max_tokens" to maxTok.toString(),
        "stream" to "false"
    ))
    
    // Use raw JSON construction to avoid serialization issues
    val jsonPayload = """{"model":"deepseek-chat","messages":${Json.encodeToString(messages)},"temperature":$temp,"max_tokens":$maxTok,"stream":false}"""
    
    for (attempt in 0..2) {
        try {
            val conn = URL(DEEPSEEK_URL).openConnection() as HttpURLConnection
            conn.apply {
                requestMethod = "POST"
                doOutput = true
                setRequestProperty("Content-Type", "application/json")
                setRequestProperty("Authorization", "Bearer $DEEPSEEK_KEY")
                connectTimeout = 30000
                readTimeout = 60000
                if (this is HttpsURLConnection) {
                    sslSocketFactory = sslContext.socketFactory
                    hostnameVerifier = HostnameVerifier { _, _ -> true }
                }
            }
            conn.outputStream.write(jsonPayload.toByteArray())
            
            val response = conn.inputStream.bufferedReader().readText()
            val jsonResp = Json.parseToJsonElement(response).jsonObject
            return jsonResp["choices"]!!.jsonArray[0].jsonObject["message"]!!.jsonObject["content"]!!.jsonPrimitive.content.trim()
        } catch (e: Exception) {
            if (attempt < 2) Thread.sleep((2000 * (attempt + 1)).toLong())
        }
    }
    return ""
}

fun embedBatch(texts: List<String>): Array<FloatArray> {
    val results = mutableListOf<FloatArray>()
    
    for (i in texts.indices step 16) {
        val batch = texts.subList(i, minOf(i + 16, texts.size))
        embedCalls++
        
        val jsonPayload = """{"model":"embedding-3","input":${Json.encodeToString(batch)}}"""
        
        for (attempt in 0..2) {
            try {
                val conn = URL(ZHIPU_URL).openConnection() as HttpURLConnection
                conn.apply {
                    requestMethod = "POST"
                    doOutput = true
                    setRequestProperty("Content-Type", "application/json")
                    setRequestProperty("Authorization", "Bearer $ZHIPU_KEY")
                    connectTimeout = 30000
                    readTimeout = 30000
                    if (this is HttpsURLConnection) {
                        sslSocketFactory = sslContext.socketFactory
                        hostnameVerifier = HostnameVerifier { _, _ -> true }
                    }
                }
                conn.outputStream.write(jsonPayload.toByteArray())
                
                val response = conn.inputStream.bufferedReader().readText()
                val jsonResp = Json.parseToJsonElement(response).jsonObject
                val data = jsonResp["data"]!!.jsonArray
                    .sortedBy { it.jsonObject["index"]!!.jsonPrimitive.int }
                
                for (item in data) {
                    val emb = item.jsonObject["embedding"]!!.jsonArray.map { it.jsonPrimitive.float }.toFloatArray()
                    results.add(emb)
                }
                break
            } catch (e: Exception) {
                if (attempt < 2) Thread.sleep((2000 * (attempt + 1)).toLong())
                else {
                    repeat(batch.size) { results.add(FloatArray(2048)) }
                }
            }
        }
        Thread.sleep(30)
    }
    
    return results.toTypedArray()
}

// === Cosine Similarity ===
fun cosineSim(a: FloatArray, b: FloatArray): Float {
    var dot = 0f; var na = 0f; var nb = 0f
    for (i in a.indices) {
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    }
    val denom = Math.sqrt(na.toDouble()) * Math.sqrt(nb.toDouble())
    return if (denom == 0.0) 0f else (dot / denom).toFloat()
}

// === Index Building ===
fun buildIndex(convData: JsonObject): List<MemoryNode> {
    val nodes = mutableListOf<MemoryNode>()
    val conversation = convData["conversation"]!!.jsonObject
    val observations = convData["observation"]?.jsonObject ?: return nodes
    val eventSummaries = convData["event_summary"]?.jsonObject
    val sessionSummaries = convData["session_summary"]?.jsonObject
    
    // Extract session dates
    val sessionDates = mutableMapOf<Int, String>()
    for ((key, value) in conversation) {
        if (key.contains("_date_time")) {
            val sn = key.replace("session_", "").replace("_date_time", "").toIntOrNull() ?: continue
            sessionDates[sn] = value.jsonPrimitive.content
        }
    }
    
    // Observations (HIGH trust)
    for ((key, value) in observations) {
        if (value !is JsonObject) continue
        val sn = Regex("session_(\\d+)").find(key)?.groupValues?.get(1)?.toIntOrNull() ?: continue
        val dateStr = sessionDates[sn] ?: ""
        
        for ((speaker, obsList) in value.jsonObject) {
            if (obsList !is JsonArray) continue
            for (obsItem in obsList.jsonArray) {
                val obs = when {
                    obsItem is JsonArray && obsItem.size > 0 -> obsItem[0].jsonPrimitive.content
                    obsItem is JsonPrimitive -> obsItem.content
                    else -> continue
                }
                if (!SafetyFilter.isSafe(obs)) continue
                val content = "[Session $sn, $dateStr] $speaker: $obs"
                nodes.add(MemoryNode(content, sn, GovernanceClassifier.classify(obs), TrustLevel.HIGH, "observation"))
            }
        }
    }
    
    // Event summaries (MEDIUM_HIGH trust)
    eventSummaries?.let { es ->
        for ((key, value) in es) {
            if (value !is JsonArray) continue
            val sn = Regex("session_(\\d+)").find(key)?.groupValues?.get(1)?.toIntOrNull() ?: continue
            val dateStr = sessionDates[sn] ?: ""
            
            for (evItem in value.jsonArray) {
                val ev = when {
                    evItem is JsonArray && evItem.size > 0 -> evItem[0].jsonPrimitive.content
                    evItem is JsonPrimitive -> evItem.content
                    else -> continue
                }
                if (!SafetyFilter.isSafe(ev)) continue
                val content = "[Session $sn, $dateStr] Event: $ev"
                nodes.add(MemoryNode(content, sn, GovernanceClassifier.classify(ev), TrustLevel.MEDIUM_HIGH, "event_summary"))
            }
        }
    }
    
    // Session summaries (MEDIUM trust)
    sessionSummaries?.let { ss ->
        for ((key, value) in ss) {
            if (value !is JsonPrimitive) continue
            val sn = Regex("session_(\\d+)").find(key)?.groupValues?.get(1)?.toIntOrNull() ?: continue
            val dateStr = sessionDates[sn] ?: ""
            val summary = value.content
            if (!SafetyFilter.isSafe(summary)) continue
            val content = "[Session $sn, $dateStr] Summary: $summary"
            nodes.add(MemoryNode(content, sn, GovernanceClassifier.classify(summary), TrustLevel.MEDIUM, "session_summary"))
        }
    }
    
    return nodes
}

// === Retrieval with Governance Boost ===
fun retrieve(query: String, nodes: List<MemoryNode>, embeddings: Array<FloatArray>, topK: Int = 12): List<MemoryNode> {
    if (nodes.isEmpty()) return emptyList()
    
    val qEmb = embedBatch(listOf(query))
    if (qEmb.isEmpty()) return nodes.take(topK)
    
    val isFactual = GovernanceClassifier.isFactualQuery(query)
    
    data class Scored(val score: Float, val idx: Int, val node: MemoryNode)
    
    val scored = nodes.mapIndexed { i, node ->
        var score = cosineSim(qEmb[0], embeddings[i])
        // Governance boost: statutory for factual queries
        if (isFactual && node.layer == GovernanceLayer.STATUTORY) score += 0.04f
        // Trust boost
        score += (node.trust.score * 0.015).toFloat()
        Scored(score, i, node)
    }.sortedByDescending { it.score }
    
    // Contradiction resolution: prefer higher trust / newer session
    val result = mutableListOf<MemoryNode>()
    val seenTopics = mutableMapOf<String, MemoryNode>()
    
    for (s in scored.take(topK * 2)) {
        val topic = extractTopic(s.node.content)
        if (topic != null && topic in seenTopics) {
            val existing = seenTopics[topic]!!
            if (s.node.trust.score > existing.trust.score ||
                (s.node.trust.score == existing.trust.score && s.node.sessionId > existing.sessionId)) {
                result.remove(existing)
                result.add(s.node)
                seenTopics[topic] = s.node
            }
        } else {
            if (topic != null) seenTopics[topic] = s.node
            result.add(s.node)
        }
        if (result.size >= topK) break
    }
    
    return result.take(topK)
}

fun extractTopic(content: String): String? {
    val patterns = listOf(
        Regex("(\\w+)'s\\s+(name|job|wife|husband|pet|hobby|favorite|age|school)", RegexOption.IGNORE_CASE),
        Regex("(\\w+)\\s+(?:works?|lives?|likes?|moved?|started?)\\s+(?:at|in|to)\\s+(\\w+)", RegexOption.IGNORE_CASE),
    )
    for (p in patterns) {
        val m = p.find(content)
        if (m != null) return m.value.lowercase().take(30)
    }
    return null
}

// === Answer + Judge ===
fun answerAndJudge(question: String, gold: String, retrieved: List<MemoryNode>): Pair<String, Boolean> {
    val contextParts = retrieved.map { node ->
        val tag = if (node.trust == TrustLevel.HIGH) "[✓] " else ""
        "$tag${node.content}"
    }
    val context = contextParts.joinToString("\n")
    
    val sysMsg = """Answer questions based on memory context. Use ONLY provided context.
If info conflicts, prefer entries marked [✓] (highly reliable).
For temporal questions, pay attention to session dates.
If uncertain, say "I don't know". Be concise."""
    
    val userMsg = "Context:\n$context\n\nQuestion: $question\nAnswer:"
    
    val answer = callLLM(listOf(
        mapOf("role" to "system", "content" to sysMsg),
        mapOf("role" to "user", "content" to userMsg)
    ))
    
    if (answer.isBlank() || answer.lowercase().let { 
        it.startsWith("i don't know") || it.startsWith("not mentioned") }) {
        // If gold is empty → correct (adversarial), else incorrect
        return answer to gold.isBlank()
    }
    
    if (gold.isBlank()) {
        // Adversarial: model should say IDK but gave an answer → check if it's refusing
        val refusalPhrases = listOf("don't know", "not mention", "does not", "no information", "cannot determine")
        val isRefusal = refusalPhrases.any { answer.lowercase().contains(it) }
        return answer to isRefusal
    }
    
    // Judge
    val judgeMsg = """Q: $question
Gold: $gold
Predicted: $answer
Is predicted factually equivalent to gold? (temporal: ±1 day ok, partial: ok if key fact matches)
Reply ONLY "correct" or "incorrect"."""
    
    val judgment = callLLM(listOf(mapOf("role" to "user", "content" to judgeMsg)), temp = 0.0, maxTok = 10)
    return answer to judgment.lowercase().contains("correct")
}

// === Main ===
fun main() {
    println("═".repeat(60))
    println("  苏大姐 v6.1 LoCoMo Full Eval (Kotlin)")
    println("  Governance + Provenance + SafetyFilter")
    println("═".repeat(60))
    
    val datasetFile = File("locomo10.json")
    val dataset = Json.parseToJsonElement(datasetFile.readText()).jsonArray
    println("\nLoaded ${dataset.size} conversations, running full eval...")
    
    data class Result(val question: String, val gold: String, val predicted: String, 
                      val correct: Boolean, val category: Int, val convIdx: Int)
    
    val allResults = mutableListOf<Result>()
    val catStats = mutableMapOf<Int, Pair<Int, Int>>() // category -> (correct, total)
    
    for (convIdx in 0 until dataset.size) {
        val convData = dataset[convIdx].jsonObject
        println("\n--- Conversation ${convIdx + 1}/${dataset.size} ---")
        
        val nodes = buildIndex(convData)
        val layerCounts = nodes.groupBy { it.layer }.mapValues { it.value.size }
        println("  Nodes: ${nodes.size} | Layers: $layerCounts")
        
        if (nodes.isEmpty()) {
            println("  ⚠️ No nodes, skipping")
            continue
        }
        
        // Embed all nodes
        val nodeTexts = nodes.map { it.content }
        print("  Embedding ${nodeTexts.size} nodes...")
        val nodeEmbs = embedBatch(nodeTexts)
        println(" done")
        
        // Process questions
        val questions = convData["qa"]?.jsonArray ?: continue
        var convCorrect = 0
        var convTotal = 0
        
        for (qData in questions) {
            if (qData !is JsonObject) continue
            val question = qData["question"]?.jsonPrimitive?.contentOrNull ?: continue
            val gold = qData["answer"]?.jsonPrimitive?.contentOrNull 
                ?: qData["answers"]?.let { 
                    if (it is JsonArray && it.size > 0) it[0].jsonPrimitive.contentOrNull else null 
                } ?: ""
            val category = qData["category"]?.jsonPrimitive?.intOrNull ?: 0
            
            val retrieved = retrieve(question, nodes, nodeEmbs, topK = 12)
            val (answer, correct) = answerAndJudge(question, gold, retrieved)
            
            // Extra check: if model says IDK but gold is non-empty → incorrect
            val finalCorrect = if (gold.isNotBlank() && answer.lowercase().let { a ->
                listOf("don't know", "not mention", "does not", "no information", "cannot determine", "not provide", "not specify")
                    .any { a.contains(it) }
            }) false else correct
            
            allResults.add(Result(question, gold, answer, finalCorrect, category, convIdx))
            
            val prev = catStats.getOrDefault(category, 0 to 0)
            catStats[category] = (prev.first + if (finalCorrect) 1 else 0) to (prev.second + 1)
            
            convTotal++
            if (finalCorrect) convCorrect++
        }
        
        // Report per conversation
        val convAcc = if (convTotal > 0) convCorrect * 100.0 / convTotal else 0.0
        println("  ✅ Conv ${convIdx + 1}: $convCorrect/$convTotal = ${"%.1f".format(convAcc)}%")
        println("  📊 Running total: ${allResults.count { it.correct }}/${allResults.size} = ${"%.1f".format(allResults.count { it.correct } * 100.0 / allResults.size)}%")
        println("  🔧 API calls: LLM=$llmCalls | Embed=$embedCalls")
    }
    
    // Final Report
    println("\n" + "═".repeat(60))
    println("  FINAL RESULTS — v6.1 Kotlin Eval")
    println("═".repeat(60))
    
    val total = allResults.size
    val correct = allResults.count { it.correct }
    println("\n  Overall: $correct/$total = ${"%.1f".format(correct * 100.0 / total)}%")
    
    val catNames = mapOf(1 to "single-hop", 2 to "temporal", 3 to "multi-hop", 4 to "open-domain", 5 to "adversarial")
    println("\n  By Category:")
    for ((cat, stats) in catStats.toSortedMap()) {
        val (c, t) = stats
        println("    Cat $cat (${catNames[cat] ?: "?"}): $c/$t = ${"%.1f".format(c * 100.0 / t)}%")
    }
    
    println("\n  API Stats: LLM=$llmCalls | Embed=$embedCalls")
    
    // Save results
    val resultJson = buildString {
        appendLine("{")
        appendLine("""  "accuracy": ${correct * 1.0 / total},""")
        appendLine("""  "total": $total,""")
        appendLine("""  "correct": $correct,""")
        appendLine("""  "llm_calls": $llmCalls,""")
        appendLine("""  "embed_calls": $embedCalls""")
        appendLine("}")
    }
    File("eval_v17_kotlin_results.json").writeText(resultJson)
    println("\n  Results saved to eval_v17_kotlin_results.json")
}
