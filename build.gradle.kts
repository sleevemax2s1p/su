plugins {
    kotlin("jvm") version "1.9.22"
    kotlin("plugin.serialization") version "1.9.22"
    application
}

group = "com.memory"
version = "6.1.0"

repositories {
    mavenCentral()
}

dependencies {
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.7.3")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.2")
}

application {
    mainClass.set("EvalV17FullKt")
}

tasks.withType<org.jetbrains.kotlin.gradle.tasks.KotlinCompile> {
    kotlinOptions.jvmTarget = "11"
}

// 单独编译 eval（跳过其他有依赖缺失的模块）
sourceSets {
    main {
        kotlin {
            // 只编译 eval 文件（其他模块有未实现的接口依赖）
            setSrcDirs(listOf("src/main/kotlin"))
            include("EvalV17Full.kt")
        }
    }
}
