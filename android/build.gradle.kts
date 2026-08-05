// 根构建脚本
// AGP 9 内置 Kotlin：不再应用 org.jetbrains.kotlin.android；
// 如需高于 AGP 运行时的 KGP 版本，通过 buildscript classpath 声明
buildscript {
    dependencies {
        classpath("org.jetbrains.kotlin:kotlin-gradle-plugin:2.4.10")
    }
}

plugins {
    id("com.android.application") version "9.3.1" apply false
    id("org.jetbrains.kotlin.plugin.compose") version "2.4.10" apply false
    id("org.jetbrains.kotlin.plugin.serialization") version "2.4.10" apply false
}
