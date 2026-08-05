plugins {
    id("com.android.application")
    // AGP 9 内置 Kotlin，无需 org.jetbrains.kotlin.android
    id("org.jetbrains.kotlin.plugin.compose")
    id("org.jetbrains.kotlin.plugin.serialization")
}

val apiBaseUrl = providers.gradleProperty("API_BASE_URL")
    .getOrElse("http://127.0.0.1:8000/")

android {
    namespace = "com.tzb.safeguard"
    compileSdk = 37

    defaultConfig {
        applicationId = "com.tzb.safeguard"
        minSdk = 26
        targetSdk = 37
        versionCode = 1
        versionName = "0.1.0"

        // 默认配合 adb reverse；真机可用 -PAPI_BASE_URL=http://局域网IP:8000/
        buildConfigField("String", "API_BASE_URL", "\"$apiBaseUrl\"")
        // Mock 开关：true 时由 MockInterceptor 返回本地数据，false 走真实后端
        buildConfigField("boolean", "MOCK_MODE", "false")

        ndk {
            abiFilters += listOf("armeabi-v7a", "arm64-v8a")
        }
    }

    buildTypes {
        debug {
            buildConfigField("boolean", "MOCK_MODE", "false")
        }
        release {
            isMinifyEnabled = false
            buildConfigField("boolean", "MOCK_MODE", "false")
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlin {
        compilerOptions {
            jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
        }
    }
}

dependencies {
    // Compose BOM 统一 Compose 家族版本
    implementation(platform("androidx.compose:compose-bom:2026.05.01"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-extended")

    implementation("androidx.core:core-ktx:1.19.0")
    implementation("androidx.activity:activity-compose:1.13.0")
    implementation("androidx.navigation:navigation-compose:2.9.8")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.11.0")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.11.0")

    // 网络层：Retrofit + OkHttp + kotlinx.serialization
    implementation("com.squareup.retrofit2:retrofit:2.11.0")
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation("com.squareup.okhttp3:logging-interceptor:4.12.0")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.11.0")
    implementation("com.jakewharton.retrofit:retrofit2-kotlinx-serialization-converter:1.0.0")

    // 萤石原生直播：支持设备原始 H.264/H.265 码流
    implementation("io.github.ezviz-open:ezviz-sdk:5.30.2")
    implementation("com.google.code.gson:gson:2.13.2")

    debugImplementation("androidx.compose.ui:ui-tooling")
}
