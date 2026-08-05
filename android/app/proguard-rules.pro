# kotlinx.serialization
-keepattributes *Annotation*, InnerClasses
-keep class kotlinx.serialization.** { *; }
-keepclassmembers class com.tzb.safeguard.data.model.** { *** Companion; }
-keepclasseswithmembers class com.tzb.safeguard.data.model.** { kotlinx.serialization.KSerializer serializer(...); }
# Retrofit / OkHttp
-keepattributes Signature, Exceptions
-dontwarn okhttp3.**
-dontwarn retrofit2.**
