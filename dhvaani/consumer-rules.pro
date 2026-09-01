# JNI calls these by name via GetMethodID / RegisterNatives.
-keep class zone.dhvaani.tts.DhVaani { *; }
-keep interface zone.dhvaani.tts.DhVaani$ProgressListener { *; }
-keepclasseswithmembernames class * { native <methods>; }
