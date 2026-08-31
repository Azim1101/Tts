plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.dhvaani.app"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.dhvaani.app"
        minSdk = 24
        targetSdk = 34
        versionCode = 6
        versionName = "1.1"
        // Only ship the 64-bit ARM ABI that real phones run. This drops the stale
        // x86_64/armeabi-v7a native libs, shrinking the APK and ensuring the
        // optimised arm64 XNNPACK/NNAPI .so is the one used.
        // For emulator testing add "x86_64" back on a debug-only variant.
        ndk {
            abiFilters += listOf("arm64-v8a")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
        debug {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        viewBinding = true
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.3")
    // MNN-only build. The native MNN runtime (libMNN.so + libmnncore.so) is
    // fetched by scripts/setup_mnn_runtime.sh into app/src/main/jniLibs/arm64-v8a
    // before the build. The Java bindings live in com.taobao.android.mnn.
    implementation(fileTree("libs") { include("*.jar", "*.aar") })
}
