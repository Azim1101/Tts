package com.taobao.android.mnn;

import android.util.Log;

/**
 * JNI bridge to the `libmnncore.so` shipped by the MNN Android package.
 *
 * The methods and signatures match MNN's `project/android/demo`
 * (`mnnnetnative.cpp`) so the official `libmnncore.so` can provide the
 * implementation without any custom C++ code in this app.
 */
public class MNNNetNative {

    static {
        loadAndLog("MNN");
        loadAndLog("MNN_Vulkan");
        loadAndLog("MNN_CL");
        loadAndLog("MNN_GL");
        System.loadLibrary("mnncore");
    }

    private static void loadAndLog(String name) {
        try {
            System.loadLibrary(name);
        } catch (Throwable ce) {
            Log.w("DhVaani.Mnn", "load " + name + " failed: " + ce.getMessage());
        }
    }

    // Net
    protected static native long nativeCreateNetFromFile(String modelName);

    protected static native long nativeCreateNetFromBuffer(byte[] buffer);

    protected static native void nativeReleaseNet(long netPtr);

    // Session
    protected static native long nativeCreateSession(long netPtr, int forwardType, int numThread, String[] saveTensors, String[] outputTensors);

    protected static native void nativeReleaseSession(long netPtr, long sessionPtr);

    protected static native int nativeRunSession(long netPtr, long sessionPtr);

    protected static native int nativeReshapeSession(long netPtr, long sessionPtr);

    protected static native long nativeGetSessionInput(long netPtr, long sessionPtr, String name);

    protected static native long nativeGetSessionOutput(long netPtr, long sessionPtr, String name);

    // Tensor
    protected static native void nativeReshapeTensor(long netPtr, long tensorPtr, int[] dims);

    protected static native int[] nativeTensorGetDimensions(long tensorPtr);

    protected static native void nativeSetInputIntData(long netPtr, long tensorPtr, int[] data);

    protected static native void nativeSetInputFloatData(long netPtr, long tensorPtr, float[] data);

    // If dest is null, return length
    protected static native int nativeTensorGetData(long tensorPtr, float[] dest);

    protected static native int nativeTensorGetIntData(long tensorPtr, int[] dest);
}
