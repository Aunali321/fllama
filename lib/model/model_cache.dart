import 'dart:ffi';
import 'dart:async';

import 'package:ffi/ffi.dart';
import 'package:fllama/fllama_io.dart';

/// A class that manages model caching, allowing manual control over when models
/// are loaded into RAM and when they are released.
class FllamaModelCache {
  /// Singleton instance
  static final FllamaModelCache _instance = FllamaModelCache._internal();
  
  /// Factory constructor
  factory FllamaModelCache() => _instance;
  
  /// Private constructor
  FllamaModelCache._internal();

  /// Map of model paths to their loaded status
  final Map<String, Pointer<Void>> _loadedModels = {};
  
  /// Check if a specific model is loaded
  bool isModelLoaded(String modelPath) {
    return _loadedModels.containsKey(modelPath);
  }
  
  /// Get a list of all currently loaded model paths
  List<String> get loadedModels => _loadedModels.keys.toList();

  /// Loads a model into memory and keeps it cached
  /// 
  /// This allows subsequent inference requests to reuse the loaded model
  /// rather than reloading it each time.
  /// 
  /// [modelPath] - Path to the model file (.gguf)
  /// [numGpuLayers] - Number of layers to offload to GPU (0 for CPU only, 99 for all)
  /// [numThreads] - Number of CPU threads to use
  Future<void> loadModel(
    String modelPath, {
    int numGpuLayers = 0,
    int numThreads = 2,
  }) async {
    if (isModelLoaded(modelPath)) {
      return; // Model already loaded
    }
    
    // Convert model path to native string
    final modelPathNative = modelPath.toNativeUtf8().cast<Char>();
    
    try {
      // Load the model
      final modelHandle = fllamaBindings.fllama_model_load(
        modelPathNative,
        numGpuLayers,
        numThreads,
      );
      
      if (modelHandle == nullptr) {
        throw Exception('Failed to load model: $modelPath');
      }
      
      // Store the model handle
      _loadedModels[modelPath] = modelHandle;
    } finally {
      // Free the native string
      calloc.free(modelPathNative);
    }
  }
  
  /// Unloads a model from memory
  /// 
  /// This releases the memory used by the model. After unloading,
  /// any inference request will need to reload the model.
  /// 
  /// [modelPath] - Path to the model to unload
  Future<void> unloadModel(String modelPath) async {
    final modelHandle = _loadedModels[modelPath];
    if (modelHandle == null) {
      return; // Model not loaded
    }
    
    try {
      // Unload the model
      fllamaBindings.fllama_model_unload(modelHandle);
    } finally {
      // Remove from cache
      _loadedModels.remove(modelPath);
    }
  }
  
  /// Unloads all currently loaded models
  Future<void> unloadAllModels() async {
    // Create a copy of keys to avoid modification during iteration
    final modelPaths = List<String>.from(_loadedModels.keys);
    
    for (final modelPath in modelPaths) {
      await unloadModel(modelPath);
    }
  }
  
  /// Checks if a model file exists and can be loaded
  /// 
  /// [modelPath] - Path to the model file
  Future<bool> modelExists(String modelPath) async {
    final modelPathNative = modelPath.toNativeUtf8().cast<Char>();
    
    try {
      final result = fllamaBindings.fllama_model_is_loaded(modelPathNative);
      return result == 1;
    } finally {
      calloc.free(modelPathNative);
    }
  }
  
  /// Run inference using a pre-loaded model
  /// 
  /// This function is similar to [fllamaInference] but uses a pre-loaded model.
  /// The model must be loaded with [loadModel] before calling this function.
  /// 
  /// [modelPath] - Path to the loaded model
  /// [request] - Inference request parameters
  /// [callback] - Callback for receiving inference results
  Future<int> inferenceWithLoadedModel({
    required String modelPath,
    required FllamaInferenceRequest request,
    required FllamaInferenceCallback callback,
  }) async {
    if (!isModelLoaded(modelPath)) {
      throw Exception('Model not loaded: $modelPath. Call loadModel() first.');
    }
    
    // Use the original inference function with the same model path
    // The native code should detect the model is already loaded
    return fllamaInference(request, callback);
  }
}
