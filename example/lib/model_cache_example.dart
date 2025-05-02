import 'package:fllama/fllama_io.dart';
import 'package:fllama/fllama_universal.dart';

/// Example demonstrating how to use the FllamaModelCache to manually control
/// when models are loaded and unloaded from memory.
void main() async {
  print('FllamaModelCache Example');
  
  // Access the model cache singleton
  final modelCache = FllamaModelCache();
  
  // Example model path - update this to point to your model
  final modelPath = '/path/to/your/model.gguf';
  
  try {
    // Manually load the model into memory
    print('Loading model: $modelPath');
    await modelCache.loadModel(
      modelPath,
      numGpuLayers: 0,  // 0 for CPU only, 99 for all available GPU layers
      numThreads: 2,    // Number of CPU threads to use
    );
    print('Model loaded successfully');
    
    // Check if model is loaded
    print('Model loaded: ${modelCache.isModelLoaded(modelPath)}');
    
    // Show all loaded models
    print('All loaded models: ${modelCache.loadedModels}');
    
    // Run inference with the loaded model
    // The model won't be loaded again for each inference request
    for (int i = 0; i < 3; i++) {
      print('\nRunning inference #${i + 1}');
      
      final request = FllamaInferenceRequest(
        contextSize: 4096,
        input: 'Tell me a short joke.',
        maxTokens: 50,
        modelPath: modelPath,
        numGpuLayers: 0,
        penaltyFrequency: 0,
        penaltyRepeat: 1.1,
        temperature: 0.7,
        topP: 0.9,
        numThreads: 2,
      );
      
      await fllamaInference(request, (response, openaiResponseJsonString, done) {
        if (done) {
          print('Inference complete: $response');
        } else {
          // Streaming response
          print('Token: $response');
        }
      });
      
      // Small delay between inferences
      await Future.delayed(Duration(seconds: 1));
    }
    
    // Unload the model when you're done to free memory
    print('\nUnloading model: $modelPath');
    await modelCache.unloadModel(modelPath);
    print('Model unloaded: ${!modelCache.isModelLoaded(modelPath)}');
    
  } catch (e) {
    print('Error: $e');
  }
}

/// Example of caching multiple models and switching between them
Future<void> multipleModelExample() async {
  final modelCache = FllamaModelCache();
  
  final model1Path = '/path/to/first/model.gguf';
  final model2Path = '/path/to/second/model.gguf';
  
  try {
    // Load both models initially
    print('Loading model 1 and model 2');
    await modelCache.loadModel(model1Path);
    await modelCache.loadModel(model2Path);
    
    print('Models loaded: ${modelCache.loadedModels}');
    
    // Run inference with model 1
    print('Running inference with model 1');
    final request1 = FllamaInferenceRequest(
      contextSize: 4096,
      input: 'What is your favorite color?',
      maxTokens: 50,
      modelPath: model1Path,
      numGpuLayers: 0,
      penaltyFrequency: 0,
      penaltyRepeat: 1.1,
      temperature: 0.7,
      topP: 0.9,
      numThreads: 2,
    );
    
    await fllamaInference(request1, (response, openaiResponseJsonString, done) {
      if (done) print('Model 1 response: $response');
    });
    
    // Run inference with model 2
    print('Running inference with model 2');
    final request2 = FllamaInferenceRequest(
      contextSize: 4096,
      input: 'What is your favorite color?',
      maxTokens: 50,
      modelPath: model2Path,
      numGpuLayers: 0,
      penaltyFrequency: 0,
      penaltyRepeat: 1.1,
      temperature: 0.7,
      topP: 0.9,
      numThreads: 2,
    );
    
    await fllamaInference(request2, (response, openaiResponseJsonString, done) {
      if (done) print('Model 2 response: $response');
    });
    
    // Unload all models when done
    print('Unloading all models');
    await modelCache.unloadAllModels();
    print('Models loaded: ${modelCache.loadedModels}');
    
  } catch (e) {
    print('Error: $e');
  }
}
