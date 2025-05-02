#include "fllama.h"
#include "fllama_inference_queue.h"

// Platform-specific includes
#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

#if TARGET_OS_IOS
#include "../ios/llama.cpp/include/llama.h"
#elif TARGET_OS_OSX
#include "../macos/llama.cpp/include/llama.h"
#else
#include "llama.cpp/include/llama.h"
#endif

#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <atomic>

// Global model cache
struct ModelInstance {
    llama_model* model;
    llama_context* ctx;
    int num_gpu_layers;
    int num_threads;
    std::atomic<int> ref_count;
    std::chrono::time_point<std::chrono::steady_clock> last_used;
    
    ModelInstance(llama_model* m, llama_context* c, int gpu_layers, int threads) 
        : model(m), ctx(c), num_gpu_layers(gpu_layers), num_threads(threads), ref_count(1),
          last_used(std::chrono::steady_clock::now()) {}
};

static std::mutex g_models_mutex;
static std::unordered_map<std::string, ModelInstance*> g_loaded_models;

// Helper for log messages
static void log_model_message(const std::string& message) {
    std::cerr << "[fllama_model_manager] " << message << std::endl;
}

// Access function for inference queue to use loaded models
ModelInstance* get_cached_model_instance(const std::string& model_path) {
    std::lock_guard<std::mutex> lock(g_models_mutex);
    auto it = g_loaded_models.find(model_path);
    if (it != g_loaded_models.end()) {
        it->second->ref_count++;
        it->second->last_used = std::chrono::steady_clock::now();
        return it->second;
    }
    return nullptr;
}

void release_cached_model_instance(const std::string& model_path) {
    std::lock_guard<std::mutex> lock(g_models_mutex);
    auto it = g_loaded_models.find(model_path);
    if (it != g_loaded_models.end()) {
        it->second->ref_count--;
        it->second->last_used = std::chrono::steady_clock::now();
    }
}

extern "C" {

FFI_PLUGIN_EXPORT fllama_model_handle fllama_model_load(const char* model_path, int num_gpu_layers, int num_threads) {
    if (!model_path) {
        log_model_message("Error: model_path is null");
        return nullptr;
    }
    
    std::string path_str(model_path);
    
    // Lock to prevent race conditions
    std::lock_guard<std::mutex> lock(g_models_mutex);
    
    // Check if model is already loaded
    auto it = g_loaded_models.find(path_str);
    if (it != g_loaded_models.end()) {
        // Model already loaded, increase reference count
        it->second->ref_count++;
        log_model_message("Model already loaded, increased reference count: " + path_str);
        return static_cast<fllama_model_handle>(it->second);
    }
    
    // Model not loaded yet, create new instance
    log_model_message("Loading model: " + path_str);
    
    // Initialize llama parameters
    llama_model_params model_params = llama_model_default_params();
    
    // Set GPU layers parameter
    model_params.n_gpu_layers = num_gpu_layers;
    
    // Load the model
    llama_model* model = llama_load_model_from_file(model_path, model_params);
    if (!model) {
        log_model_message("Failed to load model: " + path_str);
        return nullptr;
    }
    
    // Create context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_threads = num_threads;
    
    llama_context* ctx = llama_new_context_with_model(model, ctx_params);
    if (!ctx) {
        llama_free_model(model);
        log_model_message("Failed to create context for model: " + path_str);
        return nullptr;
    }
    
    log_model_message("Successfully loaded model with " + std::to_string(num_gpu_layers) + 
                     " GPU layers and " + std::to_string(num_threads) + " threads: " + path_str);
    
    // Store the model instance
    ModelInstance* instance = new ModelInstance(model, ctx, num_gpu_layers, num_threads);
    g_loaded_models[path_str] = instance;
    
    log_model_message("Successfully loaded model: " + path_str);
    return static_cast<fllama_model_handle>(instance);
}

FFI_PLUGIN_EXPORT void fllama_model_unload(fllama_model_handle model_handle) {
    if (!model_handle) {
        log_model_message("Error: Attempted to unload null model handle");
        return;
    }
    
    std::lock_guard<std::mutex> lock(g_models_mutex);
    
    // Find the model in our map
    ModelInstance* instance = static_cast<ModelInstance*>(model_handle);
    
    // Find the model path in our map
    std::string model_path;
    for (const auto& pair : g_loaded_models) {
        if (pair.second == instance) {
            model_path = pair.first;
            break;
        }
    }
    
    if (model_path.empty()) {
        log_model_message("Error: Model handle not found in loaded models");
        return;
    }
    
    // Decrease reference count
    instance->ref_count--;
    
    if (instance->ref_count <= 0) {
        // Free the model and context
        if (instance->ctx) {
            llama_free(instance->ctx);
        }
        if (instance->model) {
            llama_free_model(instance->model);
        }
        
        // Remove from map and delete instance
        g_loaded_models.erase(model_path);
        delete instance;
        
        log_model_message("Unloaded model: " + model_path);
    } else {
        log_model_message("Decreased reference count for model: " + model_path + 
                          " (remaining: " + std::to_string(instance->ref_count) + ")");
    }
}

FFI_PLUGIN_EXPORT int fllama_model_is_loaded(const char* model_path) {
    if (!model_path) {
        return 0;
    }
    
    std::string path_str(model_path);
    std::lock_guard<std::mutex> lock(g_models_mutex);
    
    return g_loaded_models.find(path_str) != g_loaded_models.end() ? 1 : 0;
}

FFI_PLUGIN_EXPORT void fllama_inference_with_model(struct fllama_inference_request request,
                                              fllama_model_handle model_handle,
                                              fllama_inference_callback callback) {
    if (!model_handle) {
        // Provide error feedback
        const char* error_msg = "Error: model handle is null";
        callback(error_msg, "{\"error\": \"model handle is null\"}", 1);
        return;
    }
    
    // Cast model handle back to our instance
    ModelInstance* instance = static_cast<ModelInstance*>(model_handle);
    
    // Update last used timestamp
    instance->last_used = std::chrono::steady_clock::now();
    
    // Increment reference count while in use
    instance->ref_count++;
    
    try {
        // Call the regular inference function - the model is already cached internally
        fllama_inference(request, callback);
    } catch (...) {
        // Decrement reference count if there was an error
        instance->ref_count--;
        throw; // Re-throw the exception
    }
    
    // Note: We don't decrement ref_count here because the inference is asynchronous
    // The callback function will handle proper cleanup when inference is complete
}

} // extern "C"
