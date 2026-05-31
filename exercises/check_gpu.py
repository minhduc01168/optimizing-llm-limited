"""
GPU Validation Script - Kiểm tra GPU trước khi chạy exercises
"""

import sys


def check_gpu():
    """Kiểm tra GPU có sẵn sàng không"""
    print("="*50)
    print("GPU VALIDATION")
    print("="*50)
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("❌ CUDA không khả dụng!")
            print("   Hãy kiểm tra:")
            print("   1. NVIDIA driver đã cài chưa?")
            print("   2. CUDA toolkit đã cài chưa?")
            print("   3. PyTorch có CUDA support không?")
            print(f"   PyTorch version: {torch.__version__}")
            print(f"   CUDA available: {torch.cuda.is_available()}")
            return False
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"✅ GPU found: {gpu_name}")
        print(f"   Memory: {gpu_memory:.1f} GB")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   PyTorch version: {torch.__version__}")
        
        # Kiểm tra T4
        if "T4" in gpu_name:
            print(f"   ✅ NVIDIA T4 detected - Perfect!")
        elif gpu_memory >= 15:
            print(f"   ✅ GPU đủ mạnh cho exercises")
        else:
            print(f"   ⚠️ GPU có thể không đủ VRAM cho một số exercises")
        
        return True
        
    except ImportError:
        print("❌ PyTorch chưa cài đặt!")
        print("   Chạy: pip install torch")
        return False
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return False


def check_dependencies():
    """Kiểm tra các dependencies cần thiết"""
    print("\n" + "="*50)
    print("DEPENDENCY CHECK")
    print("="*50)
    
    required = [
        ("torch", "PyTorch"),
        ("transformers", "HuggingFace Transformers"),
        ("pynvml", "NVIDIA Management Library"),
    ]
    
    optional = [
        ("bitsandbytes", "Quantization"),
        ("vllm", "vLLM Serving"),
        ("fastapi", "FastAPI"),
        ("redis", "Redis"),
    ]
    
    all_ok = True
    
    print("\nRequired packages:")
    for module, name in required:
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} - MISSING")
            all_ok = False
    
    print("\nOptional packages:")
    for module, name in optional:
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ⚠️ {name} - Not installed (needed for some exercises)")
    
    return all_ok


def main():
    print("LLM Engineering Lab - Environment Check")
    print("="*50)
    
    gpu_ok = check_gpu()
    deps_ok = check_dependencies()
    
    print("\n" + "="*50)
    print("RESULT")
    print("="*50)
    
    if gpu_ok and deps_ok:
        print("✅ Environment ready! Bạn có thể bắt đầu exercises.")
        return 0
    elif gpu_ok:
        print("⚠️ GPU OK nhưng thiếu một số dependencies.")
        print("   Chạy: pip install -r requirements.txt")
        return 0
    else:
        print("❌ Environment chưa sẵn sàng.")
        print("   Hãy cài đặt GPU driver và CUDA toolkit.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
