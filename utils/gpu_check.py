#!/usr/bin/env python3
"""
Script para verificar detección de GPU NVIDIA
"""


def check_gpu_status():
    """Verifica el estado completo de la GPU"""

    print("🔍 VERIFICANDO ESTADO DE GPU")
    print("=" * 50)

    # 1. Verificar NVIDIA drivers
    print("\n1️⃣ NVIDIA Drivers:")
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ nvidia-smi funciona")
            # Extraer info básica
            lines = result.stdout.split('\n')
            for line in lines:
                if 'RTX' in line or 'GTX' in line:
                    print(f"   🎮 GPU encontrada: {line.strip()}")
                elif 'CUDA Version' in line:
                    cuda_version = line.split('CUDA Version: ')[1].split()[0]
                    print(f"   🔧 CUDA Driver Version: {cuda_version}")
        else:
            print("❌ nvidia-smi no funciona")
            print("   Instala los drivers NVIDIA")
    except FileNotFoundError:
        print("❌ nvidia-smi no encontrado")
        print("   Instala los drivers NVIDIA")

    # 2. Verificar PyTorch
    print("\n2️⃣ PyTorch:")
    try:
        import torch
        print(f"✅ PyTorch instalado: {torch.__version__}")
        print(f"   🔧 CUDA disponible: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"   🎮 Dispositivos CUDA: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"   📊 GPU {i}: {props.name}")
                print(f"   💾 VRAM: {props.total_memory / 1e9:.1f} GB")
                print(f"   🔢 CUDA Capability: {props.major}.{props.minor}")
        else:
            print("   ❌ CUDA no disponible en PyTorch")
            print(f"   🔧 PyTorch compilado con CUDA: {torch.version.cuda}")
            if torch.version.cuda is None:
                print("   ⚠️ PyTorch instalado SIN soporte CUDA")

    except ImportError:
        print("❌ PyTorch no instalado")

    # 3. Verificar sentence-transformers
    print("\n3️⃣ Sentence Transformers:")
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ sentence-transformers instalado")

        # Probar cargar modelo pequeño en GPU
        try:
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda')
            print("✅ Modelo cargado en GPU exitosamente")

            # Test rápido
            test_text = ["Hello world"]
            embedding = model.encode(test_text)
            print(f"✅ Test embedding: {embedding.shape}")

        except Exception as e:
            print(f"❌ Error cargando modelo en GPU: {e}")

    except ImportError:
        print("❌ sentence-transformers no instalado")

    # 4. Recomendaciones
    print("\n💡 RECOMENDACIONES:")
    print("=" * 50)

    try:
        import torch
        if not torch.cuda.is_available():
            print("🔧 Reinstalar PyTorch con CUDA:")
            print("   pip uninstall torch torchvision torchaudio")
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    except ImportError:
        print("🔧 Instalar PyTorch con CUDA:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")


def test_embedding_performance():
    """Test de rendimiento para comparar CPU vs GPU"""
    try:
        import torch
        from sentence_transformers import SentenceTransformer
        import time

        if not torch.cuda.is_available():
            print("❌ CUDA no disponible, saltando test de rendimiento")
            return

        print("\n🏃‍♂️ TEST DE RENDIMIENTO")
        print("=" * 50)

        # Datos de prueba
        test_texts = [f"This is test sentence number {i}" for i in range(100)]

        # Test CPU
        print("\n🐌 Probando en CPU...")
        model_cpu = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
        start_time = time.time()
        embeddings_cpu = model_cpu.encode(test_texts, batch_size=32)
        cpu_time = time.time() - start_time
        print(f"   ⏱️ Tiempo CPU: {cpu_time:.2f} segundos")

        # Test GPU
        print("\n🚀 Probando en GPU...")
        model_gpu = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda')
        start_time = time.time()
        embeddings_gpu = model_gpu.encode(test_texts, batch_size=64)
        gpu_time = time.time() - start_time
        print(f"   ⏱️ Tiempo GPU: {gpu_time:.2f} segundos")

        # Comparación
        speedup = cpu_time / gpu_time
        print(f"\n🏆 Aceleración GPU: {speedup:.1f}x más rápido")

        if speedup > 2:
            print("✅ GPU funcionando correctamente")
        else:
            print("⚠️ GPU no está dando la aceleración esperada")

    except Exception as e:
        print(f"❌ Error en test de rendimiento: {e}")


if __name__ == "__main__":
    check_gpu_status()
    test_embedding_performance()