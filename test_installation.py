"""
InfoMAE Installation Test Script
설치가 올바르게 되었는지 확인합니다
"""
import sys
import os

def test_basic_imports():
    """기본 패키지 import 테스트"""
    print("\n" + "="*60)
    print("Testing Basic Imports...")
    print("="*60)
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        
        import torchvision
        print(f"✓ TorchVision {torchvision.__version__}")
        
        import timm
        print(f"✓ timm {timm.__version__}")
        
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
        
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
        
        import seaborn
        print(f"✓ Seaborn {seaborn.__version__}")
        
        import PIL
        print(f"✓ Pillow {PIL.__version__}")
        
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
        
        import einops
        print(f"✓ einops")
        
        from tqdm import tqdm
        print(f"✓ tqdm")
        
        print("\n✅ All basic imports successful!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import failed: {e}")
        return False


def test_pytorch_cuda():
    """PyTorch CUDA 지원 테스트"""
    print("\n" + "="*60)
    print("Testing PyTorch & CUDA...")
    print("="*60)
    
    try:
        import torch
        
        print(f"PyTorch version: {torch.__version__}")
        print(f"Python version: {sys.version.split()[0]}")
        
        # CUDA 확인
        cuda_available = torch.cuda.is_available()
        print(f"\nCUDA available: {cuda_available}")
        
        if cuda_available:
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                print(f"\nGPU {i}:")
                print(f"  Name: {torch.cuda.get_device_name(i)}")
                print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
            
            # GPU 테스트
            print("\nTesting GPU computation...")
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.mm(x, y)
            print("✓ GPU computation successful")
        else:
            print("\n⚠️  No GPU detected. Training will run on CPU.")
            print("For GPU support:")
            print("  1. Install CUDA from NVIDIA")
            print("  2. Reinstall PyTorch with CUDA:")
            print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        
        print("\n✅ PyTorch test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ PyTorch test failed: {e}")
        return False


def test_infomae_modules():
    """InfoMAE 모듈 import 테스트"""
    print("\n" + "="*60)
    print("Testing InfoMAE Modules...")
    print("="*60)
    
    try:
        # Config
        from config import get_config, Config
        print("✓ config module")
        
        config = get_config('stage1')
        print(f"✓ Config loaded (stage1)")
        print(f"  - Use surprisal attention: {config.model.use_surprisal_attention}")
        print(f"  - Adaptive masking: {config.model.adaptive_masking}")
        
        # Models
        from models.infomae import InfoMAE, SurprisalWeightedAttention
        print("✓ InfoMAE model")
        
        # Data
        from data.datasets import build_dataset, build_dataloader
        print("✓ data.datasets module")
        
        # Engine
        from engine import Trainer, LinearProbe
        print("✓ engine module")
        
        # Utils
        from utils.losses import InfoMAELoss, compute_mutual_information
        print("✓ utils.losses module")
        
        from utils.metrics import compute_reconstruction_metrics, compute_attention_selectivity
        print("✓ utils.metrics module")
        
        from utils.visualization import visualize_reconstruction, visualize_attention_maps
        print("✓ utils.visualization module")
        
        print("\n✅ All InfoMAE modules imported successfully!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Module import failed: {e}")
        print("\nMake sure you're in the correct directory:")
        print(f"Current directory: {os.getcwd()}")
        return False


def test_model_instantiation():
    """모델 생성 테스트"""
    print("\n" + "="*60)
    print("Testing Model Instantiation...")
    print("="*60)
    
    try:
        import torch
        from models.infomae import InfoMAE
        from config import get_config
        
        config = get_config('stage2')
        
        print("Creating InfoMAE model...")
        model = InfoMAE(
            img_size=config.model.img_size,
            patch_size=config.model.patch_size,
            embed_dim=config.model.embed_dim,
            depth=config.model.depth,
            num_heads=config.model.num_heads,
            use_surprisal_attention=config.model.use_surprisal_attention,
            adaptive_masking=config.model.adaptive_masking,
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ Model created successfully")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        print(f"  - Model size: {total_params * 4 / 1e6:.2f} MB (FP32)")
        
        # Forward pass 테스트
        print("\nTesting forward pass...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        dummy_input = torch.randn(2, 3, 224, 224).to(device)
        
        with torch.no_grad():
            loss, pred, mask, surprisal, latent = model(
                dummy_input,
                mask_ratio=0.75,
                lambda_weight=1.0,
                alpha=3.0,
                gamma=1.0,
            )
        
        print(f"✓ Forward pass successful")
        print(f"  - Loss: {loss.item():.4f}")
        print(f"  - Pred shape: {pred.shape}")
        print(f"  - Surprisal shape: {surprisal.shape}")
        print(f"  - Latent shape: {latent.shape}")
        
        print("\n✅ Model instantiation test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loading():
    """데이터 로딩 테스트 (CIFAR-10으로 빠른 테스트)"""
    print("\n" + "="*60)
    print("Testing Data Loading (CIFAR-10)...")
    print("="*60)
    
    try:
        import torch
        from torchvision import datasets, transforms
        
        print("Downloading CIFAR-10 (for testing)...")
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ])
        
        dataset = datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )
        
        print(f"✓ Dataset loaded: {len(dataset)} samples")
        
        # DataLoader 테스트
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
        
        images, labels = next(iter(loader))
        print(f"✓ DataLoader working")
        print(f"  - Batch shape: {images.shape}")
        print(f"  - Labels shape: {labels.shape}")
        
        print("\n✅ Data loading test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Data loading test failed: {e}")
        return False


def test_directories():
    """필요한 디렉토리 확인"""
    print("\n" + "="*60)
    print("Checking Directory Structure...")
    print("="*60)
    
    required_dirs = [
        'models',
        'data',
        'utils',
        'scripts',
    ]
    
    optional_dirs = [
        'outputs',
        'pretrained',
        'logs',
    ]
    
    all_good = True
    
    print("\nRequired directories:")
    for dir_name in required_dirs:
        if os.path.isdir(dir_name):
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ❌ {dir_name}/ (MISSING)")
            all_good = False
    
    print("\nOptional directories:")
    for dir_name in optional_dirs:
        if os.path.isdir(dir_name):
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ⚠️  {dir_name}/ (will be created automatically)")
    
    print("\nRequired files:")
    required_files = [
        'main.py',
        'config.py',
        'engine.py',
        'requirements.txt',
        'README.md',
    ]
    
    for file_name in required_files:
        if os.path.isfile(file_name):
            print(f"  ✓ {file_name}")
        else:
            print(f"  ❌ {file_name} (MISSING)")
            all_good = False
    
    if all_good:
        print("\n✅ Directory structure is correct!")
    else:
        print("\n⚠️  Some required files/directories are missing")
    
    return all_good


def main():
    """모든 테스트 실행"""
    print("\n" + "="*60)
    print("InfoMAE Installation Verification")
    print("="*60)
    print(f"Python: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print("="*60)
    
    tests = [
        ("Directory Structure", test_directories),
        ("Basic Imports", test_basic_imports),
        ("PyTorch & CUDA", test_pytorch_cuda),
        ("InfoMAE Modules", test_infomae_modules),
        ("Model Instantiation", test_model_instantiation),
        ("Data Loading", test_data_loading),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except KeyboardInterrupt:
            print("\n\nTest interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Unexpected error in {test_name}: {e}")
            results.append((test_name, False))
    
    # 최종 결과
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print("="*60)
    print(f"Results: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("\n🎉 All tests passed! InfoMAE is ready to use!")
        print("\nNext steps:")
        print("  1. Quick test: bash scripts/quick_start.sh")
        print("  2. Download pretrained: bash scripts/download_pretrained.sh")
        print("  3. Run experiments: bash run_experiments.sh")
        print("\nFor more info, see:")
        print("  - README.md")
        print("  - USAGE.md")
        print("  - IMPLEMENTATION_CHECK.md")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        print("\nPlease check the error messages above and:")
        print("  1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("  2. Verify you're in the correct directory")
        print("  3. Check Python version >= 3.8")
        return 1


if __name__ == '__main__':
    sys.exit(main())

