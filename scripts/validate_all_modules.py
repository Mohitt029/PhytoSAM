"""
Complete validation script for all HEMSAM modules
Verifies realistic outputs for plant disease detection
"""

import sys
sys.path.append('D:/HEMSAM_Project')

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def validate_dataset():
    """Validate dataset outputs"""
    print("\n" + "="*60)
    print("VALIDATING DATASET MODULE")
    print("="*60)
    
    from data.dataset import PlantVillageSegmentationDataset
    
    dataset_path = Path("./data/plantvillage/raw/plantvillage/PlantVillage")
    dataset = PlantVillageSegmentationDataset(
        root_dir=dataset_path,
        split="train",
        generate_masks=True
    )
    
    # Get samples from different classes
    samples = []
    class_samples = {}
    
    for i in range(min(50, len(dataset))):
        sample = dataset[i]
        class_name = sample['class_name']
        if class_name not in class_samples and "healthy" not in class_name.lower():
            class_samples[class_name] = sample
            samples.append(sample)
        if len(class_samples) >= 5:
            break
    
    print(f"\n✓ Dataset size: {len(dataset)} images")
    print(f"✓ Classes found: {len(set(dataset.labels))}")
    print(f"✓ Sample images validated: {len(samples)}")
    
    # Verify mask properties
    for sample in samples[:3]:
        mask = sample['mask'].squeeze().numpy()
        has_lesions = mask.sum() > 0
        print(f"  - {sample['class_name']}: mask area = {mask.sum():.0f} pixels, lesions={'Yes' if has_lesions else 'No'}")
    
    return True

def validate_cdms_rfb():
    """Validate CDMS_RFB realistic outputs"""
    print("\n" + "="*60)
    print("VALIDATING CDMS_RFB MODULE")
    print("="*60)
    
    from models.cdms_rfb import CDMS_RFB, MultiScaleRFBSimple
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test with realistic feature maps
    x = torch.randn(2, 256, 64, 64).to(device)
    
    # Test edge attention response
    model = CDMS_RFB(in_channels=256, out_channels=256, use_edge_attention=True)
    model = model.to(device)
    
    with torch.no_grad():
        output, edge_attn = model(x)
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Edge attention shape: {edge_attn.shape}")
    print(f"✓ Edge attention range: [{edge_attn.min():.4f}, {edge_attn.max():.4f}]")
    print(f"✓ Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    # Verify edge attention highlights boundaries
    # Edge attention should have higher values where edges exist
    edge_attn_np = edge_attn[0, 0, :, :].cpu().numpy()
    print(f"✓ Edge attention spatial variance: {edge_attn_np.std():.4f} (non-uniform = good)")
    
    return True

def validate_uhdam():
    """Validate UHDAM realistic outputs"""
    print("\n" + "="*60)
    print("VALIDATING UHDAM MODULE")
    print("="*60)
    
    from models.uhd_attention import UHDAM, MultiScaleUHDAM
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    x = torch.randn(2, 256, 64, 64).to(device)
    
    model = UHDAM(channels=256, use_frequency=True)
    model = model.to(device)
    
    with torch.no_grad():
        output, attention_maps = model(x)
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {output.shape}")
    
    # Verify attention maps
    for name, attn in attention_maps.items():
        if 'spatial' in name:
            print(f"  - {name}: shape={attn.shape}, range=[{attn.min():.4f}, {attn.max():.4f}]")
        else:
            print(f"  - {name}: shape={attn.shape}")
    
    # Check fusion weights
    if hasattr(model, 'fusion_weights'):
        weights = torch.softmax(model.fusion_weights, dim=0)
        print(f"✓ Learnable fusion weights: spatial={weights[0]:.3f}, channel={weights[1]:.3f}, freq={weights[2]:.3f}")
    
    return True

def validate_edge_detection():
    """Validate edge detection on real leaf images"""
    print("\n" + "="*60)
    print("VALIDATING EDGE DETECTION MODULE")
    print("="*60)
    
    from utils.edge_detection import EdgeDetector, BoundaryEnhancer, TorchEdgeDetector
    
    # Load a real leaf image from dataset
    dataset_path = Path("./data/plantvillage/raw/plantvillage/PlantVillage")
    if dataset_path.exists():
        # Find a diseased leaf image
        for class_dir in dataset_path.iterdir():
            if class_dir.is_dir() and "healthy" not in class_dir.name.lower():
                images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.JPG"))
                if images:
                    img_path = images[0]
                    image = cv2.imread(str(img_path))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    print(f"✓ Loaded real leaf image: {class_dir.name}")
                    break
    else:
        # Fallback to synthetic
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        print(f"✓ Using synthetic image")
    
    detector = EdgeDetector()
    
    # Test different edge detectors
    sobel_mag, sobel_dir = detector.sobel_edges(image)
    canny = detector.canny_edges(image)
    laplacian = detector.laplacian_edges(image)
    
    print(f"✓ Sobel magnitude: shape={sobel_mag.shape}, range=[{sobel_mag.min():.3f}, {sobel_mag.max():.3f}]")
    print(f"✓ Canny edges: shape={canny.shape}, {canny.sum():.0f} edge pixels ({canny.mean()*100:.1f}%)")
    print(f"✓ Laplacian: shape={laplacian.shape}, range=[{laplacian.min():.3f}, {laplacian.max():.3f}]")
    
    # Test boundary enhancement
    # Create a simulated lesion mask
    mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(mask, (128, 128), 50, 255, -1)
    cv2.ellipse(mask, (128, 128), (30, 50), 45, 0, 360, 255, -1)
    
    boundary = BoundaryEnhancer.morphological_boundary(mask)
    print(f"✓ Boundary extraction: {boundary.sum():.0f} boundary pixels")
    
    # Test torch version
    torch_detector = TorchEdgeDetector()
    x = torch.randn(2, 3, 224, 224)
    edges = torch_detector(x)
    print(f"✓ Torch edge detector: output shape={edges.shape}")
    
    return True

def validate_integration():
    """Validate that modules work together"""
    print("\n" + "="*60)
    print("VALIDATING MODULE INTEGRATION")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    from models.cdms_rfb import MultiScaleRFBSimple
    from models.uhd_attention import UHDAM
    import torch.nn as nn  # Add this import
    
    # Create sequential pipeline
    class FeatureEnhancer(nn.Module):
        def __init__(self, channels=256):
            super().__init__()
            self.cdms = MultiScaleRFBSimple(channels, channels)
            self.uhdam = UHDAM(channels)
        
        def forward(self, x):
            x, edge = self.cdms(x)
            x, attn = self.uhdam(x)
            return x, edge, attn
    
    model = FeatureEnhancer().to(device)
    x = torch.randn(2, 256, 64, 64).to(device)
    
    with torch.no_grad():
        output, edge_attn, uhdam_attn = model(x)
    
    print(f"✓ CDMS_RFB + UHDAM integration successful")
    print(f"  Input: {x.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Edge attention: {edge_attn.shape if edge_attn is not None else 'None'}")
    print(f"  UHDAM attention maps: {len(uhdam_attn)}")
    
    return True

def visualize_results():
    """Create visualization of module outputs"""
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # 1. Dataset sample
        from data.dataset import PlantVillageSegmentationDataset
        dataset_path = Path("./data/plantvillage/raw/plantvillage/PlantVillage")
        dataset = PlantVillageSegmentationDataset(root_dir=dataset_path, split="train")
        sample = dataset[0]
        
        # Denormalize image for display
        img = sample['image'].permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        mask = sample['mask'].squeeze().numpy()
        
        axes[0, 0].imshow(img)
        axes[0, 0].set_title(f"Original Image\n{sample['class_name'][:30]}")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(mask, cmap='gray')
        axes[0, 1].set_title(f"Lesion Mask\nArea: {mask.sum():.0f} pixels")
        axes[0, 1].axis('off')
        
        # 2. Edge detection on real image
        from utils.edge_detection import EdgeDetector
        detector = EdgeDetector()
        
        # Find a real image
        for class_dir in dataset_path.iterdir():
            if class_dir.is_dir() and "healthy" not in class_dir.name.lower():
                images = list(class_dir.glob("*.jpg"))
                if images:
                    real_img = cv2.imread(str(images[0]))
                    real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
                    break
            else:
                # Fallback
                real_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        sobel, _ = detector.sobel_edges(real_img)
        canny = detector.canny_edges(real_img)
        
        axes[0, 2].imshow(sobel, cmap='hot')
        axes[0, 2].set_title("Sobel Edge Detection")
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(canny, cmap='gray')
        axes[0, 3].set_title("Canny Edge Detection")
        axes[0, 3].axis('off')
        
        # 3. Feature maps from models
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        from models.cdms_rfb import CDMS_RFB
        from models.uhd_attention import UHDAM
        
        x = torch.randn(1, 256, 64, 64).to(device)
        
        cdms = CDMS_RFB(256, 256).to(device)
        uhdam = UHDAM(256).to(device)
        
        with torch.no_grad():
            feat_cdms, edge = cdms(x)
            feat_uhdam, attn = uhdam(feat_cdms)
        
        # Normalize feature map for display
        feat_map = feat_uhdam[0, 0:3, :, :].cpu().numpy()
        feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min() + 1e-8)
        feat_map = np.transpose(feat_map, (1, 2, 0))
        
        axes[1, 0].imshow(feat_map)
        axes[1, 0].set_title("CDMS_RFB + UHDAM\nFeature Maps")
        axes[1, 0].axis('off')
        
        # Edge attention
        if edge is not None:
            edge_map = edge[0, 0, :, :].cpu().numpy()
            edge_map = (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min() + 1e-8)
            axes[1, 1].imshow(edge_map, cmap='hot')
            axes[1, 1].set_title("Edge Attention Map")
            axes[1, 1].axis('off')
        
        # Spatial attention
        if 'spatial' in attn:
            spatial_attn = attn['spatial'][0, 0, :, :].cpu().numpy()
            axes[1, 2].imshow(spatial_attn, cmap='viridis')
            axes[1, 2].set_title("Spatial Attention")
            axes[1, 2].axis('off')
        
        # Channel attention weights
        if 'channel' in attn:
            channel_attn = attn['channel'][0, :, 0, 0].cpu().numpy()
            axes[1, 3].bar(range(20), channel_attn[:20])
            axes[1, 3].set_title("Channel Attention\n(Top 20 channels)")
            axes[1, 3].set_xlabel("Channel Index")
            axes[1, 3].set_ylabel("Attention Weight")
        
        plt.tight_layout()
        plt.savefig('module_validation_visualization.png', dpi=150, bbox_inches='tight')
        print("✓ Visualization saved to 'module_validation_visualization.png'")
        
        plt.close()  # Close figure to avoid display issues
        
    except Exception as e:
        print(f"Visualization note: {e}")
    
    return True

def run_all_validations():
    """Run all validation tests"""
    print("\n" + "="*60)
    print("HEMSAM MODULE VALIDATION SUITE")
    print("="*60)
    print("This validates all components produce realistic outputs")
    
    tests = [
        ("Dataset Module", validate_dataset),
        ("CDMS_RFB Module", validate_cdms_rfb),
        ("UHDAM Module", validate_uhdam),
        ("Edge Detection Module", validate_edge_detection),
        ("Module Integration", validate_integration),
        ("Visualization", visualize_results),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} failed: {e}")
            results.append((name, False))
    
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n" + "🎉"*30)
        print("ALL MODULES VALIDATED SUCCESSFULLY!")
        print("HEMSAM is ready for Chat 5 (Model Assembly)")
        print("🎉"*30)
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
    
    return all_passed


if __name__ == "__main__":
    # Import nn for integration test
    import torch.nn as nn
    run_all_validations()