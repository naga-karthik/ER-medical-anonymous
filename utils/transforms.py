
from monai.transforms import (Compose, CropForegroundd, LoadImaged, RandFlipd, 
            RandRotate90d, NormalizeIntensityd, HistogramNormalized, EnsureTyped, Invertd, 
            DivisiblePadd, RandWeightedCropd, Rand3DElasticd, RandSpatialCropSamplesd)

# transforms for the MS Brain dataset
def train_transforms(crop_size, num_samples_pv, lbl_key="label"):
    return Compose([   
            LoadImaged(keys=["image", lbl_key], ensure_channel_first=True),
            CropForegroundd(keys=["image", lbl_key], source_key="image"),     # crops >0 values with a bounding box
            RandWeightedCropd(keys=["image", lbl_key], w_key=lbl_key, spatial_size=crop_size, 
                num_samples=num_samples_pv),
            RandFlipd(keys=["image", lbl_key], spatial_axis=[0], prob=0.50,),
            RandFlipd(keys=["image", lbl_key], spatial_axis=[1], prob=0.50,),
            RandFlipd(keys=["image", lbl_key],spatial_axis=[2],prob=0.50,),
            RandRotate90d(keys=["image", lbl_key], prob=0.10, max_k=3,),
            HistogramNormalized(keys=["image"], mask=None),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True),
        ])

def val_transforms(lbl_key="label"):
    return Compose([
            LoadImaged(keys=["image", lbl_key], ensure_channel_first=True),
            CropForegroundd(keys=["image", lbl_key], source_key="image"),
            HistogramNormalized(keys=["image"], mask=None),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True),
        ])

def test_transforms(lbl_key="label"):
    return Compose([
            LoadImaged(keys=["image", lbl_key], ensure_channel_first=True),
            HistogramNormalized(keys=["image"], mask=None),  
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True),
        ])

# =================================================================================================

# transforms for the SCGM dataset
def train_transforms_scgm(crop_size, lbl_key="label", z_pad=32):
    return Compose([
            LoadImaged(keys=["image", lbl_key], ensure_channel_first=True),
            CropForegroundd(keys=["image", lbl_key], source_key="image"),     # crops >0 values with a bounding box
            RandSpatialCropSamplesd(keys=["image", lbl_key], num_samples=4, roi_size=crop_size, random_size=False, 
                                    random_center=False,),
            DivisiblePadd(keys=["image", lbl_key], k=z_pad),
            Rand3DElasticd(keys=["image", lbl_key], sigma_range=(3.5, 4.5), magnitude_range=(28.0, 30.0), prob=0.3),
            RandFlipd(keys=["image", lbl_key], spatial_axis=[0], prob=0.50,),
            RandFlipd(keys=["image", lbl_key], spatial_axis=[1], prob=0.50,),
            RandFlipd(keys=["image", lbl_key],spatial_axis=[2],prob=0.50,),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True),
            ])

def val_transforms_scgm(lbl_key="label"):
    return Compose([
            LoadImaged(keys=["image", lbl_key], ensure_channel_first=True),
            CropForegroundd(keys=["image", lbl_key], source_key="image"),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True),
        ])

def test_transforms_scgm(lbl_key="label"):
    return Compose([
            LoadImaged(keys=["image", lbl_key], ensure_channel_first=True),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True),
        ])

# =================================================================================================

def test_post_pred_transforms(pred_key="pred", lbl_key="label", test_transforms=None):
    """
    post-processing transforms for testing - taken from 
    https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/torch/unet_inference_dict.py#L66
    """
    return Compose([
                EnsureTyped(keys=[pred_key, lbl_key]),
                Invertd(keys=pred_key, transform=test_transforms, orig_keys="image",  meta_keys="pred_meta_dict",  
                        orig_meta_keys=["image_meta_dict"],  meta_key_postfix="meta_dict", nearest_interp=False, 
                        to_tensor=True),
            ])
