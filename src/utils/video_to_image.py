from typing import Any, Callable, Dict, Optional

from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.folder import find_classes,make_dataset
from torchvision.transforms import v2
import torchvision.transforms.functional as f
import os

def save_images(root, output_path,size=(64,64),step_between_clips=100):
    extensions = ("avi","mp4",)
    frames_per_clip: int = 1
    step_between_clips: int = step_between_clips
    frame_rate: Optional[int] = None
    fold: int = 1
    train: bool = True
    transform: Optional[Callable] = None
    _precomputed_metadata: Optional[Dict[str, Any]] = None
    num_workers: int = 8
    _video_width: int = 0
    _video_height: int = 0
    _video_min_dimension: int = 0
    _audio_samples: int = 0
    output_format: str = "TCHW"

    classes, class_to_idx = find_classes(root)
    samples = make_dataset(root, class_to_idx, extensions=extensions, is_valid_file=None)
    video_list = [x[0] for x in samples]
    video_clips = VideoClips(
        video_list,
        frames_per_clip,
        step_between_clips,
        frame_rate,
        _precomputed_metadata,
        num_workers=num_workers,
        _video_width=_video_width,
        _video_height=_video_height,
        _video_min_dimension=_video_min_dimension,
        _audio_samples=_audio_samples,
        output_format=output_format,
    )
    transform_normal = v2.Compose([
        v2.Resize(size=(64,64)),
        ])
    for i in range(video_clips.num_clips()):
        video, _, _, _ = video_clips.get_clip(i)
        video = transform_normal(video).squeeze()
        image = f.to_pil_image(video)
        image_name = f"image{i}.png"
        print("Saving",os.path.join(output_path, image_name))
        image.save(os.path.join(output_path, image_name))


if __name__ == "__main__":
    output_path = "/HOME/simclr_pytorch/datasets/ImAnimalia2/"
    target_size = (64, 64)
    dataset_path = "/HOME/simclr_pytorch/datasets/animalia/"
    save_images(dataset_path,output_path,target_size,step_between_clips=70)

