from typing import List, Any, Dict, Optional

import PIL
import torch
from torchvision import tv_tensors
from torchvision.transforms.v2 import Transform
from torchvision.transforms.v2._utils import query_size
from PIL.Image import Image


class Glare(Transform):
    """Simulate light glare on an image.

    Args:
        glare_center (tuple, optional): Center of the glare.
        glare_radius (int, optional): Radius of the glare.
        glare_intensity (float, optional): Intensity of the glare.

    """

    def __init__(self, glare_center: Optional[tuple] = None, glare_radius: Optional[int] = None,
                 glare_intensity: Optional[float] = None):
        super().__init__()

        self.glare_center = glare_center
        self.glare_radius = glare_radius
        self.glare_intensity = glare_intensity

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        orig_h, orig_w = query_size(flat_inputs)

        rand_w, rand_h = torch.rand(2)

        if self.glare_center is None:
            self.glare_center = (orig_h * rand_h // 1, orig_w * rand_w // 1)

        if self.glare_radius is None:
            min_rad = min(orig_w, orig_h) // 10
            max_rad = min_rad * 4
            self.glare_radius = torch.randint(min_rad, max_rad, (1,))[0]

        if self.glare_intensity is None:
            self.glare_intensity = torch.randint(50, 255, (1,))

        return dict(
            glare_center=self.glare_center,
            glare_radius=self.glare_radius,
            glare_intensity=self.glare_intensity,
            image_size=(orig_h, orig_w),
        )

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if len(params) < 1:
            return inpt

        if isinstance(inpt, tv_tensors.BoundingBoxes):
            return inpt

        H, W = params["image_size"]
        center_y, center_x = params["glare_center"]
        radius = params["glare_radius"]
        mask_y, mask_x = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W))
        dist = torch.sqrt(torch.pow(mask_x - center_x / W, 2) + torch.pow(mask_y - center_y / H, 2))
        mask = torch.exp(torch.neg(torch.pow(dist, 2)) / (2 * torch.pow(radius / W, 2)))

        intensity = params["glare_intensity"]
        if isinstance(inpt, PIL.Image.Image):
            inpt = tv_tensors.Image(inpt)

        the_glare = (mask * intensity).to(torch.uint8)

        outpt = inpt * (torch.ones_like(mask) - mask)
        outpt = outpt.to(torch.uint8)
        outpt = outpt + the_glare

        return outpt
