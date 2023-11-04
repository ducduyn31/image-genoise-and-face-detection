from typing import Any, Dict, List, Optional, Union

import PIL.Image
import torch
from torchvision.transforms.v2 import Transform
from torchvision import tv_tensors


class Overlay(Transform):
    """ Overlay a layer on top of an image.

    Args:
        color (tuple, optional): The layer to overlay on top of the image.
        alpha (float, optional): The alpha value of the overlay.
    """

    def __init__(self, color: Optional[Union[tuple, int]] = None, alpha: Optional[float] = None):
        super().__init__()
        self.color = color
        self.alpha = alpha

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        if self.color is None:
            self.color = torch.Tensor((128, 128, 128))
        elif isinstance(self.color, int):
            self.color = torch.Tensor((self.color, self.color, self.color))

        if self.alpha is None:
            self.alpha = 0.2

        return dict(
            color=self.color,
            alpha=self.alpha,
        )

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if len(params) < 1:
            return inpt

        if isinstance(inpt, tv_tensors.BoundingBoxes):
            return inpt

        overlay_color = params["color"]
        alpha = params["alpha"]

        if isinstance(inpt, PIL.Image.Image):
            inpt = tv_tensors.Image(inpt)

        overlay = overlay_color.unsqueeze(1).unsqueeze(2).expand_as(inpt)
        outpt = inpt * (1 - alpha) + overlay * alpha
        outpt = outpt.to(torch.uint8)

        return outpt
