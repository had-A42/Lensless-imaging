import torch
from torch import nn

import lensless

def LenslessWrapper(nn.Module):
    def __init__(self, recon_method: str, recon_kwargs, freeze: bool = False) -> None:
        super().__init__()

        self.recon = getattr(lensless, recon_method)(**recon_kwargs)

        if freeze:
            for param in self.parameters():
                param.requires_grad = False   

    def forward(self, lensless: torch.tensor, psfs: torch.tensor | None=None) -> torch.tensor:
        if isinstance(self.recon, nn.Module):
            recon_lensed = self.recon(lensless, psfs=psfs)

        
        # if corners_list is not None:
        #     assert roi_kwargs is not None, "Define ROI kwargs to fix perspective"
        #     recon_lensed = fix_perspective(recon_lensed, corners_list, roi_kwargs)

        # if roi_kwargs is not None:
        #     roi_indexes = get_roi_indexes(
        #         n_dim=len(recon_lensed.shape), axis=(-3, -2), **roi_kwargs
        #     )

        #     recon_lensed = recon_lensed[roi_indexes]

        return recon_lensed