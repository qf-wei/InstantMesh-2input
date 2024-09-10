import torch

state_dict = torch.load("/media/hdd2/qf_wei/InstantMesh-2input/logs/zero123plus-finetune/checkpoints/step=00061000.ckpt", map_location="cpu")

keys_to_remove = {
    "betas",
    "alphas_cumprod",
    "alphas_cumprod_prev",
    "sqrt_alphas_cumprod",
    "sqrt_one_minus_alphas_cumprod",
    "sqrt_recip_alphas_cumprod",
    "sqrt_recipm1_alphas_cumprod",
}

updated_state_dict = {
    key.replace("unet.unet.", ""): value
    for key, value in state_dict["state_dict"].items()
    if not any(k in key for k in keys_to_remove)
}
#save the updated state_dict
torch.save(updated_state_dict, "/media/hdd2/qf_wei/InstantMesh-2input/logs/zero123plus-finetune/checkpoints/recon-1-2-61000-v1.ckpt")
