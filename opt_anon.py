import os
from diffusers import StableDiffusionPipeline
import torch
import numpy as np
import torch.utils.data
from einops import rearrange
from PIL import Image, ImageOps
from torchvision import transforms
from utils import compute_score, load_model_by_repo_id, pil_to_input

class PoisonGeneration(object):
    def __init__(self, device, eps=0.05, contrast=True, fr=False, HF_TOKEN=""):
        self.eps = eps
        # self.target_concept = target_concept
        self.device = device
        self.contrast = contrast
        self.full_sd_model = self.load_model()
        self.transform = self.resizer()
        if fr:
            fr_id = 'minchul/cvlface_adaface_vit_base_kprpe_webface4m'
            aligner_id = 'minchul/cvlface_DFA_mobilenet'
            fr_path = os.path.expanduser('~/.cvlface_cache/minchul/cvlface_adaface_vit_base_kprpe_webface4m')
            aligner_path = os.path.expanduser('~/.cvlface_cache/minchul/cvlface_DFA_mobilenet')
            
            self.fr_model = load_model_by_repo_id(repo_id=fr_id,
                                            save_path=fr_path,
                                            HF_TOKEN=HF_TOKEN).to(device)
            self.aligner = load_model_by_repo_id(repo_id=aligner_id,
                                            save_path=aligner_path,
                                            HF_TOKEN=HF_TOKEN).to(device)

    def resizer(self):
        image_transforms = transforms.Compose(
            [
                transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(512),
            ]
        )
        return image_transforms

    def load_model(self):
        pipeline = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            safety_checker=None,
            torch_dtype=torch.float32,
        )
        pipeline = pipeline.to(self.device)
        return pipeline

    def get_latent(self, tensor):
        latent_features = self.full_sd_model.vae.encode(tensor).latent_dist.mean
        return latent_features
    
    def get_tensor(self, latent):
        img_tensor = self.full_sd_model.vae.decode(latent).sample
        return img_tensor

    def _resize_and_pad_image(self, image: Image.Image, target_size: int = 512, padding_color=(0, 0, 0)) -> Image.Image:
        original_width, original_height = image.size

        if original_width > original_height:
            scale_factor = target_size / original_width
        else:
            scale_factor = target_size / original_height

        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        delta_w = target_size - new_width
        delta_h = target_size - new_height

        padding_left = delta_w // 2
        padding_top = delta_h // 2
        padding_right = delta_w - padding_left
        padding_bottom = delta_h - padding_top

        padded_image = ImageOps.expand(resized_image, border=(padding_left, padding_top, padding_right, padding_bottom), fill=padding_color)
        return padded_image

    def _remove_padding_from_tensor(self, tensor, original_width, original_height, target_size=512):
        temp_img = tensor2img(tensor)

        if original_width > original_height:
            scale_factor = target_size / original_width
        else:
            scale_factor = target_size / original_height
            
        content_width = int(original_width * scale_factor)
        content_height = int(original_height * scale_factor)

        delta_w = target_size - content_width
        delta_h = target_size - content_height
            
        padding_left = delta_w // 2
        padding_top = delta_h // 2

        cropped_img = temp_img.crop((padding_left, padding_top, padding_left + content_width, padding_top + content_height))
            
        final_img = cropped_img.resize((original_width, original_height), Image.Resampling.LANCZOS)
        return final_img

    def generate_one(self, id_image, anon_image):

        original_id_width, original_id_height = id_image.size
        
        id_image = self._resize_and_pad_image(id_image, target_size=512)
        anon_image = self._resize_and_pad_image(anon_image, target_size=512)
        

        resized_id_image = self.transform(id_image)
        source_tensor = img2tensor(resized_id_image).to(self.device)

        resized_anon_image = self.transform(anon_image)
        target_tensor = img2tensor(resized_anon_image).to(self.device)

        target_tensor = target_tensor
        source_tensor = source_tensor

        with torch.no_grad():
            target_latent = self.get_latent(target_tensor)
            source_latent = self.get_latent(source_tensor)

        modifier = torch.randn_like(source_tensor) * 0.01

        t_size = 100
        max_change = self.eps / 0.5  # scale from 0,1 to -1,1
        step_size = max_change
        mu = 1.0  # Decay factor for momentum
        g_t = torch.zeros_like(source_tensor)  # Initialize momentum term
        alpha = 1 if self.contrast else 0
        beta = 1  # Complementary weight
        
        for i in range(t_size):         
            actual_step_size = step_size - (step_size - step_size / 100) / t_size * i
            modifier.requires_grad_(True)

            adv_tensor = torch.clamp(modifier + source_tensor, -1, 1)
            adv_latent = self.get_latent(adv_tensor)
            recon_tensor = self.get_tensor(adv_latent)

            # Define loss function
            loss_away = (adv_latent - source_latent).norm()  # Move away from the source
            loss_close = (adv_latent - target_latent).norm()  # Move closer to the target
            loss_cvl = compute_score(source_tensor.float(), recon_tensor.float(), aligner=self.aligner, fr_model=self.fr_model)

            tot_loss = -alpha * loss_away + beta * loss_close - 0.5 * loss_cvl

            grad = torch.autograd.grad(tot_loss, modifier)[0]
            
            # Apply momentum iterative FGSM (MI-FGSM)
            g_t = mu * g_t + grad / torch.norm(grad, p=1)
            modifier = modifier - torch.sign(g_t) * actual_step_size
            modifier = torch.clamp(modifier, -max_change, max_change)
            modifier = modifier.detach()

            if i % 50 == 0:
                print("# Iter: {}\tLoss Away: {:.3f}\tLoss Close: {:.3f}\tTot Loss: {:.3f}".format(
                    i, loss_away.mean().item(), loss_close.mean().item(), tot_loss.mean().item()))

        final_adv_batch = torch.clamp(modifier + source_tensor, -1.0, 1.0)

        final_img = self._remove_padding_from_tensor(final_adv_batch, original_id_width, original_id_height, target_size=512)
        # final_img = tensor2img(final_adv_batch)

        return final_img

    def generate_all(self, id_folder: str, anon_folder: str, output_folder: str):

        os.makedirs(output_folder, exist_ok=True)

        for fname in os.listdir(id_folder):
            # 이미지 확장자 필터링
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            id_path   = os.path.join(id_folder,   fname)
            anon_path = os.path.join(anon_folder, f"{fname[:-3]}png") # id가 jpg일 때, anon이 png이면
            # anon_path = os.path.join(anon_folder, fname)

            if not os.path.isfile(anon_path):
                print(f"[SKIP] There is no anonymized image for '{fname}'")
                continue

            # 이미지 로드
            id_img   = Image.open(id_path).convert("RGB")
            anon_img = Image.open(anon_path).convert("RGB")

            # 개별 이미지 생성
            adv_img = self.generate_one(id_img, anon_img)

            # 파일명에서 확장자 제거 후 .png로 변경
            base_name = os.path.splitext(fname)[0]
            out_path  = os.path.join(output_folder, f"{base_name}.png")

            # PNG로 저장
            adv_img.save(out_path, format="PNG")
            print(f"[SAVE] Poisoned image saved to: {out_path}")


def img2tensor(cur_img):
    cur_img = cur_img.resize((512, 512), resample=Image.Resampling.BICUBIC)
    cur_img = np.array(cur_img)
    img = (cur_img / 127.5 - 1.0).astype(np.float32)
    img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img).unsqueeze(0)
    return img


def tensor2img(cur_img):
    if len(cur_img) == 512:
        cur_img = cur_img.unsqueeze(0)

    cur_img = torch.clamp((cur_img.detach() + 1.0) / 2.0, min=0.0, max=1.0)
    cur_img = 255. * rearrange(cur_img[0], 'c h w -> h w c').cpu().numpy()
    cur_img = Image.fromarray(cur_img.astype(np.uint8))
    return cur_img
