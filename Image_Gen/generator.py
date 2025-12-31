import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionXLImg2ImgPipeline
)
import json
import os
import time
from tqdm import tqdm
from datetime import datetime
from PIL import Image


class ImageGenerator:
    def __init__(self, config_path="config.json"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device detected: {self.device}")

        if self.device == "cpu":
            self.dtype = torch.float32
            print("[INFO] Running on CPU. Using float32.")
        else:
            self.dtype = torch.float16
            print("[INFO] Running on GPU. Using float16.")

        self.config = self.load_config(config_path)
        self.current_model = None
        self.current_pipeline = None

        self.MODELS_CONFIG = {
            "sd_v1.5": {
                "id": "runwayml/stable-diffusion-v1-5",
                "type": "standard",
                "name": "Stable Diffusion v1.5"
            },
            # "sd_v2.1": {
            #     "id": "stabilityai/stable-diffusion-2-1",
            #     "type": "standard",
            #     "name": "Stable Diffusion v2.1"
            # },
            "sdxl": {
                "id": "stabilityai/stable-diffusion-xl-base-1.0",
                "type": "xl",
                "name": "SDXL 1.0 Base"
            }
        }

    def load_config(self, path):
        default_config = {
            "default_steps": 15,
            "default_guidance": 7.5,
            "default_width": 512,
            "default_height": 512,
            "output_folder": "outputs"
        }
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return {**default_config, **json.load(f)}
            except Exception:
                pass
        return default_config

    def save_config(self, path="config.json"):
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=4)

    def load_model(self, model_key):
        if model_key not in self.MODELS_CONFIG:
            raise ValueError(f"Model {model_key} not found")

        if self.current_model == model_key and self.current_pipeline is not None:
            return

        print(f"\n[INFO] Loading model: {model_key}...")

        if self.current_pipeline is not None:
            del self.current_pipeline
            self.current_pipeline = None
            if self.device == "cuda":
                torch.cuda.empty_cache()

        model_info = self.MODELS_CONFIG[model_key]

        load_args = {
            "torch_dtype": self.dtype,
            "use_safetensors": True
        }
        if self.device == "cuda":
            load_args["variant"] = "fp16"

        try:
            if model_info["type"] == "xl":
                self.current_pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_info["id"],
                    **load_args
                )
            else:
                self.current_pipeline = StableDiffusionPipeline.from_pretrained(
                    model_info["id"],
                    **load_args
                )

            self.current_pipeline.to(self.device)
            self.current_model = model_key
            print(f"[SUCCESS] {model_key} loaded.")

        except Exception as e:
            print(f"[ERROR] Failed to load {model_key}: {e}")
            self.current_model = None
            raise e

    def generate_batch(self, model_key, prompt, num_images=1, output_dir=None):
        self.load_model(model_key)

        out_path = output_dir if output_dir else self.config["output_folder"]
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        is_xl = self.MODELS_CONFIG[model_key]["type"] == "xl"
        if is_xl and self.device == "cpu":
            # Зменшуємо розмір ще сильніше для тесту SDXL на CPU(ноуту 13 років :'( )
            width, height = 512, 512
        elif is_xl:
            width, height = 1024, 1024
        else:
            width, height = self.config["default_width"], self.config["default_height"]

        results = []
        print(f"\n[GENERTE] Processing '{prompt}'...")

        for i in tqdm(range(num_images), desc="Generating"):
            generator = torch.Generator(device=self.device)
            seed = generator.seed()

            image = self.current_pipeline(
                prompt=prompt,
                num_inference_steps=self.config["default_steps"],
                guidance_scale=self.config["default_guidance"],
                width=width,
                height=height,
                generator=generator
            ).images[0]

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{out_path}/{model_key}_{timestamp}_{i}.png"
            image.save(filename)
            results.append(filename)

        return results

    def generate_img2img(self, model_key, prompt, init_image_path, strength=0.75):
        if not os.path.exists(init_image_path):
            print(f"[ERROR] Not found: {init_image_path}")
            return None

        init_image = Image.open(init_image_path).convert("RGB")
        is_xl = self.MODELS_CONFIG[model_key]["type"] == "xl"

        target_size = (512, 512) if (self.device == "cpu") else ((1024, 1024) if is_xl else (512, 512))
        init_image = init_image.resize(target_size)

        self.load_model(model_key)
        print(f"\n[IMG2IMG] Switching pipeline...")

        if is_xl:
            img2img_pipe = StableDiffusionXLImg2ImgPipeline(**self.current_pipeline.components)
        else:
            img2img_pipe = StableDiffusionImg2ImgPipeline(**self.current_pipeline.components)

        img2img_pipe.to(self.device)

        result = img2img_pipe(
            prompt=prompt,
            image=init_image,
            strength=strength,
            guidance_scale=self.config["default_guidance"],
            num_inference_steps=self.config["default_steps"]
        ).images[0]

        out_path = self.config["output_folder"]
        if not os.path.exists(out_path): os.makedirs(out_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{out_path}/img2img_{model_key}_{timestamp}.png"
        result.save(filename)
        self.current_model = None
        return filename

    def compare_models(self, prompt, seed=42, output_dir="comparisons"):
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        results = []

        print(f"\n[COMPARE] Starting comparison.")

        for key, info in self.MODELS_CONFIG.items():
            try:
                self.load_model(key)
                generator = torch.Generator(device=self.device).manual_seed(seed)

                is_xl = info["type"] == "xl"
                if is_xl and self.device == "cpu":
                    width, height = 512, 512
                elif is_xl:
                    width, height = 1024, 1024
                else:
                    width, height = 512, 512

                start_time = time.time()
                image = self.current_pipeline(
                    prompt=prompt,
                    width=width,
                    height=height,
                    generator=generator,
                    num_inference_steps=self.config["default_steps"]
                ).images[0]
                gen_time = round(time.time() - start_time, 2)

                filename = f"{key}_seed{seed}.png"
                image.save(os.path.join(output_dir, filename))

                results.append({
                    "model": info["name"], "time": gen_time,
                    "image": filename, "settings": f"Steps: {self.config['default_steps']}"
                })
            except Exception as e:
                print(f"Skipping {key} due to error: {e}")

        html_rows = "".join([
            f"<tr><td><b>{r['model']}</b></td><td><img src='{r['image']}' width='256'></td><td>{r['time']}s</td><td>{r['settings']}</td></tr>"
            for r in results])
        html = f"<html><body><h1>Comparison: {prompt}</h1><table border='1'>{html_rows}</table></body></html>"

        with open(os.path.join(output_dir, f"report_{seed}.html"), "w") as f:
            f.write(html)

    def generate_from_template(self, template_name, variables, model_key="sd_v1.5"):
        prompts_file = "prompts.json"
        if not os.path.exists(prompts_file):
            default_data = {
                "templates": {"portrait": "portrait of {subject}, {style}", "landscape": "{location}, {weather}"}}
            with open(prompts_file, "w") as f: json.dump(default_data, f)

        with open(prompts_file, "r") as f:
            data = json.load(f)
        if template_name not in data["templates"]: raise ValueError("Template not found")

        final_prompt = data["templates"][template_name].format(**variables)
        return self.generate_batch(model_key, final_prompt, num_images=1)

if __name__ == "__main__":
    gen = ImageGenerator()
    gen.save_config()
    gen.compare_models("astronaut riding a horse")
