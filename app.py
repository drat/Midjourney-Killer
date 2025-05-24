#!/usr/bin/env python
# ==============================================================================
#  File        : app.py
#  Project     : MidJourney-style AI Image Generator
#  Description : Aplikasi web untuk menghasilkan gambar AI dengan berbagai gaya
#                visual seperti MidJourney, menggunakan Stable Diffusion dan
#                Gradio sebagai antarmuka pengguna. Terdapat filtering kata 
#                terlarang untuk keamanan dan negative prompt agar hasil gambar
#                lebih bersih dan realistis.
#
#  Dibuat oleh : _drat | 2025
# ==============================================================================


import os
import random
import uuid
import json

import gradio as gr
import numpy as np
from PIL import Image
import spaces
import torch
from diffusers import DiffusionPipeline
from typing import Tuple

# Atau bisa override di sini secara hardcoded:
bad_words = [
    "nude", "nudity", "sex", "sexual", "erotic", "nsfw", "porn", "penis", "vagina", 
    "breasts", "boobs", "nipples", "testicles", "genital", "cum", "ejaculation", "sperm",
    "masturbation", "orgasm", "dildo", "bondage", "fetish", "gore", "blood", "violence",
    "death", "dead", "murder", "decapitated", "beheaded", "dismembered", "hanged", "abuse",
    "torture", "molest", "rape", "incest", "child", "minor", "underage", "loli", "shota",
    "pedophile", "bestiality", "zoophilia", "animal sex", "scat", "feces", "poop", "urine",
    "piss", "shit", "anus", "anal", "rectum", "grotesque", "hentai"
]
# bad_words_negative juga bisa di-hardcode sesuai kebutuhan

# ================== DEFAULT NEGATIVE PROMPT UNTUK MIDJOURNEY/SD ==================

default_negative = (
    "blurry, grainy, out of focus, bad anatomy, extra fingers, mutated hands, "
    "deformed, poorly drawn face, poorly drawn hands, text, watermark, signature, "
    "logo, cut off, worst quality, low quality, jpeg artifacts, morbid, mutilated, "
    "disfigured, tiling, duplicate, cropped, frame, error, ugly, gross proportions, "
    "long neck, cloned face, malformed limbs, fused fingers, missing fingers, "
    "missing arms, missing legs, extra limbs, extra arms, extra legs, mutated, "
    "mutation, bad proportions, out of frame, disconnected limbs, low resolution, "
    "overexposed, underexposed, bad lighting, unrealistic"
)
# Ini adalah daftar kata/kalimat yang selalu digunakan di negative prompt untuk mencegah hasil yang jelek/aneh dari AI

# ========================= FUNGSI FILTER/SANITASI PROMPT ========================

def check_text(prompt, negative=""):
    """
    Fungsi untuk melakukan filter pada prompt (baik prompt utama maupun negatif)
    supaya tidak mengandung kata-kata terlarang atau konten terlarang.
    Return True jika ada kata terlarang, False jika aman.
    """

    # Gabungkan prompt positif dan negatif jadi satu string untuk scanning
    combined = f"{prompt} {negative}".lower().strip()

    # Cek kata terlarang di BAD_WORDS (baik di prompt utama maupun negatif)
    for word in bad_words:
        if word.lower().strip() in combined:
            return True

    # Cek kata terlarang khusus BAD_WORDS_NEGATIVE di prompt negatif saja
    negative_lower = negative.lower().strip()
    for word in bad_words_negative:
        if word.lower().strip() in negative_lower:
            return True

    return False

# Style presets
style_list = [
    {
        "name": "2560 x 1440",
        "prompt": "hyper-realistic 4K image of {prompt}. ultra-detailed, lifelike, high-resolution, sharp, vibrant colors, photorealistic",
        "negative_prompt": "cartoonish, low resolution, blurry, simplistic, abstract, deformed, ugly",
    },
    {
        "name": "Photo",
        "prompt": "cinematic photo {prompt}. 35mm photograph, film, bokeh, professional, 4k, highly detailed",
        "negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly",
    },   
    {
        "name": "Cinematic",
        "prompt": "cinematic still {prompt}. emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
        "negative_prompt": "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
    },
    {
        "name": "Anime",
        "prompt": "anime artwork {prompt}. anime style, key visual, vibrant, studio anime, highly detailed",
        "negative_prompt": "photo, deformed, black and white, realism, disfigured, low contrast",
    },
    {
        "name": "3D Model",
        "prompt": "professional 3d model {prompt}. octane render, highly detailed, volumetric, dramatic lighting",
        "negative_prompt": "ugly, deformed, noisy, low poly, blurry, painting",
    },
    {
        "name": "(No style)",
        "prompt": "{prompt}",
        "negative_prompt": "",
    },
]

styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in style_list}
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "2560 x 1440"

def apply_style(style_name: str, positive: str, negative: str = "") -> Tuple[str, str]:
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    if not negative:
        negative = ""
    return p.replace("{prompt}", positive), n + negative

DESCRIPTION = """## MidJourney
Generator Gambar AI Gratis dan Tanpa Batas
Buat gambar unik, kreatif, dan berkualitas tinggi secara otomatis dengan teknologi AI canggih tanpa batasan!
"""

# =====================================================================
# KONFIGURASI DAN INISIALISASI PIPELINE DI AWAL
# =====================================================================

# Jika CUDA/GPU tidak tersedia, tambahkan peringatan ke user di UI.
if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>⚠️Running on CPU, This may not work on CPU.</p>"

# Mendefinisikan nilai maksimum untuk random seed (untuk reproduksibilitas gambar)
MAX_SEED = np.iinfo(np.int32).max

# Menentukan apakah contoh prompt akan dicache oleh Gradio.
# Contoh hanya dicache jika GPU tersedia dan CACHE_EXAMPLES diset di environment.
CACHE_EXAMPLES = torch.cuda.is_available() and os.getenv("CACHE_EXAMPLES", "0") == "1"

# Maksimum resolusi gambar yang diperbolehkan (bisa diubah lewat env variable)
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "2048"))

# Opsi untuk menggunakan torch.compile (kompilasi graf UNet untuk performa, jika didukung)
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE", "0") == "1"

# Opsi untuk mengaktifkan CPU offload (model dipindah ke CPU saat idle, menghemat VRAM)
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", "0") == "1"

# Jumlah gambar yang dihasilkan per prompt (bisa diubah sesuai kebutuhan UI)
NUM_IMAGES_PER_PROMPT = 1

# =====================================================================
# INISIALISASI PIPELINE DIFFUSION (SUPPORT GPU/CPU)
# =====================================================================

# Menentukan device yang akan dipakai ('cuda' untuk GPU, 'cpu' jika tidak ada GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inisialisasi model pipeline:
# Jika menggunakan GPU (CUDA tersedia)
if torch.cuda.is_available():
    pipe = DiffusionPipeline.from_pretrained(
        "SG161222/RealVisXL_V3.0_Turbo",
        torch_dtype=torch.float16,          # float16 hanya didukung di GPU
        use_safetensors=True,               # gunakan file .safetensors (lebih aman)
        add_watermarker=False,              # nonaktifkan watermark otomatis
        variant="fp16"                      # pastikan varian float16
    )
    pipe2 = DiffusionPipeline.from_pretrained(
        "SG161222/RealVisXL_V2.02_Turbo",
        torch_dtype=torch.float16,
        use_safetensors=True,
        add_watermarker=False,
        variant="fp16"
    )
    # Opsi: model bisa di-offload ke CPU saat idle (hemat VRAM)
    if ENABLE_CPU_OFFLOAD:
        pipe.enable_model_cpu_offload()
        pipe2.enable_model_cpu_offload()
    else:
        # Pindahkan model ke GPU
        pipe.to(device)
        pipe2.to(device)
    # Opsi: compile graf UNet dengan torch.compile (hanya jika torch >= 2.0)
    if USE_TORCH_COMPILE:
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
        pipe2.unet = torch.compile(pipe2.unet, mode="reduce-overhead", fullgraph=True)
else:
    # Jika hanya CPU, load pipeline dengan setting default (tanpa float16)
    pipe = DiffusionPipeline.from_pretrained(
        "SG161222/RealVisXL_V3.0_Turbo",
        use_safetensors=True,
        add_watermarker=False
    )
    pipe.to(device)
    pipe2 = DiffusionPipeline.from_pretrained(
        "SG161222/RealVisXL_V2.02_Turbo",
        use_safetensors=True,
        add_watermarker=False
    )
    pipe2.to(device)

# =====================================================================
# FUNGSI UNTUK MENYIMPAN GAMBAR KE FILE PNG DENGAN NAMA UNIK
# =====================================================================
def save_image(img):
    unique_name = str(uuid.uuid4()) + ".png"
    img.save(unique_name)
    return unique_name

# =====================================================================
# FUNGSI UNTUK RANDOMISASI SEED (JIKA DIPILIH OLEH USER)
# =====================================================================
def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

# =====================================================================
# FUNGSI UTAMA UNTUK GENERASI GAMBAR (AKAN DIPANGGIL DARI UI GRADIO)
# =====================================================================
@spaces.GPU(enable_queue=True)   # Dekorator: prioritas eksekusi di GPU pada HuggingFace Spaces
def generate(
    prompt: str,
    negative_prompt: str = "",
    use_negative_prompt: bool = False,
    style: str = DEFAULT_STYLE_NAME,
    seed: int = 0,
    width: int = 1024,
    height: int = 1024,
    guidance_scale: float = 3,
    randomize_seed: bool = False,
    use_resolution_binning: bool = True,
    progress=gr.Progress(track_tqdm=True),
):
    # Validasi isi prompt agar tidak mengandung kata terlarang
    if check_text(prompt, negative_prompt):
        raise ValueError("Prompt contains restricted words.")
    
    # Terapkan preset style pada prompt (style dan negative prompt)
    prompt, negative_prompt = apply_style(style, prompt, negative_prompt)
    
    # Atur seed (acak jika diaktifkan)
    seed = int(randomize_seed_fn(seed, randomize_seed))
    generator = torch.Generator().manual_seed(seed)

    # Jika opsi negative prompt dimatikan oleh user
    if not use_negative_prompt:
        negative_prompt = ""
    negative_prompt += default_negative    # Tambahkan default negative prompt

    # Siapkan semua parameter untuk pipeline
    options = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "guidance_scale": guidance_scale,
        "num_inference_steps": 25,                  # Jumlah langkah inference (standar SD)
        "generator": generator,                     # Untuk memastikan seed konsisten
        "num_images_per_prompt": NUM_IMAGES_PER_PROMPT,
        "use_resolution_binning": use_resolution_binning,
        "output_type": "pil",                       # Output gambar tipe PIL
    }
    
    # Jalankan inference di kedua pipeline dan gabungkan hasilnya
    images = pipe(**options).images + pipe2(**options).images

    # Simpan semua gambar ke file dengan nama unik, return path dan seed yang dipakai
    image_paths = [save_image(img) for img in images]
    return image_paths, seed


examples = [
    "Kucing dalam close-up di dekat jendela, dengan gaya Annie Leibovitz dan Wes Anderson, di dalam kabin pedesaan, dengan kedalaman bidang yang dangkal dan efek film vintage grain. --ar 85:128 --v 6.0 --stil mentah",
    "Daria, karakter utama dari serial animasi Daria Morgendorffer, dengan ekspresi serius, tampilan sensual yang sangat menarik, gadis yang sangat menawan, cantik dan karismatik, pose menggoda, wanita berkacamata dengan figur mempesona dan bentuk menarik, figur ukuran nyata",
    "Daun anthurium besar berwarna hijau tua, tampak dekat, foto dari atas, dalam gaya Unsplash, menggunakan kamera Hasselblad h6d400c --ar 85:128 --v 6.0 --gaya mentah",
    "Close-up wanita pirang dengan kedalaman bidang dangkal, bokeh, fokus dangkal, minimalisme, menggunakan lensa Canon EF di Fujifilm xh2s, tampilan sinematik --ar 85:128 --v 6.0 --gaya mentah"
]

css = '''
.gradio-container{max-width: 700px !important}
h1{text-align:center}
'''
with gr.Blocks(css=css, theme="bethecloud/storj_theme") as demo:
    gr.Markdown(DESCRIPTION)
    gr.DuplicateButton(
        value="Duplicate Space for private use",
        elem_id="duplicate-button",
        visible=os.getenv("SHOW_DUPLICATE_BUTTON") == "1",
    )
    with gr.Group():
        with gr.Row():
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )
            run_button = gr.Button("Run")
        result = gr.Gallery(label="Result", columns=1, preview=True)
    with gr.Accordion("Advanced options", open=False):
        use_negative_prompt = gr.Checkbox(label="Use negative prompt", value=True, visible=True)
        negative_prompt = gr.Text(
            label="Negative prompt",
            max_lines=1,
            placeholder="Enter a negative prompt",
            value="(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
            visible=True,
        )
        with gr.Row():
            num_inference_steps = gr.Slider(
                label="Steps",
                minimum=10,
                maximum=60,
                step=1,
                value=30,
            )
        with gr.Row():
            num_images_per_prompt = gr.Slider(
                label="Images",
                minimum=1,
                maximum=5,
                step=1,
                value=2,
            )
        seed = gr.Slider(
            label="Seed",
            minimum=0,
            maximum=MAX_SEED,
            step=1,
            value=0,
            visible=True
        )
        randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
        with gr.Row(visible=True):
            width = gr.Slider(
                label="Width",
                minimum=512,
                maximum=2048,
                step=8,
                value=1024,
            )
            height = gr.Slider(
                label="Height",
                minimum=512,
                maximum=2048,
                step=8,
                value=1024,
            )
        with gr.Row():
            guidance_scale = gr.Slider(
                label="Guidance Scale",
                minimum=0.1,
                maximum=20.0,
                step=0.1,
                value=6,
            )
    with gr.Row(visible=True):
        style_selection = gr.Radio(
            show_label=True,
            container=True,
            interactive=True,
            choices=STYLE_NAMES,
            value=DEFAULT_STYLE_NAME,
            label="Image Style",
        )
    gr.Examples(
        examples=examples,
        inputs=prompt,
        outputs=[result, seed],
        fn=generate,
        cache_examples=CACHE_EXAMPLES,
    )

    use_negative_prompt.change(
        fn=lambda x: gr.update(visible=x),
        inputs=use_negative_prompt,
        outputs=negative_prompt,
        api_name=False,
    )

    gr.on(
        triggers=[
            prompt.submit,
            negative_prompt.submit,
            run_button.click,
        ],
        fn=generate,
        inputs=[
            prompt,
            negative_prompt,
            use_negative_prompt,
            style_selection,
            seed,
            width,
            height,
            guidance_scale,
            randomize_seed,
        ],
        outputs=[result, seed],
        api_name="run",
    )

if __name__ == "__main__":
    demo.queue(max_size=20).launch()
