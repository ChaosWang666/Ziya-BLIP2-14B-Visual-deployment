import gradio as gr
import re
from PIL import Image
import torch
from io import BytesIO
import hashlib
import os
from transformers import LlamaForCausalLM, LlamaTokenizer, BlipImageProcessor, BitsAndBytesConfig, AutoModelForCausalLM
import json

default_chatbox = []

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def is_chinese(text):
    zh_pattern = re.compile(u'[\u4e00-\u9fa5]+')
    return zh_pattern.search(text)

AUTH_TOKEN = os.getenv("AUTH_TOKEN")
#该部分需要修改模型路径为自己的路径
LM_MODEL_PATH = "./ziya_v1"
lm_model = LlamaForCausalLM.from_pretrained(
    LM_MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16,
    use_auth_token=AUTH_TOKEN,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True))
tokenizer = LlamaTokenizer.from_pretrained(LM_MODEL_PATH, use_auth_token=AUTH_TOKEN)

# visual model
OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
#该部分需要修改模型路径为自己的路径
model = AutoModelForCausalLM.from_pretrained(
    "./Ziya-BLIP2-14B-Visual-v1 ", 
    trust_remote_code=True, use_auth_token=AUTH_TOKEN,
    torch_dtype=torch.float16)
model.cuda()  # if you use on cpu, comment this line
model.language_model = lm_model
image_size = model.config.vision_config.image_size
image_processor = BlipImageProcessor(
    size={"height": image_size, "width": image_size},
    image_mean=OPENAI_CLIP_MEAN,
    image_std=OPENAI_CLIP_STD,
)

def post(
        input_text,
        temperature,
        top_p,
        image_prompt,
        result_previous,
        hidden_image
        ):
    result_text = [(ele[0], ele[1]) for ele in result_previous]
    previous_querys = []
    previous_outputs = []
    for i in range(len(result_text)-1, -1, -1):
        if result_text[i][0] == "":
            del result_text[i]
        else:
            previous_querys.append(result_text[i][0])
            previous_outputs.append(result_text[i][1])
            
    is_zh = is_chinese(input_text)

    if image_prompt is None:
        print("Image empty")
        if is_zh:
            result_text.append((input_text, '图片为空！请上传图片并重试。'))
        else:
            result_text.append((input_text, 'Image empty! Please upload a image and retry.'))
        return input_text, result_text, hidden_image
    elif input_text == "":
        print("Text empty")
        result_text.append((input_text, 'Text empty! Please enter text and retry.'))
        return "", result_text, hidden_image              

    generate_config = {
        "max_new_tokens": 128,
        "top_p": top_p,
        "temperature": temperature,
        "repetition_penalty": 1.18,
    }
    img = Image.open(image_prompt)
    pixel_values = image_processor(
        img, 
        return_tensors="pt").pixel_values.to(
            model.device).to(model.dtype)
    output_buffer = BytesIO()
    img.save(output_buffer, "PNG")
    byte_data = output_buffer.getvalue()
    md = hashlib.md5()
    md.update(byte_data)
    img_hash = md.hexdigest()
    if img_hash != hidden_image:
        previous_querys = []
        previous_outputs = []
        result_text = []

    answer = model.chat(
        tokenizer=tokenizer,
        pixel_values=pixel_values,
        query=input_text,
        previous_querys=previous_querys,
        previous_outputs=previous_outputs,
        **generate_config,
    )          

    result_text.append((input_text, answer))
    print(result_text)
    return "", result_text, img_hash


def clear_fn(value):
    return "", default_chatbox, None

def clear_fn2(value):
    return default_chatbox

def io_fn(a, b, c):
    print(f"call io_fn")
    return a, b


def main():
    gr.close_all()

    with gr.Blocks(css='style.css') as demo:

        with gr.Row():
            with gr.Column(scale=4.5):
                with gr.Group():
                    input_text = gr.Textbox(label='Input Text', placeholder='Please enter text prompt below and press ENTER.')
                    with gr.Row():
                        run_button = gr.Button('Generate')
                        clear_button = gr.Button('Clear')
                    image_prompt = gr.Image(type="filepath", label="Image Prompt", value=None)
                with gr.Row():
                    temperature = gr.Slider(maximum=1, value=0.7, minimum=0, label='Temperature')
                    top_p = gr.Slider(maximum=1, value=0.1, minimum=0, label='Top P')
            with gr.Column(scale=5.5):
                result_text = gr.components.Chatbot(label='Multi-round conversation History', value=[]).style(height=450)
                hidden_image_hash = gr.Textbox(visible=False)

        run_button.click(fn=post,inputs=[input_text, temperature, top_p, image_prompt, result_text, hidden_image_hash],
                         outputs=[input_text, result_text, hidden_image_hash])
        input_text.submit(fn=post,inputs=[input_text, temperature, top_p, image_prompt, result_text, hidden_image_hash],
                         outputs=[input_text, result_text, hidden_image_hash])
        clear_button.click(fn=clear_fn, inputs=clear_button, outputs=[input_text, result_text, image_prompt])
        image_prompt.upload(fn=clear_fn2, inputs=clear_button, outputs=[result_text])
        image_prompt.clear(fn=clear_fn2, inputs=clear_button, outputs=[result_text])

    demo.queue(concurrency_count=10)
    demo.launch(share=False,server_name="127.0.0.1", server_port=7886)


if __name__ == '__main__':
    main()
