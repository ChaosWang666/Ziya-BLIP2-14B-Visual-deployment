## 大模型部署实战（二）——Ziya-BLIP2-14B-Visual
原文链接：[http://www.yourmetaverse.cn/llm/208/](http://www.yourmetaverse.cn/llm/208/)
Ziya-Visual多模态大模型基于姜子牙通用大模型V1训练，具有视觉问答和对话能力。今年3月份OpenAI发布具有识图能力的多模态大模型GPT-4，遗憾的是，时至今日绝大部分用户也都还没有拿到GPT-4输入图片的权限，Ziya-Visual参考了Mini-GPT4、LLaVA等优秀的开源实现，补齐了Ziya的识图能力，使中文用户群体可以体验到结合视觉和语言两大模态的大模型的卓越能力。本文主要用于Ziya-LLaMA-13B的本地部署。

在线体验地址：
- 博主自己部署的地址：[http://www.yourmetaverse.cn:39001/](http://www.yourmetaverse.cn:39001/)
- 官方部署的地址：[https://huggingface.co/spaces/IDEA-CCNL/Ziya-BLIP2-14B-Visual-v1-Demo](https://huggingface.co/spaces/IDEA-CCNL/Ziya-BLIP2-14B-Visual-v1-Demo)

本文所使用的代码详见：[https://github.com/ChaosWang666/Ziya-BLIP2-14B-Visual-deployment](https://github.com/ChaosWang666/Ziya-BLIP2-14B-Visual-deployment)

## 1. 部署准备
### 1.1 硬件环境
显卡最低显存为28GB，可以为一张A100（40GB）。

### 1.2 python环境

```bash
cpm_kernels==1.0.11
gradio==3.34.0
huggingface_hub==0.15.1
Pillow==9.5.0
torch==1.12.1+cu113
tqdm==4.64.1
transformers==4.29.2
accelerate==0.17.1
```
### 1.3 模型下载
- **（1）Ziya-LLaMA-13B-v1.1模型的下载和转换**
该部分内容详见：[https://www.yourmetaverse.cn/llm/188/](https://www.yourmetaverse.cn/llm/188/)或者[https://welearnnlp.blog.csdn.net/article/details/131125481](https://welearnnlp.blog.csdn.net/article/details/131125481)

- **（2）Ziya-BLIP2-14B-Visual-v1参数下载**
BLIP模型的参数从下面网址下载：
[https://huggingface.co/IDEA-CCNL/Ziya-BLIP2-14B-Visual-v1/tree/main](https://huggingface.co/IDEA-CCNL/Ziya-BLIP2-14B-Visual-v1/tree/main)

## 2. 模型部署
首先说明一下项目的文件系统目录

```bash
-ziya_v1.1
-Ziya-BLIP2-14B-Visual-v1 
-launch.py
```
其中ziya_v1.1为转换后的ziya模型参数文件夹，Ziya-BLIP2-14B-Visual-v1为huggingface官网下载的blip模型权重文件。

使用下面这段代码进行模型部署。该代码修改自[https://huggingface.co/spaces/IDEA-CCNL/Ziya-BLIP2-14B-Visual-v1-Demo/tree/main](https://huggingface.co/spaces/IDEA-CCNL/Ziya-BLIP2-14B-Visual-v1-Demo/tree/main)
在官方代码的基础上增加了一些参数和优化内容。

```python
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
```
该代码中需要修改的地方已经在代码块上标出。

全面内容见github [https://github.com/ChaosWang666/Ziya-BLIP2-14B-Visual-deployment](https://github.com/ChaosWang666/Ziya-BLIP2-14B-Visual-deployment)
