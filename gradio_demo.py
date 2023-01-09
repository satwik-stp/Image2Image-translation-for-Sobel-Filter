import torch
from torchvision import transforms
from torchvision.utils import save_image
from src.experiment import train

model = train.Pix2Pix.load_from_checkpoint("models/epoch=4-step=1320.ckpt",hparams=torch.load("models/epoch=4-step=1320.ckpt",map_location=torch.device('cpu'))["hyper_parameters"])
model.eval()

def predict(inp):
    inp=inp.resize((128,128))
    inp = transforms.ToTensor()(inp).unsqueeze(0)
    with torch.no_grad():
        prediction = model(inp)
    save_image(prediction[0],"prediction.png")
    return "prediction.png"

import gradio as gr

gr.Interface(fn=predict,
             inputs=gr.Image(type="pil"),
             outputs="image",
             ).launch()