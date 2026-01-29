import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image



st.set_page_config(page_title="Fruit Classifier", layout="centered")
st.title("🍎 Fruit Image Classifier")
st.write("Upload an image of a fruit to get prediction")

device=torch.device('cpu')
class_names = ['apple', 'banana', 'grapes', 'mango', 'orange']


@st.cache_resource
def load_model():
    model=models.resnet18(pretrained=False)
    model.fc = nn.Linear(512, 5)
    model.load_state_dict(torch.load('model.pth', map_location=device))
    model.to(device)
    model.eval()
    return model
model = load_model()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean
    return tensor.clamp(0, 1)


uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


def predict_image(image):
    img = image.convert("RGB")
    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)
    vis_tensor = denormalize(input_tensor.squeeze()).permute(1, 2, 0)

    return class_names[pred.item()], confidence.item(), vis_tensor


if uploaded_file is not None:
    image = Image.open(uploaded_file)

    label, confidence , vis_tensor= predict_image(image)
    st.image(vis_tensor.numpy(), caption="Uploaded Image", use_column_width=True)


    st.markdown("### 🔍 Prediction Result")

    if confidence < 0.5:
        st.warning("⚠️ Model is not confident. Image may contain multiple fruits or invalid fruit image.")
    else:
        st.success(f"**Prediction:** {label}")
        st.info(f"**Confidence:** {confidence*100:.2f}%")












