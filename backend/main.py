import io
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torchvision.transforms as transforms

# 1. PHẢI CÓ KIẾN TRÚC MODEL Ở ĐÂY (Ví dụ ông dùng Swin)
# từ model_architecture import MyMathModel 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. LOAD MODEL KHI STARTUP
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = MyMathModel() # Thay bằng Class model của ông
# model.load_state_dict(torch.load("model.pth", map_location=device))
# model.eval().to(device)

@app.post("/predict")
async def predict_latex(file: UploadFile = File(...)):
    # Đọc ảnh
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    # 3. TIỀN XỬ LÝ (Phải giống hệt lúc ông train model)
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # Kích thước model yêu cầu
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,)) 
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    # 4. CHẠY MODEL ĐOÁN CHỮ
    with torch.no_grad():
        # output = model(input_tensor)
        # latex_result = decode_prediction(output) # Hàm giải mã output thành chữ
        
        # Tạm thời để test xem nó có vào đây không:
        latex_result = "Model .pth da nhan anh!" 
    
    return {"latex": latex_result}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)