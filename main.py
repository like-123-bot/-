from __future__ import annotations

import base64
import io
import os
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from PIL import Image

from model import DigitCNN


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.pth"
IMAGE_SIZE = 28
MNIST_MEAN = 0.131015
MNIST_STD = 0.308540
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CanvasPayload(BaseModel):
    data_url: str


app = FastAPI(title="CNN Handwritten Digit Recognizer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_model() -> DigitCNN:
    model = DigitCNN().to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


MODEL = load_model()


def shift_on_black(image: Image.Image, shift_x: int, shift_y: int) -> Image.Image:
    shifted = Image.new("L", image.size, 0)
    width, height = image.size
    src_left = max(0, -shift_x)
    src_top = max(0, -shift_y)
    src_right = min(width, width - shift_x)
    src_bottom = min(height, height - shift_y)
    dst_left = max(0, shift_x)
    dst_top = max(0, shift_y)
    if src_right > src_left and src_bottom > src_top:
        crop = image.crop((src_left, src_top, src_right, src_bottom))
        shifted.paste(crop, (dst_left, dst_top))
    return shifted


def preprocess_image(image: Image.Image) -> Image.Image:
    pil = image.convert("RGBA")
    white_bg = Image.new("RGBA", pil.size, (255, 255, 255, 255))
    pil = Image.alpha_composite(white_bg, pil).convert("L")

    arr = np.asarray(pil).astype(np.float32)
    if arr.mean() > 127:
        arr = 255.0 - arr

    arr[arr < 20] = 0
    ys, xs = np.where(arr > 20)
    if len(xs) == 0 or len(ys) == 0:
        raise HTTPException(status_code=400, detail="没有检测到有效笔迹，请重新输入。")

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    digit = Image.fromarray(np.clip(arr[y0 : y1 + 1, x0 : x1 + 1], 0, 255).astype(np.uint8))

    w, h = digit.size
    scale = 20.0 / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    digit = digit.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)
    left = (IMAGE_SIZE - new_w) // 2
    top = (IMAGE_SIZE - new_h) // 2
    canvas.paste(digit, (left, top))

    centered = np.asarray(canvas).astype(np.float32)
    ys, xs = np.where(centered > 20)
    if len(xs) and len(ys):
        weights = centered[ys, xs]
        cx = float((xs * weights).sum() / weights.sum())
        cy = float((ys * weights).sum() / weights.sum())
        canvas = shift_on_black(canvas, int(round(IMAGE_SIZE / 2 - cx)), int(round(IMAGE_SIZE / 2 - cy)))
    return canvas


def predict_image(image: Image.Image) -> dict[str, object]:
    processed = preprocess_image(image)
    arr = np.asarray(processed).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).view(1, 1, IMAGE_SIZE, IMAGE_SIZE)
    tensor = ((tensor - MNIST_MEAN) / MNIST_STD).to(DEVICE)

    with torch.no_grad():
        logits = MODEL(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    top_indices = np.argsort(probs)[::-1][:3]
    prediction = int(top_indices[0])
    return {
        "prediction": prediction,
        "confidence": float(probs[prediction]),
        "top3": [{"digit": int(i), "probability": float(probs[i])} for i in top_indices],
        "probabilities": [{"digit": i, "probability": float(probs[i])} for i in range(10)],
    }


def image_from_data_url(data_url: str) -> Image.Image:
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    try:
        raw = base64.b64decode(data_url)
        return Image.open(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="画板图片解析失败。") from exc


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "device": str(DEVICE)}


@app.post("/api/predict-upload")
async def predict_upload(file: UploadFile = File(...)) -> dict[str, object]:
    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="上传文件不是有效图片。") from exc
    return predict_image(image)


@app.post("/api/predict-canvas")
async def predict_canvas(payload: CanvasPayload) -> dict[str, object]:
    return predict_image(image_from_data_url(payload.data_url))


# 核心修改部分：全新的HTML/CSS样式
HTML = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>手写数字识别 | CNN深度学习模型</title>
  <style>
    :root { 
      --primary: #6366f1; 
      --primary-dark: #4f46e5;
      --secondary: #10b981;
      --secondary-dark: #059669;
      --accent: #f59e0b;
      --dark: #1e293b;
      --light: #f8fafc;
      --gray: #64748b;
      --light-gray: #e2e8f0;
      --danger: #ef4444;
      --shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
      --shadow-hover: 0 8px 30px rgba(0, 0, 0, 0.12);
      --radius: 12px;
      --transition: all 0.3s ease;
    }
    * { 
      box-sizing: border-box; 
      margin: 0;
      padding: 0;
    }
    body { 
      min-height: 100vh; 
      font-family: "Inter", "Microsoft YaHei", sans-serif; 
      color: var(--dark); 
      background: linear-gradient(135deg, #f0f4ff 0%, #e6e9ff 100%);
      line-height: 1.6;
    }
    .container { 
      max-width: 1280px; 
      margin: 0 auto; 
      padding: 2rem 1rem; 
    }
    /* 顶部标题区域 */
    .header {
      margin-bottom: 2rem;
      text-align: center;
    }
    .header__title {
      font-size: 2.5rem;
      font-weight: 800;
      color: var(--primary-dark);
      margin-bottom: 0.5rem;
      position: relative;
      display: inline-block;
    }
    .header__title::after {
      content: '';
      position: absolute;
      bottom: -8px;
      left: 10%;
      width: 80%;
      height: 4px;
      background: linear-gradient(90deg, var(--primary), var(--secondary));
      border-radius: 2px;
    }
    .header__subtitle {
      color: var(--gray);
      max-width: 800px;
      margin: 0 auto;
      font-size: 1.1rem;
    }
    .status-bar {
      display: flex;
      gap: 1rem;
      justify-content: center;
      flex-wrap: wrap;
      margin: 1rem 0 2rem;
    }
    .status-chip {
      padding: 0.5rem 1rem;
      border-radius: 999px;
      background: white;
      color: var(--primary);
      font-weight: 600;
      font-size: 0.9rem;
      box-shadow: var(--shadow);
      border: 1px solid var(--light-gray);
    }
    /* 主工作区 */
    .workspace {
      display: grid;
      grid-template-columns: minmax(0, 1.5fr) minmax(380px, 1fr);
      gap: 2rem;
      align-items: start;
    }
    @media (max-width: 992px) {
      .workspace {
        grid-template-columns: 1fr;
      }
    }
    /* 卡片样式 */
    .card {
      background: white;
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      overflow: hidden;
      transition: var(--transition);
    }
    .card:hover {
      box-shadow: var(--shadow-hover);
    }
    .card__header {
      padding: 1.5rem;
      background: linear-gradient(135deg, var(--primary), var(--primary-dark));
      color: white;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .card__title {
      font-size: 1.4rem;
      font-weight: 700;
    }
    .card__body {
      padding: 1.5rem;
    }
    /* 标签页样式 */
    .tabs {
      display: flex;
      gap: 0.5rem;
    }
    .tab {
      padding: 0.5rem 1.2rem;
      border-radius: 8px;
      cursor: pointer;
      font-weight: 600;
      background: rgba(255, 255, 255, 0.2);
      color: white;
      border: none;
      transition: var(--transition);
    }
    .tab.active {
      background: white;
      color: var(--primary-dark);
    }
    .tab:hover:not(.active) {
      background: rgba(255, 255, 255, 0.3);
    }
    /* 面板样式 */
    .panel {
      display: none;
    }
    .panel.active {
      display: block;
    }
    /* 上传区域 */
    .upload-area {
      border: 2px dashed var(--light-gray);
      border-radius: var(--radius);
      padding: 2rem;
      text-align: center;
      background: var(--light);
      transition: var(--transition);
      margin-bottom: 1.5rem;
    }
    .upload-area:hover {
      border-color: var(--primary);
      background: #fafbff;
    }
    .upload-area__title {
      font-weight: 600;
      margin-bottom: 0.5rem;
      color: var(--dark);
    }
    .upload-area__desc {
      color: var(--gray);
      font-size: 0.9rem;
      margin-bottom: 1rem;
    }
    #fileInput {
      width: 100%;
      max-width: 500px;
      padding: 0.8rem;
      border: 1px solid var(--light-gray);
      border-radius: 8px;
      background: white;
      font-family: inherit;
      margin: 0 auto;
      display: block;
    }
    #preview {
      display: none;
      width: 200px;
      height: 200px;
      object-fit: contain;
      margin: 1rem auto 0;
      border-radius: 8px;
      border: 1px solid var(--light-gray);
      background: white;
      box-shadow: var(--shadow);
    }
    /* 绘画区域 */
    .draw-area {
      display: grid;
      grid-template-columns: 320px 1fr;
      gap: 1.5rem;
      align-items: start;
    }
    @media (max-width: 768px) {
      .draw-area {
        grid-template-columns: 1fr;
      }
    }
    #canvas {
      width: 100%;
      max-width: 300px;
      height: 300px;
      background: white;
      border-radius: var(--radius);
      border: 2px solid var(--light-gray);
      touch-action: none;
      box-shadow: inset 0 0 10px rgba(0, 0, 0, 0.05);
    }
    .draw__tips {
      color: var(--gray);
      font-size: 0.95rem;
      margin-bottom: 1.5rem;
      line-height: 1.7;
    }
    /* 按钮样式 */
    .btn-group {
      display: flex;
      gap: 1rem;
      flex-wrap: wrap;
    }
    .btn {
      padding: 0.8rem 1.5rem;
      border-radius: 8px;
      border: none;
      font-weight: 700;
      cursor: pointer;
      transition: var(--transition);
      font-size: 0.95rem;
    }
    .btn--primary {
      background: var(--primary);
      color: white;
    }
    .btn--primary:hover:not(:disabled) {
      background: var(--primary-dark);
      transform: translateY(-2px);
    }
    .btn--secondary {
      background: var(--secondary);
      color: white;
    }
    .btn--secondary:hover:not(:disabled) {
      background: var(--secondary-dark);
      transform: translateY(-2px);
    }
    .btn:disabled {
      opacity: 0.6;
      cursor: not-allowed;
      transform: none;
    }
    /* 错误提示 */
    .error {
      margin-top: 1rem;
      color: var(--danger);
      font-weight: 600;
      min-height: 1.6rem;
      text-align: center;
    }
    /* 结果区域 */
    .result__header {
      background: linear-gradient(135deg, var(--secondary), var(--secondary-dark));
    }
    .result__stage {
      display: grid;
      grid-template-columns: 160px 1fr;
      gap: 1.5rem;
      align-items: center;
      padding: 1.5rem;
      background: #f0fdf4;
      border-radius: var(--radius);
      margin-bottom: 2rem;
    }
    @media (max-width: 768px) {
      .result__stage {
        grid-template-columns: 1fr;
        text-align: center;
      }
    }
    .confidence-ring {
      width: 150px;
      height: 150px;
      border-radius: 50%;
      background: conic-gradient(var(--primary) calc(var(--score,0) * 1%), #e2e8f0 0);
      display: grid;
      place-items: center;
      position: relative;
      margin: 0 auto;
    }
    .confidence-ring::after {
      content: "";
      position: absolute;
      width: 110px;
      height: 110px;
      background: white;
      border-radius: 50%;
      box-shadow: inset 0 0 0 1px var(--light-gray);
    }
    .metric {
      position: relative;
      z-index: 1;
      font-size: 60px;
      font-weight: 900;
      line-height: 1;
      color: var(--primary-dark);
    }
    .result__caption {
      color: var(--gray);
      font-weight: 600;
      margin-bottom: 0.5rem;
    }
    .confidence-text {
      font-size: 28px;
      font-weight: 800;
      color: var(--dark);
      margin-bottom: 0.5rem;
    }
    .mini-note {
      color: var(--gray);
      font-size: 0.9rem;
      line-height: 1.6;
    }
    /* 表格样式 */
    .table {
      width: 100%;
      border-collapse: collapse;
      margin: 1rem 0;
      font-size: 0.95rem;
    }
    .table th {
      padding: 1rem;
      text-align: left;
      background: var(--light);
      color: var(--dark);
      font-weight: 700;
      border-bottom: 2px solid var(--light-gray);
    }
    .table td {
      padding: 1rem;
      border-bottom: 1px solid var(--light-gray);
    }
    .table tr:hover td {
      background: #f8fafc;
    }
    /* 进度条样式 */
    .bars {
      display: grid;
      gap: 1rem;
      margin: 1rem 0;
    }
    .bar-row {
      display: grid;
      grid-template-columns: 30px 1fr 60px;
      gap: 1rem;
      align-items: center;
    }
    .bar-track {
      height: 16px;
      background: var(--light-gray);
      border-radius: 999px;
      overflow: hidden;
      box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);
    }
    .bar-fill {
      height: 100%;
      background: linear-gradient(90deg, var(--primary), var(--secondary));
      border-radius: 999px;
      transition: width 0.3s ease;
    }
    /* 区域标题 */
    .section-title {
      font-size: 1.1rem;
      font-weight: 800;
      color: var(--dark);
      margin: 2rem 0 1rem;
      display: flex;
      align-items: center;
    }
    .section-title::before {
      content: '';
      display: inline-block;
      width: 4px;
      height: 20px;
      background: var(--primary);
      margin-right: 0.8rem;
      border-radius: 2px;
    }
    /* 历史记录 */
    .history-wrap {
      max-height: 250px;
      overflow-y: auto;
      border: 1px solid var(--light-gray);
      border-radius: var(--radius);
    }
    .history-wrap::-webkit-scrollbar {
      width: 6px;
    }
    .history-wrap::-webkit-scrollbar-track {
      background: var(--light);
      border-radius: 3px;
    }
    .history-wrap::-webkit-scrollbar-thumb {
      background: var(--primary);
      border-radius: 3px;
    }
  </style>
</head>
<body>
  <div class="container">
    <!-- 头部区域 -->
    <header class="header">
      <h1 class="header__title">CNN 手写数字识别系统</h1>
      <p class="header__subtitle">基于PyTorch构建的深度学习识别系统，支持图片上传和在线手写绘制两种方式，实时展示识别结果、置信度和概率分布</p>
      <div class="status-bar">
        <span class="status-chip" id="serviceChip">检测服务状态中...</span>
        <span class="status-chip">模型准确率 99.64%</span>
        <span class="status-chip">支持 0-9 数字识别</span>
      </div>
    </header>

    <!-- 主工作区 -->
    <main class="workspace">
      <!-- 输入区域 -->
      <section class="card">
        <div class="card__header">
          <h2 class="card__title">输入区域</h2>
          <div class="tabs">
            <button class="tab active" data-target="upload-panel">上传图片</button>
            <button class="tab" data-target="draw-panel">手写绘制</button>
          </div>
        </div>
        <div class="card__body">
          <!-- 上传面板 -->
          <div id="upload-panel" class="panel active">
            <div class="upload-area">
              <h3 class="upload-area__title">上传手写数字图片</h3>
              <p class="upload-area__desc">建议使用白底黑字的数字图片，系统会自动裁剪、居中并标准化处理</p>
              <input type="file" id="fileInput" accept="image/*" />
              <img id="preview" alt="图片预览" />
            </div>
            <div class="btn-group">
              <button class="btn btn--primary" id="predictUploadBtn">识别上传图片</button>
            </div>
          </div>

          <!-- 绘制面板 -->
          <div id="draw-panel" class="panel">
            <div class="draw-area">
              <canvas id="canvas" width="300" height="300"></canvas>
              <div>
                <p class="draw__tips">在画布上书写 0-9 之间的单个数字，系统会将笔迹转换为 28x28 的模型输入格式，通过CNN模型进行识别。建议使用粗笔迹书写，确保数字清晰可见。</p>
                <div class="btn-group">
                  <button class="btn btn--primary" id="predictCanvasBtn">识别手写数字</button>
                  <button class="btn btn--secondary" id="clearCanvasBtn">清空画布</button>
                </div>
              </div>
            </div>
          </div>

          <!-- 错误提示 -->
          <div class="error" id="errorBox"></div>
        </div>
      </section>

      <!-- 结果区域 -->
      <aside class="card">
        <div class="card__header result__header">
          <h2 class="card__title">识别结果</h2>
        </div>
        <div class="card__body">
          <!-- 核心结果 -->
          <div class="result__stage">
            <div class="confidence-ring" id="confidenceRing" style="--score:0">
              <div class="metric" id="predictionValue">-</div>
            </div>
            <div>
              <p class="result__caption">当前识别结果</p>
              <p class="confidence-text" id="confidenceValue">置信度: -</p>
              <p class="mini-note">识别完成后，下方图表展示模型对每个数字类别的置信度分布</p>
            </div>
          </div>

          <!-- Top3 结果 -->
          <h3 class="section-title">Top 3 识别结果</h3>
          <table class="table" id="top3Table">
            <thead>
              <tr>
                <th>排名</th>
                <th>数字</th>
                <th>置信度</th>
              </tr>
            </thead>
            <tbody></tbody>
          </table>

          <!-- 概率分布 -->
          <h3 class="section-title">数字概率分布</h3>
          <div class="bars" id="probabilityBars"></div>

          <!-- 历史记录 -->
          <h3 class="section-title">识别历史</h3>
          <div class="history-wrap">
            <table class="table" id="historyTable">
              <thead>
                <tr>
                  <th>输入方式</th>
                  <th>识别结果</th>
                  <th>置信度</th>
                  <th>Top3 结果</th>
                </tr>
              </thead>
              <tbody></tbody>
            </table>
          </div>
        </div>
      </aside>
    </main>
  </div>

  <script>
    // 基础变量初始化
    const tabs = document.querySelectorAll(".tab");
    const panels = document.querySelectorAll(".panel");
    const history = [];
    const errorBox = document.getElementById("errorBox");
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    const preview = document.getElementById("preview");
    const fileInput = document.getElementById("fileInput");
    const uploadBtn = document.getElementById("predictUploadBtn");
    const canvasBtn = document.getElementById("predictCanvasBtn");
    
    // 初始化画布
    function resetCanvas() {
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.lineWidth = 20;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.strokeStyle = "#1e293b";
    }
    resetCanvas();
    
    // 画布绘制逻辑
    let drawing = false;
    function pointFromEvent(event) {
      const rect = canvas.getBoundingClientRect();
      const touch = event.touches ? event.touches[0] : event;
      return {
        x: (touch.clientX - rect.left) * canvas.width / rect.width,
        y: (touch.clientY - rect.top) * canvas.height / rect.height
      };
    }
    function startDraw(event) {
      drawing = true;
      const p = pointFromEvent(event);
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      event.preventDefault();
    }
    function draw(event) {
      if (!drawing) return;
      const p = pointFromEvent(event);
      ctx.lineTo(p.x, p.y);
      ctx.stroke();
      event.preventDefault();
    }
    function endDraw() {
      drawing = false;
    }
    
    // 绑定画布事件
    canvas.addEventListener("mousedown", startDraw);
    canvas.addEventListener("mousemove", draw);
    window.addEventListener("mouseup", endDraw);
    canvas.addEventListener("touchstart", startDraw, { passive: false });
    canvas.addEventListener("touchmove", draw, { passive: false });
    window.addEventListener("touchend", endDraw);
    
    // 标签页切换
    tabs.forEach(tab => {
      tab.addEventListener("click", () => {
        tabs.forEach(x => x.classList.remove("active"));
        panels.forEach(x => x.classList.remove("active"));
        tab.classList.add("active");
        document.getElementById(tab.dataset.target).classList.add("active");
        errorBox.textContent = "";
      });
    });
    
    // 文件预览
    fileInput.addEventListener("change", () => {
      const file = fileInput.files[0];
      if (!file) {
        preview.style.display = "none";
        return;
      }
      preview.src = URL.createObjectURL(file);
      preview.style.display = "block";
    });
    
    // 检查服务状态
    async function refreshHealth() {
      try {
        const response = await fetch("/health");
        const data = await response.json();
        document.getElementById("serviceChip").textContent = `服务正常 | ${data.device}`;
      } catch {
        document.getElementById("serviceChip").textContent = "服务状态异常";
      }
    }
    refreshHealth();
    
    // 渲染历史记录
    function renderHistory() {
      const body = document.querySelector("#historyTable tbody");
      body.innerHTML = "";
      history.forEach(item => {
        const row = document.createElement("tr");
        row.innerHTML = `
          <td>${item.mode}</td>
          <td>${item.prediction}</td>
          <td>${item.confidence}</td>
          <td>${item.top3}</td>
        `;
        body.appendChild(row);
      });
    }
    
    // 渲染识别结果
    function renderResult(data, mode) {
      errorBox.textContent = "";
      const confidence = data.confidence * 100;
      
      // 更新核心结果
      document.getElementById("predictionValue").textContent = data.prediction;
      document.getElementById("confidenceValue").textContent = `置信度: ${confidence.toFixed(2)}%`;
      document.getElementById("confidenceRing").style.setProperty("--score", confidence.toFixed(2));
      
      // 更新Top3表格
      const top3Body = document.querySelector("#top3Table tbody");
      top3Body.innerHTML = "";
      data.top3.forEach((item, index) => {
        const row = document.createElement("tr");
        row.innerHTML = `
          <td>第 ${index + 1} 名</td>
          <td>${item.digit}</td>
          <td>${(item.probability * 100).toFixed(2)}%</td>
        `;
        top3Body.appendChild(row);
      });
      
      // 更新概率分布条
      const bars = document.getElementById("probabilityBars");
      bars.innerHTML = "";
      data.probabilities.forEach(item => {
        const row = document.createElement("div");
        row.className = "bar-row";
        row.innerHTML = `
          <div>${item.digit}</div>
          <div class="bar-track">
            <div class="bar-fill" style="width:${(item.probability * 100).toFixed(2)}%"></div>
          </div>
          <div>${(item.probability * 100).toFixed(2)}%</div>
        `;
        bars.appendChild(row);
      });
      
      // 更新历史记录
      history.unshift({
        mode: mode === "Upload" ? "上传图片" : "手写绘制",
        prediction: data.prediction,
        confidence: `${confidence.toFixed(2)}%`,
        top3: data.top3.map(x => `${x.digit}:${(x.probability * 100).toFixed(1)}%`).join(", ")
      });
      if (history.length > 10) history.pop();
      renderHistory();
    }
    
    // 处理响应JSON
    async function checkedJson(response) {
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "识别失败，请重试");
      return data;
    }
    
    // 上传图片识别
    async function predictUpload() {
      const file = fileInput.files[0];
      if (!file) {
        errorBox.textContent = "请先选择要上传的图片";
        return;
      }
      const formData = new FormData();
      formData.append("file", file);
      uploadBtn.disabled = true;
      try {
        const response = await fetch("/api/predict-upload", {
          method: "POST",
          body: formData
        });
        renderResult(await checkedJson(response), "Upload");
      } catch (err) {
        errorBox.textContent = err.message;
      } finally {
        uploadBtn.disabled = false;
      }
    }
    
    // 手写绘制识别
    async function predictCanvas() {
      canvasBtn.disabled = true;
      try {
        const dataUrl = canvas.toDataURL("image/png");
        const response = await fetch("/api/predict-canvas", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ data_url: dataUrl })
        });
        renderResult(await checkedJson(response), "Canvas");
      } catch (err) {
        errorBox.textContent = err.message;
      } finally {
        canvasBtn.disabled = false;
      }
    }
    
    // 绑定按钮事件
    uploadBtn.addEventListener("click", predictUpload);
    canvasBtn.addEventListener("click", predictCanvas);
    document.getElementById("clearCanvasBtn").addEventListener("click", () => {
      resetCanvas();
      errorBox.textContent = "";
    });
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
