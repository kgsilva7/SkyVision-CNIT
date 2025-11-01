import os
import cv2
import time
import json
import math
import queue
import threading
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
# ===================== CONFIG =====================
VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "0")  # "0" webcam; caminho arquivo; RTSP ex: "rtsp://user:pass@ip:554/stream"
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "yolov8n.pt")
SCORE_THR = float(os.getenv("SCORE_THR", "0.35"))
IOU_THR = float(os.getenv("IOU_THR", "0.45"))
GPS_MODE = os.getenv("GPS_MODE", "simulado")  # "simulado" | "externo"
SIM_ROUTE_HZ = float(os.getenv("SIM_ROUTE_HZ", "2.0"))  # Hz do simulador de GPS
# ===================== APP =====================
app = FastAPI(title="SkyVision API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ===================== VIDEO CAP =====================
def open_video(source: str):
    try:
        src = int(source) if source.isdigit() else source
    except Exception:
        src = source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir a fonte de vídeo: {source}")
    return cap
cap = open_video(VIDEO_SOURCE)
cap_lock = threading.Lock()
# ===================== YOLO =====================
try:
    from ultralytics import YOLO
    model = YOLO(YOLO_WEIGHTS)
except Exception as e:
    model = None
    print("[AVISO] YOLO não carregado:", e)
# ===================== GPS =====================
class GpsData(BaseModel):
    lat: float
    lon: float
    alt: Optional[float] = None
    heading: Optional[float] = None
    speed: Optional[float] = None
current_gps: Optional[GpsData] = None
def sim_gps_route_thread():
    global current_gps
    # Rota simples centrada no Porto de Santos (aprox)
    lat0, lon0 = -23.95, -46.32
    t0 = time.time()
    r = 0.0015  # ~150 m
    while True:
        t = time.time() - t0
        # percurso circular
        lat = lat0 + r * math.sin(t / 30)
        lon = lon0 + r * math.cos(t / 30)
        current_gps = GpsData(lat=lat, lon=lon, alt=25.0, heading=(t*6) % 360, speed=5.0)
        time.sleep(1.0 / SIM_ROUTE_HZ)
if GPS_MODE == "simulado":
    threading.Thread(target=sim_gps_route_thread, daemon=True).start()
@app.post("/gps")
async def post_gps(data: GpsData):
    global current_gps
    current_gps = data
    return {"ok": True}
# ===================== BROADCAST =====================
class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []
        self.lock = threading.Lock()
    async def connect(self, ws: WebSocket):
        await ws.accept()
        with self.lock:
            self.active.append(ws)
    def disconnect(self, ws: WebSocket):
        with self.lock:
            if ws in self.active:
                self.active.remove(ws)
    async def broadcast(self, message: dict):
        dead = []
        for ws in list(self.active):
            try:
                await ws.send_text(json.dumps(message))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)
manager = ConnectionManager()
# ===================== INFERÊNCIA EM THREAD =====================
frame_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=2)
last_annotated = None
def infer_loop():
    global last_annotated
    if model is None:
        print("[ERRO] Modelo YOLO não disponível. Verifique as dependências e pesos.")
        return
    while True:
        with cap_lock:
            ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue
        # Inferência
        results = model(frame, verbose=False, conf=SCORE_THR, iou=IOU_THR)
        dets = []
        annotated = frame.copy()
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                cls_id = int(b.cls[0].item())
                score = float(b.conf[0].item())
                cls_name = r.names.get(cls_id, str(cls_id))
                dets.append({
                    "bbox": [x1, y1, x2, y2],
                    "cls": cls_name,
                    "score": round(score, 3),
                })
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{cls_name} {score:.2f}"
                cv2.putText(annotated, label, (x1, max(20, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        last_annotated = annotated
        gps_payload = current_gps.dict() if current_gps else None
        payload = {
            "ts": time.time(),
            "gps": gps_payload,
            "detections": dets,
            "shape": annotated.shape[:2],
        }
        # broadcast async (via background task no loop do FastAPI seria melhor; aqui simplificado)
        import asyncio
        try:
            asyncio.run(manager.broadcast(payload))
        except RuntimeError:
            # já dentro de loop event -> ignorar
            pass
# inicia thread de inferência
threading.Thread(target=infer_loop, daemon=True).start()
# ===================== STREAM MJPEG =====================
def mjpeg_generator():
    global last_annotated
    while True:
        if last_annotated is None:
            time.sleep(0.03)
            continue
        ok, jpg = cv2.imencode('.jpg', last_annotated)
        if not ok:
            continue
        frame = jpg.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.03)  # ~30 fps máx teórico
@app.get("/video")
async def video_feed():
    return StreamingResponse(mjpeg_generator(), media_type='multipart/x-mixed-replace; boundary=frame')
# ===================== WS =====================
@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Mantém conexão viva; client pode enviar pings ou filtros
            _ = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)''
# ===================== HEALTH =====================
@app.get("/")
async def root():
    return {"name": "SkyVision API", "status": "ok"}