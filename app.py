import sys
import cv2
import os
import pyaudio
import wave
import speech_recognition as sr
from faster_whisper import WhisperModel
import openai
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QTextEdit, QHBoxLayout
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal
import numpy as np

os.environ["PATH"] += ";C:\\Program Files\\NVIDIA\\CUDNN\\v9.7\\bin\\12.8"
os.environ["LD_LIBRARY_PATH"] = "C:\\Program Files\\NVIDIA\\CUDNN\\v9.7\\bin\\12.8"

# OpenAI API Key (請替換為你的 API Key)
OPENAI_API_KEY = "your_openai_api_key"
openai.api_key = OPENAI_API_KEY

# 設定使用的 GPT 模型名稱（GPT-4 or GPT-4V for vision）
GPT_MODEL_NAME = "gpt-4-vision-preview"

# 初始化 Whisper 本地模型
whisper_model = WhisperModel("medium", device="cuda", compute_type="float16")

class CameraThread(QThread):
    frame_update = pyqtSignal(QImage)
    
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        self.running = True
    
    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channel = frame.shape
                qimg = QImage(frame.data, width, height, width * channel, QImage.Format.Format_RGB888)
                self.frame_update.emit(qimg)
            cv2.waitKey(1)
    
    def stop(self):
        self.running = False
        self.quit()
        self.wait()
        self.cap.release()

class VoiceProcessingThread(QThread):
    transcription_done = pyqtSignal(str)
    image_captured = pyqtSignal(str)
    
    def __init__(self, camera_thread, captured_frame_count):
        super().__init__()
        self.camera_thread = camera_thread
        self.running = True
        self.count = captured_frame_count
    
    def run(self):
        self.process_voice_input()
    
    def process_voice_input(self):
        audio_file = "./temp/voice_input.wav"
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            #print("開始錄音...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            recognizer.energy_threshold = 300
            try:
                audio_data = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                with open(audio_file, "wb") as f:
                    f.write(audio_data.get_wav_data())
                #print("錄音結束")
            except sr.WaitTimeoutError:
                #print("❌ 錄音超時，未偵測到語音")
                self.transcription_done.emit("❌ 錄音超時，請再試一次")
                return
        
        # 擷取當前攝影機影像
        ret, frame = self.camera_thread.cap.read()
        if ret:
            img_path = "./temp/captured_frame"+str(self.count)+".jpg"
            cv2.imwrite(img_path, frame)
            self.image_captured.emit(img_path)
        
        # 語音轉文字
        segments, _ = whisper_model.transcribe(audio_file)
        transcribed_text = " ".join([segment.text for segment in segments])
        self.transcription_done.emit(transcribed_text)

class AICoachApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.initCamera()
        self.img_path = None
        self.captured_frame_count = 0
        self.chat_history = [
            {"role": "system", "content": "你是一名專業的 AI 健身教練，擅長根據使用者的動作、體態和語音問題提供專業的健身建議，請專注於健身領域的知識，並提供正確的運動指導。請自行判斷是否使用圖像生成工具幫助使用者，如果需要，請在回應的結尾加上關鍵字\"請參考下面的圖片說明\", 然後生成一段給DALL-E的prompt。格式參考(假設使用者詢問如何做深蹲):\"1. 一開始基本的徒手深蹲要先把雙腳打開與肩膀同寬，腳尖向前，雙手則可放在胸前交叉交疊或是雙手握拳。\n2. 把腳底平放在地上，將重心放在雙腳上。\n3. 吸氣時將重心慢慢往後，把臀部緩緩地下後移，想像後方有一張椅子，維持個3-5秒的時間，呼氣後再慢慢地回到原來的動作。\n請參考下面的圖片說明\nprompt for DALL-E\""}
        ]
    
    def initUI(self):
        self.setWindowTitle("AI 健身教練")
        self.setGeometry(100, 100, 1200, 600)
        
        # 左側 - 對話紀錄與輸入
        self.conversation = QTextEdit(self)
        self.conversation.setReadOnly(True)
        
        self.text_input = QTextEdit(self)
        self.text_input.setPlaceholderText("輸入你的問題...")
        
        self.send_button = QPushButton("發送", self)
        self.send_button.clicked.connect(self.process_text_input)
        
        self.voice_button = QPushButton("🎤 語音輸入", self)
        self.voice_button.clicked.connect(self.start_voice_processing)
        
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("對話紀錄:"))
        left_layout.addWidget(self.conversation)
        left_layout.addWidget(QLabel("輸入:"))
        left_layout.addWidget(self.text_input)
        left_layout.addWidget(self.send_button)
        left_layout.addWidget(self.voice_button)
        
        # 右側 - 攝影機畫面
        self.camera_label = QLabel(self)
        self.camera_label.setFixedSize(500, 400)
        
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("即時攝影機畫面:"))
        right_layout.addWidget(self.camera_label)
        
        # 主佈局
        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, 2)
        main_layout.addLayout(right_layout, 3)
        
        self.setLayout(main_layout)
        
    def process_text_input(self):
        self.voice_button.setDisabled(True)
        self.send_button.setDisabled(True)
        user_text = self.text_input.toPlainText()
        if user_text.strip():
            self.conversation.append(f'🧑‍💻 你: {user_text}')
            self.text_input.clear()
            self.img_path = None
            self.get_ai_response(user_text, None)
        self.recover_send_button()
        self.recover_voice_button()
    
    def initCamera(self):
        self.camera_thread = CameraThread()
        self.camera_thread.frame_update.connect(self.update_frame)
        self.camera_thread.start()
    
    def update_frame(self, qimg):
        self.camera_label.setPixmap(QPixmap.fromImage(qimg))
    
    def start_voice_processing(self):
        self.voice_button.setDisabled(True)
        self.send_button.setDisabled(True)
        self.voice_button.setText("🎙️ 錄音中...")
        self.voice_thread = VoiceProcessingThread(self.camera_thread, self.captured_frame_count)
        self.voice_thread.transcription_done.connect(self.display_transcription)
        self.voice_thread.image_captured.connect(self.display_image)
        self.voice_thread.start()
    
    def display_transcription(self, text):
        self.conversation.append(f'🧑‍💻 你: {text}')
        if text == "❌ 錄音超時，請再試一次":
            self.recover_voice_button()
            self.recover_send_button()
            return
        else:
            self.captured_frame_count += 1
        self.get_ai_response(text, self.img_path)
        self.recover_voice_button()
        self.recover_send_button()

    def recover_voice_button(self):
        self.voice_button.setDisabled(False)
        self.voice_button.setText("🎤 語音輸入")

    def recover_send_button(self):
        self.send_button.setDisabled(False)
    
    def display_image(self, img_path):
        self.img_path = img_path
        #print(f"影像擷取完成: {img_path}")
    
    def get_ai_response(self, user_input, image_path=None):
        self.chat_history.append({"role": "user", "content": user_input})

        messages = list(self.chat_history)
        
        if image_path:
            #print("取得擷取影像...")
            messages.append({"role": "user", "content": {"type": "image_url", "image_url": f"file://{image_path}"}})
        
        try:
            response = openai.chat.completions.create(
                model=GPT_MODEL_NAME,
                messages=messages
            )
            ai_text = response["choices"][0]["message"]["content"]
        except:
            ai_text = ""
        finally:
            self.conversation.append(f'🤖 AI 教練: {ai_text}')
            self.chat_history.append({"role": "assistant", "content": ai_text})
            self.img_path = None
            #print(self.chat_history)
    
    def closeEvent(self, event):
        self.camera_thread.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AICoachApp()
    window.show()
    sys.exit(app.exec())
