import sys
import cv2
import os
import pyaudio
import wave
import json
import speech_recognition as sr
from faster_whisper import WhisperModel
import openai
import base64
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QTextEdit, QHBoxLayout
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal
import numpy as np

os.environ["PATH"] += ";C:\\Program Files\\NVIDIA\\CUDNN\\v9.7\\bin\\12.8"
os.environ["LD_LIBRARY_PATH"] = "C:\\Program Files\\NVIDIA\\CUDNN\\v9.7\\bin\\12.8"

# OpenAI API Key (請替換為你的 API Key)
OPENAI_API_KEY = "your_api_key"
openai.api_key = OPENAI_API_KEY

# 設定使用的 GPT 模型名稱（GPT-4 or GPT-4V for vision）
GPT_MODEL_NAME = "gpt-4o-mini"
DALL_E_MODEL = "dall-e-2"

# 初始化 Whisper 本地模型
whisper_model = WhisperModel("medium", device="cuda", compute_type="float16")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

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
                frame = frame[80:560]
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
        audio_file = "./temp/voice_input"+str(self.count)+".wav"
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            recognizer.energy_threshold = 300
            try:
                audio_data = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                with open(audio_file, "wb") as f:
                    f.write(audio_data.get_wav_data())
            except sr.WaitTimeoutError:
                self.transcription_done.emit("❌ 錄音超時，請再試一次")
                return
        
        # 擷取當前攝影機影像
        ret, frame = self.camera_thread.cap.read()
        if ret:
            img_path = "./temp/captured_frame"+str(self.count)+".jpg"
            compressed_img = frame[80:560] # crop to square
            cv2.imwrite(img_path, compressed_img)
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
            {"role": "developer", "content": "你是一名專業的健身教練，擅長根據使用者的動作(會有圖片)、體態和健身等問題提供專業的健身建議，請專注於健身領域的知識，並提供正確的運動指導。當有圖片時，會協助使用者根據給予的圖像回答, 請務必嚴格按照圖片回覆使用者的請求(這是最優先的事項, 模糊的回答也可以)"},#請自行判斷是否使用圖像生成工具幫助使用者，如果需要，請在回應的結尾加上關鍵字\"請參考下面的圖片說明\", 然後生成一段給DALL-E的prompt。格式參考(假設使用者詢問如何做深蹲):\"1. 一開始基本的徒手深蹲要先把雙腳打開與肩膀同寬，腳尖向前，雙手則可放在胸前交叉交疊或是雙手握拳。\n2. 把腳底平放在地上，將重心放在雙腳上。\n3. 吸氣時將重心慢慢往後，把臀部緩緩地下後移，想像後方有一張椅子，維持個3-5秒的時間，呼氣後再慢慢地回到原來的動作。\n請參考下面的圖片說明\nprompt for DALL-E\""}
            # {'role': 'user', 'content': '如何深蹲'},
            # {'role': 'assistant', 'content': "深蹲是一個很好的全身性訓練動作，能幫助你增強腿部和核心的力量。以下是正確的深蹲步驟和要點：\n\n1. **站位**：雙腳與肩同寬，腳尖微微外展，保持穩定的站姿。\n\n2. **姿勢準備**：將胸部挺起，肩膀放鬆，眼睛直視前方。確保背部保持自然曲線，避免駝背。\n\n3. **開始下蹲**：\n   - 同時彎曲膝蓋和髖關節，臀部向後移動，讓屁股像坐在椅子上 一樣。\n   - 保持膝蓋的方向與腳尖一致，膝蓋不應超過腳尖。\n\n4. **下蹲深度**：根據個人靈活性，可以選擇下蹲到大腿與地面平行，或者更低，視你的舒適度和靈活性而定。\n\n5. **上升**：用腳跟推地，保持核心收緊，直立起來，回到起始位置。\n\n6. **呼吸**：在下蹲時吸氣，上升時呼氣。\n\n### 注意事項：\n- 確保膝蓋不內扣，這樣可以減少受傷的風險。\n- 如果感覺又疲累或不穩定，考慮減少負荷或使用支撐物。\n- 如果有任何不適或疼痛，應立即停止並尋求專業建議。\n\n進行熱身和拉伸也是很重要的，以防止受傷並提高運動表現。祝你訓練愉快！"}
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
        self.recover_voice_button()
        self.recover_send_button()
    
    def initCamera(self):
        self.camera_thread = CameraThread()
        self.camera_thread.frame_update.connect(self.update_frame)
        self.camera_thread.start()
    
    def update_frame(self, qimg):
        self.camera_label.setPixmap(QPixmap.fromImage(qimg))
    
    def start_voice_processing(self):
        self.voice_button.setDisabled(True)
        self.send_button.setDisabled(True)
        self.voice_thread = VoiceProcessingThread(self.camera_thread, self.captured_frame_count)
        self.voice_thread.transcription_done.connect(self.display_transcription)
        self.voice_thread.image_captured.connect(self.display_image)
        self.voice_thread.start()
        self.voice_button.setText("🎙️ 錄音中...")
    
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
    
    def get_ai_response(self, user_input, image_path=None):
        self.voice_button.setText("🤖 回應中...")

        messages = list(self.chat_history)
        
        if image_path:
            base64_image = encode_image(image_path)
            messages.append({"role": "user", "content": [
                        {
                            "type": "text",
                            "text": "(請根據圖片回答)"+user_input
                        },
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }}]
                    })
        else:
            messages.append({"role": "user", "content": user_input})
        
        try:
            response = openai.chat.completions.create(
                model=GPT_MODEL_NAME,
                messages=messages
            )
            ai_text = response.choices[0].message.content
            self.chat_history.append({"role": "user", "content": user_input})
            self.chat_history.append({"role": "assistant", "content": ai_text})
        except:
            ai_text = ""
        finally:
            self.conversation.append(f'🤖 AI 教練: {ai_text}')
            self.img_path = None

        # if "的圖片" in ai_text and "參考" in ai_text:
        #     ai_text = 0
        #     image_response = openai.Image.create(
        #         #model=DALL_E_MODEL,
        #         prompt=ai_text,
        #         n=1,
        #         size="256x256"
        #     )
        #     image_url = image_response["data"][0]["url"]
        #     self.image_generated.emit(image_url)
    
    def closeEvent(self, event):
        self.camera_thread.stop()
        with open("chat history(only text).json", "w", encoding="utf-8") as fp:
            json.dump(self.chat_history, fp, indent=2, ensure_ascii=False) 
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AICoachApp()
    window.show()
    sys.exit(app.exec())
