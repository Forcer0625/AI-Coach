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
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QTextEdit, QHBoxLayout, QSizePolicy, QFileDialog, QComboBox
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
whisper_model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")

INIT_CHAT_HISTORY = [
    {"role": "developer", "content": "你是一名專業的健身教練，擅長根據使用者的動作(會有圖片)、體態和健身等問題提供專業的健身建議，請專注於健身領域的知識，並提供正確的運動指導。當有圖片時，會協助使用者根據給予的圖像回答, 請務必嚴格按照圖片回覆使用者的請求(這是最優先的事項, 模糊的回答也可以)"},#請自行判斷是否使用圖像生成工具幫助使用者，如果需要，請在回應的結尾加上關鍵字\"請參考下面的圖片說明\", 然後生成一段給DALL-E的prompt。格式參考(假設使用者詢問如何做深蹲):\"1. 一開始基本的徒手深蹲要先把雙腳打開與肩膀同寬，腳尖向前，雙手則可放在胸前交叉交疊或是雙手握拳。\n2. 把腳底平放在地上，將重心放在雙腳上。\n3. 吸氣時將重心慢慢往後，把臀部緩緩地下後移，想像後方有一張椅子，維持個3-5秒的時間，呼氣後再慢慢地回到原來的動作。\n請參考下面的圖片說明\nprompt for DALL-E\""}
]

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
                frame = frame[:,80:560,:]
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
    
    def __init__(self, camera_thread, session_id, count):
        super().__init__()
        self.camera_thread = camera_thread
        self.session_id = session_id
        self.count = count
    
    def run(self):
        self.process_voice_input()
    
    def process_voice_input(self):
        audio_file = "./session/"+self.session_id+"/voice_input"+str(self.count)+".wav"
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
            frame = frame[:,80:560,:]
            img_path = "./session/"+self.session_id+"/captured_frame"+str(self.count)+".jpg"
            cv2.imwrite(img_path, frame)
            self.image_captured.emit(img_path)
        
        # 語音轉文字
        segments, _ = whisper_model.transcribe(audio_file)
        transcribed_text = " ".join([segment.text for segment in segments])
        self.transcription_done.emit(transcribed_text)

class AIProcessingThread(QThread):
    response_ready = pyqtSignal(bool)
    response_streaming = pyqtSignal(str)
    full_response = pyqtSignal(str)
    
    def __init__(self, chat_history, user_text, img_path=None):
        super().__init__()
        self.chat_history = chat_history
        self.user_text = user_text
        self.img_path = img_path
    
    def run(self):
        messages = list(self.chat_history)
        
        if self.img_path:
            base64_image = encode_image(self.img_path)
            messages.append({"role": "user", "content": [
                        {
                            "type": "text",
                            "text": "(請根據圖片回答)"+self.user_text
                        },
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }}]
                    })
        else:
            messages.append({"role": "user", "content": self.user_text})
        
        try:
            response = openai.chat.completions.create(
                model=GPT_MODEL_NAME,
                messages=messages,
                stream=True,
            )
            self.response_ready.emit(True)
            full_ai_text = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    partial_text = chunk.choices[0].delta.content
                    full_ai_text += partial_text
                    self.response_streaming.emit(partial_text)  # 逐步顯示
        except:
            full_ai_text = ""
        finally:
            self.full_response.emit(full_ai_text)

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

class AICoachApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.initCamera()
        self.img_path = None
        self.captured_frame_count = 0
        self.session_id = "0"
        self.load_session()
    
    def initUI(self):
        self.setWindowTitle("AI 健身教練")
        self.setGeometry(100, 100, 1200, 600)
        
        # 左側 - 對話紀錄與輸入
        self.conversation = QTextEdit(self)
        self.conversation.setReadOnly(True)
        
        self.text_input = QTextEdit(self)
        self.text_input.setPlaceholderText("輸入你的問題...")

        self.user_prefix = '[🧑 你]: '
        self.ai_prefix = '[🤖 AI 教練]: '

        # 載入對話
        # 新增 session 選擇下拉式選單 (0~10)
        self.session_selector = QComboBox(self)
        self.session_selector.addItems([str(i) for i in range(11)])  # 0~10
        self.session_selector.currentIndexChanged.connect(self.change_session)
        session_layout = QHBoxLayout()
        session_layout.addWidget(QLabel("選擇對話紀錄:"))
        session_layout.addWidget(self.session_selector)

        self.clear_button = QPushButton("🗑️ 清除對話", self)
        self.clear_button.clicked.connect(self.clear_session)
        self.save_button = QPushButton("💾 儲存對話", self)
        self.save_button.clicked.connect(self.save_session)
        self.load_button = QPushButton("📂 載入對話", self)
        self.load_button.clicked.connect(self.load_session)
        conversation = QHBoxLayout()
        conversation.addWidget(self.clear_button)
        conversation.addWidget(self.save_button)
        conversation.addWidget(self.load_button)
        
        self.send_button = QPushButton("發送", self)
        self.send_button.clicked.connect(self.process_text_input)
        
        self.voice_button = QPushButton("🎤 語音輸入", self)
        self.voice_button.clicked.connect(self.start_voice_processing)
        
        left_layout = QVBoxLayout()
        left_layout.addLayout(session_layout)
        left_layout.addLayout(conversation)
        left_layout.addWidget(QLabel("對話紀錄:"))
        left_layout.addWidget(self.conversation, 5)
        left_layout.addWidget(QLabel("輸入:"))
        left_layout.addWidget(self.text_input, 1)
        left_layout.addWidget(self.send_button)
        left_layout.addWidget(self.voice_button)
        
        # 右側 - 攝影機畫面
        self.camera_label = QLabel(self)
        self.camera_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)  # 讓 QLabel 可調整大小
        self.camera_label.setMinimumSize(300, 300)  # 設定最小大小，避免過小
        
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("即時攝影機畫面:"))
        right_layout.addWidget(self.camera_label)
        
        # 主佈局
        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, 3)
        main_layout.addLayout(right_layout, 2)
        
        self.setLayout(main_layout)
        
    def process_text_input(self):
        self.voice_button.setDisabled(True)
        self.send_button.setDisabled(True)
        user_text = self.text_input.toPlainText()
        if user_text.strip():
            self.conversation.append(self.user_prefix+f'{user_text}')
            self.text_input.clear()
            self.img_path = None
        self.process_ai_response(user_text)
        self.chat_history.append({"role": "user", "content": user_text})
    
    def initCamera(self):
        self.camera_thread = CameraThread()
        self.camera_thread.frame_update.connect(self.update_frame)
        self.camera_thread.start()
    
    def update_frame(self, qimg):
        if not self.camera_label.size().isEmpty():
            qpixmap = QPixmap.fromImage(qimg)
            qpixmap = qpixmap.scaled(self.camera_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.camera_label.setPixmap(qpixmap)
    
    def start_voice_processing(self):
        self.voice_button.setDisabled(True)
        self.send_button.setDisabled(True)
        self.voice_thread = VoiceProcessingThread(self.camera_thread, self.session_id, self.captured_frame_count)
        self.voice_thread.transcription_done.connect(self.display_transcription)
        self.voice_thread.image_captured.connect(self.display_image)
        self.voice_thread.start()
        self.voice_button.setText("🎙️ 錄音中...")
    
    def display_transcription(self, text):
        self.conversation.append(text)
        if text == "❌ 錄音超時，請再試一次":
            self.recover_voice_button()
            self.recover_send_button()
            return    
        self.process_ai_response(text)
        self.chat_history.append({"role": "user", "content": text})
    
    def process_ai_response(self, user_text):
        if user_text == "":
            self.recover_voice_button()
            self.recover_send_button()
            return
        self.captured_frame_count += 1
        self.voice_button.setText("🤖 回應中...")
        self.ai_thread = AIProcessingThread(self.chat_history, user_text, self.img_path)
        self.ai_thread.response_ready.connect(self.display_ai_response_ready)
        self.ai_thread.response_streaming.connect(self.display_ai_response)
        self.ai_thread.full_response.connect(self.display_ai_response_end)
        self.ai_thread.start()

    def display_ai_response_ready(self):
        self.conversation.append(self.ai_prefix)
    
    def display_ai_response(self, ai_text):
        cursor = self.conversation.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(ai_text)
        self.conversation.setTextCursor(cursor)
        self.conversation.ensureCursorVisible()

    def display_ai_response_end(self, full_ai_text):
        if full_ai_text == "":
            self.conversation.append(self.ai_prefix+'❌ AI 回應失敗，請重試。')
        else:
            self.chat_history.append({"role": "assistant", "content": full_ai_text})
        self.recover_voice_button()
        self.recover_send_button()
        self.img_path = None

    def recover_voice_button(self):
        self.voice_button.setDisabled(False)
        self.voice_button.setText("🎤 語音輸入")

    def recover_send_button(self):
        self.send_button.setDisabled(False)
    
    def display_image(self, img_path):
        self.img_path = img_path

        # 讀取圖片並轉換為 QPixmap
        pixmap = QPixmap(img_path)
        if pixmap.isNull():
            print("❌ 無法讀取圖片")
            return

        # 建立 QLabel 來顯示圖片
        image_label = QLabel()
        image_label.setPixmap(pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio))

        # 插入 QLabel 到對話紀錄
        self.conversation.append(self.user_prefix+'\n')
        self.conversation.insertHtml(f'<img src="{img_path}" width="300">')
    
    def closeEvent(self, event):
        self.camera_thread.stop()
        self.save_session()
        event.accept()

    def save_session(self):
        session_id = int(self.session_id)
        with open("./session/"+str(session_id)+"/chat_history.json", "w", encoding="utf-8") as fp:
            json.dump(self.chat_history, fp, indent=2, ensure_ascii=False)

        file_path = f"./session/{session_id}/conversation.html"

        with open(file_path, "w", encoding="utf-8") as file:
            file.write(self.conversation.toHtml())  # 儲存完整 HTML 格式
        
        #print(f"✅ 對話紀錄已儲存至 {file_path}")

    def load_session(self):
        self.session_id = str(self.session_selector.currentText())
        
        file_path = f"./session/{self.session_id}/conversation.html"
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                self.conversation.setHtml(file.read())
            with open("./session/"+self.session_id+"/chat_history.json", "r", encoding="utf-8") as fp:
                self.chat_history = json.load(fp)
            #print(f"📂 已載入對話紀錄: {file_path}")
        else:
            self.clear_session()
            #print(f"⚠️ 沒有找到對話紀錄，建立新的 session {self.session_id}")

    def clear_session(self):
        self.session_id = str(self.session_selector.currentText())
        self.conversation.clear()
        self.chat_history = list(INIT_CHAT_HISTORY)
        self.count = 0
        
    def change_session(self):
        self.save_session()
        self.load_session()

        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AICoachApp()
    window.show()
    sys.exit(app.exec())
