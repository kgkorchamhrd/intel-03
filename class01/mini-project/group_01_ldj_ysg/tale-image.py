#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import torch
import openvino as ov
from collections import namedtuple
from pathlib import Path
import os
import gdown
import sounddevice as sd
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
import time
import threading
import sys
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler

# OpenAI API 호출을 위한 클래스
from openai import OpenAI

# Global variables
previous_sentence = "null"
global_whisper_pipeline = None
global_stable_diffusion_pipe = None
latest_generated_image = None  # 메모리 캐싱을 위한 전역변수 추가
background_lock = threading.Lock()  # 스레드 동기화를 위한 잠금 추가

# API 키 읽기 함수
def read_api_key(file_path="api_gpt.txt"):
    """API 키를 파일에서 읽어옵니다."""
    try:
        with open(file_path, "r") as file:
            api_key = file.read().strip()  # 공백이나 줄바꿈 제거
            return api_key
    except FileNotFoundError:
        print(f"오류: '{file_path}' 파일을 찾을 수 없습니다.")
        return None
    except Exception as e:
        print(f"API 키를 읽는 중 오류가 발생했습니다: {e}")
        return None

# ChatGPT 클래스
class ChatGPT:
    def __init__(self):
        """OpenAI API를 사용하기 위한 클라이언트를 초기화합니다."""
        api_key = read_api_key()
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            print("API 키가 없습니다.")
            self.client = None
    
    def get_response(self, prompt, model="gpt-4o-mini-2024-07-18", temperature=0.7):
        """ChatGPT에 프롬프트를 보내고 응답을 얻습니다."""
        if not self.client:
            print("OpenAI 클라이언트가 제대로 초기화되지 않았습니다.")
            return None
        
        try:
            completion = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            response = completion.choices[0].message
            content = response.content
            return content
        except Exception as e:
            print(f"오류: {e}")
            return None

# 이미지 프롬프트 생성 함수
def generate_image_prompt(book_previous_sentence, book_current_sentence):
    chat = ChatGPT()
    input_prompt = (
        "Convert the following sentence from a children's book into a detailed image generation prompt. "
        "Include descriptive details such as artistic style, setting, lighting, and mood. "
        "Take into consideration both previous sentence and current sentence. "
        "The sentences might be in Korean, but generate the image prompt in English. "
        "For example, the output should be like: 'cyberpunk cityscape like Tokyo New York with tall buildings at dusk golden hour cinematic lighting'.\n\n"
        f"Children's book sentence: previous sentence({book_previous_sentence}) , current sentence({book_current_sentence})"
    )
    return chat.get_response(input_prompt)

# ---------- Whisper 음성 인식 모델 설정 ---------- #
def load_whisper_model():
    """Whisper 음성 인식 모델을 초기화합니다."""
    global global_whisper_pipeline
    
    if global_whisper_pipeline is not None:
        return global_whisper_pipeline
    
    print("음성 인식 모델 초기화 중...")
    whisper_model_id = "openai/whisper-medium"  # 한국어 인식 성능이 더 좋은 medium 모델 사용
    
    try:
        whisper_processor = AutoProcessor.from_pretrained(whisper_model_id)
        whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(whisper_model_id)
        
        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=whisper_model,
            tokenizer=whisper_processor.tokenizer,
            feature_extractor=whisper_processor.feature_extractor,
            device=torch.device("cpu"),
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=False
        )
        
        global_whisper_pipeline = asr_pipeline
        print("음성 인식 모델 초기화 완료!")
        return asr_pipeline
    
    except Exception as e:
        print(f"음성 인식 모델 초기화 중 오류 발생: {e}")
        return None

# ---------- Stable Diffusion 이미지 생성 설정 ---------- #
def load_stable_diffusion():
    """Stable Diffusion 모델을 로드합니다."""
    global global_stable_diffusion_pipe
    
    if global_stable_diffusion_pipe is not None:
        return global_stable_diffusion_pipe
    
    print("Stable Diffusion 모델 로딩 중...")
    model_id = "runwayml/stable-diffusion-v1-5"
    
    try:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, safety_checker=None)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to("cpu")
        
        global_stable_diffusion_pipe = pipe
        print("Stable Diffusion 모델 로딩 완료!")
        return pipe
    except Exception as e:
        print(f"Stable Diffusion 모델 로딩 중 오류 발생: {e}")
        return None

latest_wall_number = 0  # 가장 최근에 생성된 파일 번호
latest_loaded_number = 0  # 가장 최근에 로드된 파일 번호

# 수정된 이미지 생성 함수# 수정된 이미지 생성 함수
def generate_image_with_sd(prompt, output_dir="generated_images"):
    """Stable Diffusion을 사용하여 이미지를 생성합니다."""
    global latest_generated_image, latest_wall_number
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 모델 로드
    pipe = load_stable_diffusion()
    if pipe is None:
        print("Stable Diffusion 모델을 로드할 수 없습니다.")
        return None, None
    
    try:
        print(f"프롬프트: '{prompt}'로 이미지 생성 중...")
        start_time = time.time()
        
        with torch.no_grad():
            image = pipe(
                prompt=prompt,
                height=512, 
                width=512,
                num_inference_steps=20,
                guidance_scale=7.5
            ).images[0]
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        # 생성된 이미지를 메모리에 저장 (RGB 형식)
        latest_generated_image = np.array(image.convert('RGB'))
        
        # 순차적 파일명과 기본 파일명 모두 저장
        latest_wall_number += 1
        
        # 순차적 파일명
        seq_filename = f"wall_{latest_wall_number}.png"
        seq_filepath = os.path.join(output_dir, seq_filename)
        
        # 기본 파일명 (wall.png)
        std_filepath = os.path.join(output_dir, "wall.png")
        
        # 이미지 저장
        def save_image_task():
            try:
                # 순차적 파일명으로 저장
                print(f"이미지 저장 중: {seq_filepath}")
                image.save(seq_filepath)
                
                # wall.png로도 저장 (덮어쓰기)
                image.save(std_filepath)
                
                print(f"이미지가 저장되었습니다: {seq_filepath} 및 {std_filepath}")
            except Exception as e:
                print(f"이미지 저장 중 오류 발생: {e}")
        
        # 별도 스레드에서 저장
        save_thread = threading.Thread(target=save_image_task, daemon=True)
        save_thread.start()
        
        print(f"이미지가 생성되었습니다! ({generation_time:.1f}초 소요)")
        return image, seq_filepath
        
    except Exception as e:
        print(f"이미지 생성 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def find_latest_wall_file(output_dir="generated_images"):
    """generated_images 디렉토리에서 가장 최신의 wall_*.png 파일을 찾습니다."""
    try:
        wall_files = []
        for filename in os.listdir(output_dir):
            if filename.startswith("wall_") and filename.endswith(".png"):
                try:
                    # 파일 번호 추출
                    file_number = int(filename.split("_")[1].split(".")[0])
                    wall_files.append((filename, file_number))
                except (IndexError, ValueError):
                    continue
        
        if wall_files:
            # 파일 번호 기준 내림차순 정렬
            wall_files.sort(key=lambda x: x[1], reverse=True)
            return wall_files[0][0], wall_files[0][1]  # (파일명, 번호) 반환
    except Exception as e:
        print(f"디렉토리 스캔 중 오류 발생: {e}")
    
    return None, 0  # 파일을 찾지 못한 경우

# ---------------------------
# Setup U²-Net for background segmentation
# ---------------------------
ModelConfig = namedtuple("ModelConfig", ["name", "url", "model", "model_args"])
from model.u2net import U2NET, U2NETP  # Ensure model/u2net.py is available

u2net_lite = ModelConfig(
    name="u2net_lite",
    url="https://drive.google.com/uc?id=1W8E4FHIlTVstfRkYmNOjbr0VDXTZm0jD",
    model=U2NETP,
    model_args=(),
)
model_config = u2net_lite
MODEL_DIR = "model"
model_path = Path(MODEL_DIR) / model_config.name / (model_config.name + ".pth")

if not model_path.exists():
    os.makedirs(model_path.parent, exist_ok=True)
    print("Downloading U²-Net model weights...")
    with open(model_path, "wb") as f:
        gdown.download(url=model_config.url, output=str(f.name), quiet=False)
    print(f"Model weights downloaded to {model_path}")

net = model_config.model(*model_config.model_args)
net.eval()
print(f"Loading model weights from: '{model_path}'")
net.load_state_dict(torch.load(model_path, map_location="cpu"))

# Convert model to OpenVINO IR
model_ir = ov.convert_model(net, example_input=torch.zeros((1, 3, 512, 512)), input=([1, 3, 512, 512]))
core = ov.Core()
compiled_model_ir = core.compile_model(model=model_ir, device_name="CPU")
input_layer_ir = compiled_model_ir.input(0)
output_layer_ir = compiled_model_ir.output(0)

def preprocess_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (512, 512))
    input_img = np.expand_dims(np.transpose(resized, (2, 0, 1)), 0).astype(np.float32)
    mean = np.array([123.675, 116.28, 103.53]).reshape(1, 3, 1, 1)
    scale = np.array([58.395, 57.12, 57.375]).reshape(1, 3, 1, 1)
    return (input_img - mean) / scale

# 수정된 배경 로드 함수
def load_background(output_dir="generated_images"):
    """배경 이미지를 로드합니다. 첨부된 코드처럼 단순한 방식 사용"""
    # 출력 디렉토리에서 wall.png 로드 시도 (기본 파일명)
    wall_path = os.path.join(output_dir, "wall.png")
    if os.path.exists(wall_path):
        bg = cv2.imread(wall_path)
        if bg is not None:
            print(f"배경 이미지 로드: '{wall_path}'")
            return bg
    
    # 루트 디렉토리에서 시도
    if os.path.exists("wall.png"):
        bg = cv2.imread("wall.png")
        if bg is not None:
            print("배경 이미지 로드: 'wall.png'")
            return bg
    
    # 대체 이미지 시도
    alt_files = ["background.jpg", "background.png"]
    for file in alt_files:
        if os.path.exists(file):
            bg = cv2.imread(file)
            if bg is not None:
                print(f"배경 이미지 로드: '{file}'")
                return bg
    
    # 파일이 없으면 단색 배경 생성
    print("배경 이미지를 찾을 수 없습니다. 단색 배경을 사용합니다.")
    return np.ones((512, 512, 3), dtype=np.uint8) * np.array([128, 64, 0], dtype=np.uint8)  # BGR 형식

# 이미지 생성 플래그
is_generating_image = False

# 수정된 오디오 녹음 및 변환 함수
def record_and_transcribe_and_generate():
    global previous_sentence, is_generating_image, latest_generated_image
    
    is_generating_image = True
    
    try:
        # 음성 인식 모델 로드
        asr_pipeline = load_whisper_model()
        if not asr_pipeline:
            print("음성 인식 모델을 로드할 수 없습니다.")
            is_generating_image = False
            return
        
        print("Recording audio for 5 seconds...")
        fs = 16000
        duration = 5  # seconds
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
        
        # 진행 상황 표시
        for i in range(duration):
            print(f"녹음 중... {i+1}/{duration}초", end="\r")
            sys.stdout.flush()
            time.sleep(1)
            
        sd.wait()
        print("\n녹음 완료!")
        
        audio = audio.squeeze()  # Convert to 1D array
        
        print("음성을 텍스트로 변환 중...")
        transcription = asr_pipeline(audio)["text"]
        print("Transcription:", transcription)
        
        # Use the transcribed text as the current sentence
        current_sentence = transcription
        
        # Generate image prompt based on previous and current sentences
        print("\nChatGPT로 이미지 생성 프롬프트 생성 중...")
        image_prompt = generate_image_prompt(previous_sentence, current_sentence)
        
        if image_prompt:
            print("\n생성된 이미지 프롬프트:")
            print(image_prompt)
            try:
                import pyperclip
                pyperclip.copy(image_prompt)
                print("(프롬프트가 클립보드에 복사되었습니다)")
            except:
                pass
                
            # 이미지 생성 및 표시
            print("\n이미지 생성 중...")
            with background_lock:  # 이미지 생성 중 잠금 사용
                image, filepath = generate_image_with_sd(image_prompt, "generated_images")
            
            # 생성된 이미지 표시
            if image:
                try:
                    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    cv2.imshow("Generated Image", cv_image)
                    cv2.waitKey(1)  # 창이 나타나도록 잠시 기다림
                except Exception as e:
                    print(f"이미지 표시 중 오류 발생: {e}")
        else:
            print("이미지 생성 프롬프트를 생성하지 못했습니다.")
            
        # Update previous_sentence for the next transcription
        previous_sentence = current_sentence
        
    except Exception as e:
        print("Error during transcription or image generation:", e)
        import traceback
        traceback.print_exc()
    
    finally:
        is_generating_image = False

# ---------------------------
# Main loop: video capture with live background replacement
# ---------------------------

def main():
    global is_generating_image, latest_generated_image
    
    print("=" * 50)
    print("음성-텍스트-이미지 변환 통합 애플리케이션")
    print("=" * 50)
    print("\n카메라를 통해 음성을 인식하고 자동으로 이미지를 생성합니다.")
    print("1. 'T' 키를 눌러 음성 녹음 (한국어 지원)")
    print("2. 음성이 텍스트로 변환됩니다")
    print("3. 텍스트가 이미지 프롬프트로 변환되고 이미지가 자동 생성됩니다")
    print("4. 'Q' 키를 눌러 프로그램 종료")
    print("=" * 50)
    
    # 프로그램 전체에서 일관되게 사용할 출력 디렉토리 정의
    output_dir = "generated_images"
    
    # 디렉토리 없으면 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 초기 배경 이미지 로드
    background_img = load_background(output_dir)
    
    last_bg_reload_time = time.time()
    audio_thread = None  # 오디오 처리 스레드 추적용
    
    # 카메라 설정
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # 좌우 반전 적용
            frame = cv2.flip(frame, 1)
            
            # 5초마다 배경 이미지 다시 로드 (이미지 생성 중이 아닐 때만)
            current_time = time.time()
            if current_time - last_bg_reload_time >= 5 and not is_generating_image:
                try:
                    # 단순한 방식으로 배경 다시 로드
                    new_bg = load_background(output_dir)
                    if new_bg is not None:
                        background_img = new_bg
                        print("배경 이미지를 새로 로드했습니다.")
                except Exception as e:
                    print(f"배경 다시 로드 오류: {e}")
                last_bg_reload_time = current_time
    
            # 프레임 세그멘테이션 처리 및 이진 마스크 생성
            try:
                input_img = preprocess_frame(frame)
                result = compiled_model_ir([input_img])[output_layer_ir]
                mask = cv2.resize(np.squeeze(result), (frame.shape[1], frame.shape[0]))
                mask = (mask > 0.5).astype(np.uint8)
        
                # 배경 크기 조정 및 원본 프레임과 결합
                bg_resized = cv2.resize(background_img, (frame.shape[1], frame.shape[0]))
                output_frame = np.where(mask[:, :, np.newaxis] == 1, frame, bg_resized)
            except Exception as e:
                print(f"프레임 처리 오류: {e}")
                # 오류 발생 시 원본 프레임 사용
                output_frame = frame
            
            # 상태 표시 
            font = cv2.FONT_HERSHEY_SIMPLEX
            if is_generating_image:
                status_text = "Processing... Please wait"
                cv2.putText(output_frame, status_text, (10, 30), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                status_text = "Press 'T' to start voice recognition"
                cv2.putText(output_frame, status_text, (10, 30), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    
            cv2.imshow("Live Background Replacement", output_frame)
    
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("프로그램을 종료합니다.")
                break
            elif key == ord("t") and not is_generating_image:
                # 현재 생성 중이 아닐 때만 새 스레드 시작
                audio_thread = threading.Thread(target=record_and_transcribe_and_generate, daemon=True)
                audio_thread.start()
    
    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다.")
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 리소스 정리
        cap.release()
        cv2.destroyAllWindows()
        print("프로그램이 종료되었습니다.")

if __name__ == "__main__":
    main()