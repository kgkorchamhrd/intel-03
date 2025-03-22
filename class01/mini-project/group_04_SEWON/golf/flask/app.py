import os
import subprocess
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# 동영상을 저장할 폴더 (절대 경로)
SAVE_FOLDER = '/home/park/workspace/golfdb'

# test_video.py 파일의 절대 경로 (같은 폴더면 파일명만, 아니면 전체 경로)
TEST_VIDEO_SCRIPT = '/home/park/workspace/golfdb/test_video.py'

# 업로드 확장자 확인
ALLOWED_EXTENSIONS = {'mp4'}

#이미지 불러올 폴더
IMAGE_FOLDER = "/home/park/workspace/golfdb/flask/static/events"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return "❌ 업로드된 파일이 없습니다."

    file = request.files["file"]
    if file.filename == "":
        return "❌ 파일 이름이 비어있습니다."

    if file and allowed_file(file.filename):
        filename = file.filename
        # 동영상 파일을 /home/park/workspace/golfdb 에 저장
        file_path = os.path.join(SAVE_FOLDER, filename)
        file.save(file_path)

        # 업로드 후 test_video.py 실행
        try:
            # python3 test_video.py -p 저장된동영상파일명.mp4
            subprocess.run(
                ["python3", TEST_VIDEO_SCRIPT, "-p", filename], 
                check=True
            )
        except subprocess.CalledProcessError:
            return "⚠️ test_video.py 실행 중 오류가 발생했습니다!"

        # 처리 완료 후 about.html로 리다이렉트
        return redirect(url_for("about"))

    return "❌ 지원하지 않는 파일 형식입니다. (mp4만 가능)"

# @app.route("/about")
# def about():
#     return render_template("about.html")것

#이부분 부터 추가한 것
@app.route("/about")

@app.route("/about")
def about():
    """
    /home/park/workspace/golfdb/flask/static/events 폴더에서
    event_#_confidence_#.jpg 파일을 찾아
    (파일명, 이벤트번호, confidence) 형태의 리스트로 전달
    """

    # events 폴더에서 특정 패턴(event_*.jpg) 파일 리스트 가져오기
    raw_files = [f for f in os.listdir(IMAGE_FOLDER) 
                 if f.startswith("event_") and f.endswith(".jpg")]

    images_info = []
    for fname in raw_files:
        # 예: event_0_confidence_0.095.jpg → ["event", "0", "confidence", "0.095.jpg"]
        parts = fname.split("_")
        if len(parts) != 4:
            # 혹시라도 패턴이 안 맞으면 skip
            continue
        
        # 이벤트 번호 (정수)
        event_index_str = parts[1]  # "0"
        event_index = int(event_index_str)

        # confidence 값 (소수)
        conf_with_ext = parts[3]  # "0.095.jpg"
        conf_str = conf_with_ext.replace(".jpg", "")  # "0.095"
        # 필요하다면 float(conf_str)로 변환 가능. 여기서는 문자열 그대로 써도 됨
        # conf_val = float(conf_str)

        images_info.append((fname, event_index, conf_str))
    
    # event_index 기준으로 정렬(0,1,2,3,...)
    images_info.sort(key=lambda x: x[1])

    return render_template("about.html", images_info=images_info)

# def about():
#     """ /home/park/workspace/golfdb/events 폴더에서 이미지 파일을 찾아 리스트로 전달 """
    
#     # events 폴더에서 event_*.jpg 파일 리스트 가져오기
#     image_files = sorted(
#         [f for f in os.listdir(IMAGE_FOLDER) if f.startswith("event_") and f.endswith(".jpg")],
#         key=lambda x: int(x.split("_")[1].split(".")[0])  # event_0.jpg → 0, event_1.jpg → 1 순으로 정렬
#     )

#     return render_template("about.html", images=image_files)

if __name__ == "__main__":
    app.run(debug=True)

    
