"""
Music Mirror - Flask 백엔드

[브라우저 index.html] ──사진 + 정밀 측정값──▶ [app.py] ──▶ OpenAI Responses API (gpt-4o-mini, 멀티모달)
                                                 │            │ function_call: search_youtube (여러 번)
                                                 │◀───────────┘
                                                 ├─▶ youtube.py (유튜브 검색 → 실제 영상 정보)
                                                 └─▶ 결과 JSON / TTS(mp3) / 앨범커버(이미지)

실행:  python app.py  →  http://localhost:5000
"""
import base64
import colorsys
import io
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request
from PIL import Image

from prompts import PLAYLIST_EXTRA, SYSTEM_PROMPT, TOOLS, build_schema
from youtube import find_videos, is_recent, norm, search_youtube, split_title, youtube_link

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")


def check_env_file():
    """.env 를 못 찾는 흔한 실수를 실행할 때 알려준다"""
    if os.path.exists(ENV_PATH):
        return
    for wrong in (".env.txt", "env", "env.txt"):
        if os.path.exists(os.path.join(BASE_DIR, wrong)):
            print(f"⚠ '{wrong}' 파일이 있어요. 이름을 정확히 '.env' 로 바꿔주세요 (확장자 .txt 없이).")
            return
    print(f"ℹ .env 파일이 없어요. 페이지의 ⚙️ AI 설정에서 키를 넣거나, 이 위치에 .env 를 만들어주세요: {ENV_PATH}")


def valid_key(key: str) -> bool:
    """예시 문구(sk-여기에_키_입력)나 한글이 섞인 키는 쓸 수 없다"""
    if not key:
        return False
    if not key.isascii() or not key.startswith("sk-"):
        print("⚠ OPENAI_API_KEY 가 올바르지 않아요 (예시 문구가 그대로 있거나 한글이 섞여 있음). 데모 모드로 실행합니다.")
        return False
    return True


check_env_file()
load_dotenv(ENV_PATH, override=True)        # app.py 와 같은 폴더의 .env 를 읽음
API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1-mini")

# ===== 튜닝 값 (수업에서처럼 여기서 바꿔 가며 실험) =====
TEMPERATURE = 0.9      # 높을수록 다양한 곡을 떠올림
YT_RESULTS = 5         # search_youtube 한 번에 돌려줄 영상 수
MAX_TOOL_ROUNDS = 8    # 도구 호출 반복 최대 횟수 (플레이리스트 6곡 확인)
RECENT_LIMIT = 60      # 최근 추천곡 몇 곡까지 다시 안 나오게 할지
TTS_VOICE = "coral"    # DJ 목소리 (coral, nova, alloy, shimmer, onyx, sage ...)

# 페이지의 'AI 설정'에서 키·모델을 바꿀 수 있어서 전역 상태로 관리
MODEL_CHOICES = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1", "gpt-5-mini", "gpt-5"]
client = None
key_source = None              # "env" 또는 "page"


def set_api_key(key: str, source: str):
    global client, key_source
    if valid_key(key):
        from openai import OpenAI
        client = OpenAI(api_key=key)
        key_source = source
    else:
        client, key_source = None, None


set_api_key(API_KEY, "env")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    return jsonify({"ai": client is not None, "model": MODEL, "models": MODEL_CHOICES, "key_source": key_source})


@app.route("/api/config", methods=["POST"])
def config():
    """페이지에서 API 키 / 모델 변경 (키는 서버 메모리에만 두고 파일에 저장하지 않음)"""
    global MODEL
    body = request.get_json(force=True)
    key = (body.get("api_key") or "").strip()
    model = (body.get("model") or "").strip()

    if key:
        if not key.isascii():
            return jsonify({"error": "키에 한글이나 특수문자가 섞여 있어요. sk- 로 시작하는 키를 그대로 붙여넣어 주세요."}), 400
        from openai import OpenAI
        try:
            OpenAI(api_key=key).models.retrieve(model or MODEL)   # 키·모델이 맞는지 가볍게 확인
        except Exception as e:  # noqa: BLE001
            return jsonify({"error": f"키 또는 모델 확인 실패: {e}"}), 400
        set_api_key(key, "page")
    elif body.get("clear_key"):
        set_api_key(os.getenv("OPENAI_API_KEY", "").strip(), "env")

    if model:
        if client is not None and not key:
            try:
                client.models.retrieve(model)
            except Exception as e:  # noqa: BLE001
                return jsonify({"error": f"모델 '{model}' 을 쓸 수 없어요: {e}"}), 400
        MODEL = model
    return status()


# ------------------------------------------------------------------
# Function Calling: LLM이 요청한 도구를 실제로 실행
# ------------------------------------------------------------------
def run_tool(name: str, args: dict, opts: dict, trace: list) -> dict:
    print(f">>> {name} tool 호출: {args}")            # 수업 코드처럼 호출 증거 출력
    if name == "search_youtube":
        found = search_youtube(args["query"], n=YT_RESULTS + 2)
        results = [v for v in found if not is_recent(v, opts["recent"])][:YT_RESULTS]   # 최근 추천곡 영상은 뺌
        for v in results:
            opts["seen_videos"][v["video_id"]] = v     # 나중에 AI가 고른 video_id 가 진짜 검색 결과인지 확인용
        trace.append({"tool": name, "args": args, "count": len(results),
                      "titles": [f"{v['title']} ({v['channel']})" for v in results]})
        return {"query": args["query"], "results": results}
    return {"error": f"알 수 없는 도구: {name}"}


@app.route("/api/recommend", methods=["POST"])
def recommend():
    body = request.get_json(force=True)
    image = body.get("image")                    # 첫 분석 때만 사진이 옴
    text = body.get("text", "")
    prev_id = body.get("previous_response_id")   # 멀티턴: 이전 대화 id
    face = body.get("face")                      # 브라우저 정밀 측정 결과
    recent = body.get("exclude", [])[-RECENT_LIMIT:]   # 최근 추천한 곡 (브라우저가 기억해서 보내줌)
    opts = {"recent": recent, "seen_videos": {}}

    # 1차 검사: 브라우저에서 센 얼굴 수
    if image and face and face.get("count") not in (None, 1):
        return jsonify({"error": f"얼굴이 {face['count']}개 보여요. 한 명만 화면에 나와 주세요."}), 400

    if client is None:                           # 데모 모드
        if not image:
            return jsonify({"error": "후속 대화는 AI 모드에서만 가능해요"}), 400
        res = demo_analyze(image, face, recent)
        return (jsonify(res), 502) if "error" in res else jsonify(res)

    if image and face and face.get("scores"):
        text += "\n\n[정밀 측정 결과]\n" + format_measure(face)
    text += f"\n\n[조건] 한국 가수의 국내곡만. 대표곡 1곡 + 비슷한 곡 {PLAYLIST_EXTRA}곡, 모두 search_youtube 로 확인."
    if recent:
        text += "\n\n[최근 추천곡 - 다시 고르지 말 것] " + ", ".join(f"{x.get('artist', '')} - {x.get('title', '')}" for x in recent)

    # 멀티모달 입력 (Responses API 형식)
    if image:
        user_input = [{"role": "user", "content": [
            {"type": "input_text", "text": text},
            {"type": "input_image", "image_url": image, "detail": "low"},
        ]}]
    else:
        user_input = [{"role": "user", "content": text}]

    common = dict(
        model=MODEL,
        instructions=SYSTEM_PROMPT,      # instructions는 매 호출마다 다시 넣어야 함
        tools=TOOLS,
        text={"format": {"type": "json_schema", "name": "music_recommendation",
                         "schema": build_schema(), "strict": True}},
        store=True,
    )
    if not MODEL.startswith(("gpt-5", "o1", "o3", "o4")):   # 추론 모델은 temperature 미지원
        common["temperature"] = TEMPERATURE

    trace, usage_in, usage_out = [], 0, 0
    t0 = time.time()
    try:
        first = {"previous_response_id": prev_id} if prev_id else {}
        response = client.responses.create(input=user_input, **first, **common)
        # tool call 루프: LLM이 function_call 을 주면 실행해서 결과를 돌려줌
        for _ in range(MAX_TOOL_ROUNDS):
            usage_in += response.usage.input_tokens
            usage_out += response.usage.output_tokens
            calls = [item for item in response.output if item.type == "function_call"]
            if not calls:
                break
            # 여러 곡을 한 번에 확인하므로 검색을 동시에 실행 (6곡이면 6배 빨라짐)
            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(lambda c: run_tool(c.name, json.loads(c.arguments), opts, trace), calls))
            outputs = [{
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(result, ensure_ascii=False),
            } for call, result in zip(calls, results)]
            # previous_response_id 로 이어서 보내면 function_call 기록을 다시 안 보내도 됨
            response = client.responses.create(input=outputs, previous_response_id=response.id, **common)
        data = json.loads(response.output_text)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"OpenAI 호출 실패: {e}"}), 500

    # 2차 검사: AI가 본 사람 수 (사진, 포스터 속 인물은 제외하라고 지시함)
    if image and data.get("person_count", 1) != 1:
        n = data["person_count"]
        msg = "사람이 안 보여요. 얼굴이 잘 나오게 다시 찍어 주세요." if n == 0 else f"사진에 {n}명이 보여요. 한 명만 화면에 나와 주세요."
        return jsonify({"error": msg}), 400

    # 환각 검증: 모든 곡이 실제 유튜브 검색 결과에 있었는지 서버에서 한 번 더 확인
    verify_playlist(data, opts["seen_videos"])

    return jsonify({
        "data": data,
        "trace": trace,
        "response_id": response.id,
        "usage": {"input": usage_in, "output": usage_out},
        "elapsed": round(time.time() - t0, 2),
    })


EXPRESSION_KO = {"happy": "행복", "sad": "슬픔", "angry": "화남", "surprised": "놀람",
                 "neutral": "무표정", "fearful": "긴장", "disgusted": "언짢음"}


def format_face(face: dict) -> str:
    scores = sorted(face["scores"].items(), key=lambda x: x[1], reverse=True)
    return ", ".join(f"{EXPRESSION_KO.get(k, k)} {v * 100:.0f}%" for k, v in scores[:3])


def _level(v: float) -> str:
    return "매우 낮음" if v < 20 else "낮음" if v < 40 else "보통" if v < 60 else "높음" if v < 80 else "매우 높음"


def format_measure(face: dict) -> str:
    """브라우저가 여러 프레임에서 측정한 수치를 LLM이 읽기 좋은 문장으로 바꾼다"""
    lines = [f"- 평균 표정 ({face.get('samples', 1)}프레임): {format_face(face)}"]
    if face.get("stability") is not None:
        lines.append(f"- 표정 안정도: {face['stability']}/100 ({_level(face['stability'])}, 낮으면 표정이 자주 바뀜)")
    m = face.get("metrics") or {}
    names = [("smile", "미소 강도"), ("mouth_open", "입 벌림"), ("eye_open", "눈 크기(뜬 정도)"), ("brow_raise", "눈썹 올림")]
    for key, label in names:
        if key in m:
            lines.append(f"- {label}: {m[key]}/100 ({_level(m[key])})")
    if "head_tilt" in m:
        lines.append(f"- 고개 기울기: {m['head_tilt']}° ({'거의 똑바로' if abs(m['head_tilt']) < 5 else '갸웃'})")
    if "head_turn" in m:
        t = m["head_turn"]
        lines.append(f"- 고개 방향: {'정면' if abs(t) < 25 else '옆으로 돌림'} ({t})")
    sc = face.get("scene") or {}
    if "brightness" in sc:
        lines.append(f"- 화면 밝기: {sc['brightness']}/100 ({_level(sc['brightness'])})")
    if sc.get("colors"):
        lines.append("- 옷 주요색: " + ", ".join(f"{c['hex']} {c['pct']}%" for c in sc["colors"]))
    return "\n".join(lines)


def verify_track(track: dict, seen_videos: dict, main: bool = False):
    """AI가 고른 video_id 가 실제 검색 결과였고, 영상 제목에 곡명이 들어 있어야 통과"""
    info = seen_videos.get(track.get("video_id", ""))
    track["verified"] = bool(info) and norm(track["title"]) in norm(info["title"])
    track["youtube"] = youtube_link(track["artist"], track["title"])
    if info:
        track["video_title"], track["channel"] = info["title"], info["channel"]
    if track["verified"] and not main:
        track["video_ids"] = [track["video_id"]]
        return
    # 대표곡이거나 확인 안 된 곡은 실제 검색으로 재생 후보를 채움 (첫 영상이 퍼가기 금지면 다음 걸로)
    others = [v for v in find_videos(track["artist"], track["title"], n=3) if v != track.get("video_id")]
    track["video_ids"] = ([track["video_id"]] if info else []) + others


def verify_playlist(data: dict, seen_videos: dict):
    """대표곡 + 플레이리스트 전체 검증, 중복 제거"""
    song = data["song"]
    keys = {norm(song["title"]) + norm(song["artist"])}
    playlist = []
    for t in data.get("playlist", []):
        k = norm(t["title"]) + norm(t["artist"])
        if k not in keys:
            keys.add(k)
            playlist.append(t)
    data["playlist"] = playlist
    jobs = [(song, True)] + [(t, False) for t in playlist]
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(lambda j: verify_track(j[0], seen_videos, j[1]), jobs))
    for t in [song] + playlist:
        t["region"] = "국내"


# ------------------------------------------------------------------
# TTS
# ------------------------------------------------------------------
@app.route("/api/tts", methods=["POST"])
def tts():
    if client is None:
        return jsonify({"error": "TTS는 AI 모드에서만 가능해요"}), 400
    body = request.get_json(force=True)
    try:
        speech = client.audio.speech.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=body.get("text", ""),
            instructions="밤 라디오 DJ처럼 따뜻하고 다정하게, 약간 미소 띤 톤으로 한국어를 말해줘.",
        )
        return Response(speech.content, mimetype="audio/mpeg")
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"TTS 실패: {e}"}), 500


# ------------------------------------------------------------------
# 이미지 생성: 오늘의 무드 앨범 커버
# ------------------------------------------------------------------
@app.route("/api/cover", methods=["POST"])
def cover():
    if client is None:
        return jsonify({"error": "이미지 생성은 AI 모드에서만 가능해요"}), 400
    body = request.get_json(force=True)
    vibe = body.get("vibe", "")
    keywords = ", ".join(body.get("keywords", []))
    colors = ", ".join(body.get("colors", []))
    prompt = (f"'{vibe}' 분위기의 플레이리스트 앨범 커버 아트. 키워드: {keywords}. "
              f"주요 색상: {colors}. 사람 얼굴 없이 추상적이고 감성적인 일러스트, 글자 넣지 않기, 정사각형.")
    try:
        res = client.images.generate(model=IMAGE_MODEL, prompt=prompt, size="1024x1024", quality="low")
        return jsonify({"image": "data:image/png;base64," + res.data[0].b64_json})
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"이미지 생성 실패: {e}"}), 500


# ------------------------------------------------------------------
# 데모 모드 (API 키 없을 때): 표정 + 옷 색 → 분위기 검색어 → 유튜브 검색
# ------------------------------------------------------------------
TONES = {
    "warm": ("따뜻한 컬러의 에너지 넘치는 무드", "신나는"),
    "cool": ("시원한 컬러의 청량하고 감성적인 무드", "청량한"),
    "dark": ("어두운 컬러의 시크한 무드", "시크한"),
    "bright": ("밝은 파스텔 톤의 산뜻한 무드", "산뜻한"),
}
# 표정별 검색어 (앞에서부터 시도, 결과가 부족하면 다음 검색어도 합침)
EXPRESSION_QUERY = {
    "happy":     ["신나는 노래 MV", "기분 좋은 노래 MV", "댄스곡 MV"],
    "sad":       ["슬픈 발라드 MV", "이별 노래 MV"],
    "angry":     ["강렬한 노래 MV", "힙합 MV"],
    "surprised": ["강렬한 퍼포먼스 MV", "댄스곡 MV"],
    "neutral":   ["잔잔한 노래 MV", "힐링 노래 MV", "어쿠스틱 노래 MV"],
    "fearful":   ["위로 노래 MV", "힐링 노래 MV"],
    "disgusted": ["기분 전환 노래 MV", "신나는 노래 MV", "댄스곡 MV"],
}


def _track(v: dict, note: str) -> dict:
    artist, title = split_title(v["title"], v["channel"])
    return {"title": title, "artist": artist, "region": "국내", "genre": "-", "video_id": v["video_id"],
            "verified": True, "youtube": youtube_link(artist, title), "video_ids": [v["video_id"]],
            "video_title": v["title"], "channel": v["channel"], "similar_to_main": note}


def demo_analyze(data_url: str, face: dict | None, recent: list) -> dict:
    raw = base64.b64decode(data_url.split(",", 1)[1])
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = img.size
    region = img.crop((int(w * 0.3), int(h * 0.68), int(w * 0.7), int(h * 0.98))).resize((20, 20))
    px = list(region.getdata())
    r, g, b = (sum(p[i] for p in px) / len(px) for i in range(3))
    hue, sat, val = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    hue *= 360
    tone = "dark" if val < 0.3 else "bright" if (sat < 0.2 and val > 0.65) else \
           "warm" if (hue < 50 or hue > 320) else "cool"
    vibe, adj = TONES[tone]
    hex_color = "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))

    expr = (face or {}).get("expression") or "neutral"
    queries = EXPRESSION_QUERY.get(expr, EXPRESSION_QUERY["neutral"])
    found, used, ids = [], [], set()
    for q in queries:                       # 플레이리스트를 채울 만큼 모일 때까지 검색
        for v in search_youtube(q, n=10):
            key = norm(split_title(v["title"], v["channel"])[1])      # 같은 곡의 다른 영상은 한 번만
            if v["video_id"] in ids or key in ids or is_recent(v, recent) or re.search(r"#|::", v["title"]):
                continue
            ids.update({v["video_id"], key})
            found.append(v)
        used.append(q)
        if len(found) >= PLAYLIST_EXTRA + 3:
            break
    if not found:
        return {"error": "유튜브 검색 결과가 없어요. 인터넷 연결을 확인하거나 잠시 후 다시 해 주세요."}
    # 공식 MV 영상을 앞쪽으로
    found.sort(key=lambda v: not re.search(r"\bM/?V\b|official|뮤직비디오", v["title"], re.I))
    pool = found[:PLAYLIST_EXTRA + 3]
    random.shuffle(pool)
    main_v, rest = pool[0], pool[1:PLAYLIST_EXTRA + 1]
    song = _track(main_v, "")
    song["reason"] = f"'{used[0]}' 검색 결과에서 고른 곡"
    song["video_ids"] = [main_v["video_id"]]
    playlist = [_track(v, f"같은 '{used[0]}' 분위기 검색 결과") for v in rest]
    return {
        "data": {
            "person_count": 1,
            "playlist_title": f"{EXPRESSION_KO.get(expr)} 무드 플레이리스트",
            "thinking": [f"1단계: 표정 {EXPRESSION_KO.get(expr)}, 옷 평균색 {hex_color} ('{tone}' 톤)",
                         f"2단계: {', '.join(repr(q) for q in used)} 로 유튜브 검색",
                         "3단계: 공식 MV 위주로 대표곡 1곡 + 나머지 곡으로 플레이리스트 구성"],
            "appearance": {"expression": format_face(face) if face and face.get("scores") else "표정 인식 안 됨",
                           "outfit": f"옷 대표색 {hex_color}", "vibe": vibe, "colors": [hex_color]},
            "mood_keywords": [EXPRESSION_KO.get(expr, ""), adj.rstrip("한")],
            "dj_comment": "AI 설정에서 API 키를 넣으면 AI가 표정과 옷차림을 직접 보고 골라줄 거예요. 지금은 표정과 옷 색만 보고 골랐어요!",
            "song": song,
            "playlist": playlist,
            "why": {
                "observations": (format_measure(face).replace("- ", "").split("\n")[:5] if face and face.get("scores")
                                 else ["표정 인식 결과 없음"]) + [f"옷 영역 평균색 {hex_color} → '{tone}' 톤"],
                "mood_link": f"표정({EXPRESSION_KO.get(expr)})을 기준으로 '{used[0]}' 로 검색했어요.",
                "song_match": f"검색 결과에 나온 '{main_v['title']}' 영상이에요. (데모 모드는 AI 판단 없이 검색어로만 골라요)",
                "compared": [],
                "match_score": 50,
            },
        },
        "trace": [{"tool": "search_youtube", "args": {"query": q}, "count": len(found),
                   "titles": [f"{v['title']} ({v['channel']})" for v in found]} for q in used],
        "response_id": None,
        "usage": None,
        "elapsed": 0,
    }


if __name__ == "__main__":
    print("=" * 56)
    print(" Music Mirror  →  http://localhost:5000")
    print(" 모드 :", f"AI ({MODEL})" if client else "데모 (키 없음 → 페이지의 ⚙️ AI 설정에서 입력 가능)")
    print(f" 추천 : 국내곡 플레이리스트 (대표곡 1 + 비슷한 곡 {PLAYLIST_EXTRA}) · 유튜브 검색")
    print("=" * 56)
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
