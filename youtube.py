"""
유튜브 검색 (API 키 필요 없음)
- search_youtube : 검색 결과 페이지의 ytInitialData 에서 영상 정보(제목, 채널, 길이, 조회수, ID)를 뽑는다
                   → AI가 노래 DB 밖의 곡도 찾을 수 있게 해주는 도구 (Function Calling)
- find_videos    : '가수 제목' 으로 영상 ID 후보만 빠르게 가져온다 (재생용)
LLM이 영상 ID를 지어내면 없는 영상일 수 있으므로(환각), 영상 정보는 항상 실제 검색 결과에서만 가져온다.
"""
import json
import re
from urllib.parse import quote_plus

import requests


def norm(s: str) -> str:
    """비교용: 띄어쓰기·대소문자·기호 무시"""
    return re.sub(r"[^0-9a-z가-힣]", "", (s or "").lower())


def youtube_link(artist: str, title: str) -> str:
    """'유튜브에서 보기' 버튼용 검색 링크"""
    return "https://www.youtube.com/results?search_query=" + quote_plus(f"{artist} {title}")


def is_recent(video: dict, recent: list[dict]) -> bool:
    """최근 추천한 곡의 영상인지 (제목에 곡명이, 제목/채널에 가수명이 들어 있으면 같은 곡으로 봄)"""
    vt, vc = norm(video["title"]), norm(video["channel"])
    for r in recent:
        t, a = norm(r.get("title", "")), norm(r.get("artist", ""))
        if len(t) >= 2 and t in vt and (not a or a in vt or a in vc):
            return True
    return False


def split_title(video_title: str, channel: str) -> tuple[str, str]:
    """'가수 - 곡명 (Official MV)' 형태의 영상 제목에서 가수/곡명을 대략 나눈다 (데모 모드용)"""
    t = video_title
    for _ in range(4):   # 괄호 안 괄호까지 안쪽부터 지움: (OST (Part.7))
        t = re.sub(r"\s*[\[(【][^\[\]()【】]*[\])】]\s*", " ", t)
    t = re.sub(r"[()\[\]【】]", " ", t)
    t = re.sub(r"\b(official\s*(music\s*)?video|M/?V|lyric\s*video|audio)\b", "", t, flags=re.I).strip(" |-")
    q = re.search(r"['\"“‘「『](.+?)['\"”’」』]", t)          # 가수 '곡명' 형태
    if q:
        return (t[:q.start()].strip(" -_|") or channel), q.group(1).strip()
    for sep in (" - ", " – ", " _ ", "-"):
        if sep in t:
            a, b = t.split(sep, 1)
            return a.strip(), b.strip(" '\"")
    return channel, t

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
}

# 노래 한 곡이 아닐 가능성이 높은 영상 (모음집, 플레이리스트, 장시간 영상 등)
NOT_A_SONG = re.compile(r"playlist|플레이리스트|플리|모음|연속\s*듣기|1\s*hour|1시간|hours|광고없는|노래방|karaoke|reaction|리액션",
                        re.IGNORECASE)


def _fetch(query: str) -> str:
    res = requests.get("https://www.youtube.com/results", params={"search_query": query},
                       headers=HEADERS, cookies={"CONSENT": "YES+1"}, timeout=8)
    return res.text


def _seconds(length: str) -> int:
    try:
        parts = [int(x) for x in length.split(":")]
    except ValueError:
        return 0
    sec = 0
    for p in parts:
        sec = sec * 60 + p
    return sec


def _walk(node, out: list):
    """ytInitialData 안의 videoRenderer 를 모두 찾는다"""
    if isinstance(node, dict):
        if "videoRenderer" in node:
            out.append(node["videoRenderer"])
        for v in node.values():
            _walk(v, out)
    elif isinstance(node, list):
        for v in node:
            _walk(v, out)


def search_youtube(query: str, n: int = 8, songs_only: bool = True) -> list[dict]:
    """유튜브 검색 결과를 [{video_id, title, channel, length, views}] 로 돌려준다"""
    try:
        html = _fetch(query)
        m = re.search(r"var ytInitialData\s*=\s*(\{.*?\});\s*</script>", html, re.S)
        if not m:
            return []
        renderers = []
        _walk(json.loads(m.group(1)), renderers)
    except Exception as e:  # noqa: BLE001
        print("[YouTube] 검색 실패:", e)
        return []

    results, seen = [], set()
    for v in renderers:
        vid = v.get("videoId")
        if not vid or vid in seen:
            continue
        title = "".join(r.get("text", "") for r in v.get("title", {}).get("runs", []))
        channel = "".join(r.get("text", "") for r in v.get("ownerText", {}).get("runs", []))
        length = v.get("lengthText", {}).get("simpleText", "")
        views = v.get("viewCountText", {}).get("simpleText", "") or \
            "".join(r.get("text", "") for r in v.get("viewCountText", {}).get("runs", []))
        if songs_only:
            sec = _seconds(length)
            if not length or sec < 60 or sec > 9 * 60 or NOT_A_SONG.search(title):
                continue          # 라이브 방송, 쇼츠, 모음집 제외 → 노래 한 곡짜리 영상만
        seen.add(vid)
        results.append({"video_id": vid, "title": title, "channel": channel, "length": length, "views": views})
        if len(results) >= n:
            break
    return results


def find_videos(artist: str, title: str, n: int = 5) -> list[str]:
    """'가수 제목' 으로 검색해서 재생할 영상 ID 후보 n개를 돌려준다"""
    vids = [v["video_id"] for v in search_youtube(f"{artist} {title}", n=n)]
    if vids:
        return vids
    # 길이 정보가 없는 경우 등을 대비해 필터 없이 한 번 더
    try:
        ids = []
        for vid in re.findall(r'"videoId":"([\w-]{11})"', _fetch(f"{artist} {title}")):
            if vid not in ids:
                ids.append(vid)
            if len(ids) >= n:
                break
        return ids
    except Exception as e:  # noqa: BLE001
        print("[YouTube] 검색 실패:", e)
        return []
