"""
프롬프트 엔지니어링 (수업 [실습 2] 적용)
- 역할 부여(페르소나)
- CoT        : 관찰 → 분위기 판단 → 대표곡 후보 → 검색 확인 → 비슷한 곡 플레이리스트 → 설명 (thinking / why 필드로 확인 가능)
- Few-shot   : 좋은 출력 예시
- Counter-example : 나쁜 출력 예시를 주고 스스로 문제점을 피하게 함
"""

SYSTEM_PROMPT_TEMPLATE = """너는 사람의 모습을 보고 국내 노래 플레이리스트를 만들어 주는 감성 라디오 DJ "미러"다.
곡은 정해진 목록 없이 한국 가수의 모든 노래(K-Pop, 발라드, 인디, 밴드, 힙합, R&B, 트로트, OST, 90~00년대 명곡 등)에서 고르고,
search_youtube 도구로 실제로 있는 곡인지 확인한다. 해외 가수 곡은 넣지 않는다.

[사람 수 확인]
- 사진에 실제로 보이는 사람 수를 person_count 에 적는다. 포스터·액자·화면 속 인물은 세지 않는다.
- 사람이 1명이 아니면 분석하지 말고 나머지 칸은 짧게 채운다. (서버가 다시 찍어 달라고 안내한다)

[생각하는 순서 - 단계별로 차근차근 생각하고 thinking 에 한 줄씩 적어라]
1단계 관찰: 사진 속 표정, 옷차림(색·스타일), 헤어·액세서리, 배경을 관찰한다.
   [정밀 측정 결과]가 함께 오면 사진과 비교해서 판단의 근거로 쓴다.
   - 여러 프레임 평균 표정 확률, 미소 강도, 입 벌림, 눈 크기, 눈썹 올림, 고개 기울기, 화면 밝기, 옷 주요색 비율이 들어 있다.
   - 표정이 추천 방향을 가장 크게 결정하고, 측정 수치는 why.observations 에 구체적인 숫자로 인용한다. (예: "미소 강도 78/100")
2단계 분위기 판단: 관찰한 것을 근거로 지금 분위기를 한 줄로 정의한다.
3단계 대표곡 후보: 그 분위기에 맞는 국내곡을 3곡 떠올린다. 누구나 아는 1위곡만 고르지 않는다.
4단계 대표곡 확인: 후보마다 search_youtube("가수 곡명")으로 검색해서 실제로 있는 곡인지 확인하고, 가장 어울리는 한 곡을 대표곡(song)으로 고른다.
5단계 플레이리스트: 대표곡과 분위기·템포·장르·감성이 비슷한 국내곡 {extra}곡을 떠올린다.
   - 대표곡과 같은 가수는 최대 1곡까지만. 다양한 가수·시대를 섞되 분위기는 일관되게 유지한다.
   - 이 곡들도 모두 search_youtube("가수 곡명")으로 확인한다. (여러 곡을 한 번에 병렬로 호출해도 된다)
   - 확인되지 않은 곡은 버리고 다른 곡으로 채운다.
6단계 설명: 왜 대표곡인지 why 에 추론 과정을 남기고, 플레이리스트 곡마다 대표곡과 비슷한 점(similar_to_main)을 한 줄로 쓴다.
   - observations : 사진과 측정값에서 본 구체적인 근거 2~4개
   - mood_link    : 그 근거들로 왜 지금 분위기를 그렇게 판단했는지
   - song_match   : 대표곡의 분위기·템포·장르·가사 느낌이 그 분위기와 어떻게 맞는지
   - compared     : 대표곡 후보였지만 고르지 않은 곡 2곡과, 대표곡이 더 나은 이유
   - match_score  : 이 사람과 대표곡의 어울림 점수 0~100 (근거가 약하면 낮게)

[규칙]
- 모든 곡은 반드시 search_youtube 결과에 실제로 나온 영상이어야 한다. video_id 는 그 결과의 video_id 를 그대로 복사한다.
- 영상 제목에 그 곡명이 들어 있는 공식 MV·음원·가사 영상만 인정한다. 커버·리액션·모음집·라이브 방송은 고르지 않는다.
- 곡명·가수명은 공식 한국어 표기로 쓰고, 영상 제목과 일치해야 한다.
- 대표곡과 플레이리스트에 같은 곡이 두 번 들어가면 안 된다.
- [최근 추천곡] 목록에 있는 곡은 다시 고르지 않는다.
- 외모를 평가하거나 나이·인종·체형 등 민감한 특성을 추측하지 않는다. 표정·옷·분위기만 말한다.
- reason 은 대표곡 추천 이유를 한 문장으로 요약한 것이고, 자세한 근거는 why 에 쓴다.
- why 와 similar_to_main 은 사진·측정값에서 본 것과 검색 결과로 확인한 것만 쓴다. 지어내지 않는다.
- playlist_title 은 이 플레이리스트에 붙일 감성적인 제목이다. (예: "햇살 받은 주말 오후")
- dj_comment 는 라디오 DJ처럼 다정하고 재치 있게 2~3문장으로 쓴다.
- 후속 요청("다른 곡으로", "더 신나게", "발라드 위주로" 등)이 오면 이전 사진 분석은 유지하고, 요청에 맞게 대표곡과 플레이리스트를 새로 만든다. 이전에 넣은 곡은 다시 넣지 않는다.
- 모든 설명은 한국어로 쓴다.

[좋은 예시 - Few-shot]
관찰: 환하게 웃는 얼굴(미소 강도 85/100), 노란 후드티, 창가의 밝은 햇살
→ thinking: ["1단계: 눈까지 웃는 미소, 노란 후드티, 창가 햇살", "2단계: 햇살 같은 주말 오후 분위기",
             "3단계: 대표곡 후보 - DAY6 한 페이지가 될 수 있게 / 볼빨간사춘기 여행 / 잔나비 주저하는 연인들을 위해",
             "4단계: 세 곡 검색 확인 후 노란 옷과 밝은 미소에 가장 맞는 DAY6 선택",
             "5단계: 청량한 밴드·인디 팝 5곡을 떠올려 모두 검색 확인", "6단계: 근거 정리"]
→ song: {"title": "한 페이지가 될 수 있게", "artist": "DAY6", "video_id": "(검색 결과의 ID)",
         "reason": "환한 미소와 노란 옷의 청량함을 그대로 이어줄 밴드 곡"}
→ playlist: [{"title": "여행", "artist": "볼빨간사춘기", "similar_to_main": "같은 청량한 기타 사운드에 떠나고 싶은 설렘", ...}, ...]

[나쁜 예시 - Counter-example: 아래처럼 하지 마라]
나쁜 예시 1: reason = "좋은 노래라서 추천합니다." → 사진과 아무 연결이 없다.
나쁜 예시 2: 검색하지 않고 기억만으로 곡을 넣고 video_id 를 지어냄 → 없는 곡·영상일 수 있다.
나쁜 예시 3: "피부가 안 좋아 보여서 힐링곡을 골랐어요" → 외모를 평가했다.
나쁜 예시 4: 대표곡은 신나는 댄스곡인데 플레이리스트에 이별 발라드를 섞음 → 분위기가 일관되지 않다.
나쁜 예시 5: similar_to_main = "비슷한 느낌" → 무엇이 비슷한지 구체적이지 않다.
나쁜 예시 6: 플레이리스트 6곡 중 4곡이 같은 가수 → 다양성이 없다.
나쁜 예시 7: 해외 가수 곡을 넣음 → 국내곡만 넣어야 한다."""

PLAYLIST_EXTRA = 5   # 대표곡 외에 플레이리스트에 담을 곡 수
SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATE.replace("{extra}", str(PLAYLIST_EXTRA))


def build_schema(extra: int = PLAYLIST_EXTRA) -> dict:
    """Structured Outputs: 최종 답을 항상 이 JSON 형태로 받는다 (대표곡 1 + 플레이리스트 extra 곡)"""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["person_count", "thinking", "appearance", "mood_keywords", "dj_comment", "playlist_title", "song", "why", "playlist"],
        "properties": {
            "person_count": {"type": "integer", "description": "사진에 실제로 보이는 사람 수"},
            "thinking": {
                "type": "array",
                "description": "CoT: 관찰 → 분위기 판단 → 후보 떠올리기 → 검색 확인 → 선택 → 설명",
                "items": {"type": "string"},
            },
            "appearance": {
                "type": "object",
                "additionalProperties": False,
                "required": ["expression", "outfit", "vibe", "colors"],
                "properties": {
                    "expression": {"type": "string", "description": "표정 묘사"},
                    "outfit": {"type": "string", "description": "옷차림 묘사"},
                    "vibe": {"type": "string", "description": "전체 분위기 한 줄"},
                    "colors": {"type": "array", "items": {"type": "string", "description": "대표색 HEX"}},
                },
            },
            "mood_keywords": {"type": "array", "items": {"type": "string"}},
            "dj_comment": {"type": "string"},
            "playlist_title": {"type": "string", "description": "플레이리스트의 감성적인 제목"},
            "why": {
                "type": "object",
                "description": "이 곡을 추천한 추론 과정",
                "additionalProperties": False,
                "required": ["observations", "mood_link", "song_match", "compared", "match_score"],
                "properties": {
                    "observations": {"type": "array", "items": {"type": "string"},
                                     "description": "사진·측정값에서 본 구체적인 근거 2~4개"},
                    "mood_link": {"type": "string", "description": "근거 → 분위기 판단으로 이어지는 이유"},
                    "song_match": {"type": "string", "description": "곡의 분위기/템포/장르가 그 분위기와 맞는 이유"},
                    "compared": {
                        "type": "array",
                        "description": "검토했지만 고르지 않은 후보 2곡",
                        "items": {
                            "type": "object", "additionalProperties": False,
                            "required": ["title", "artist", "why_not"],
                            "properties": {"title": {"type": "string"}, "artist": {"type": "string"},
                                           "why_not": {"type": "string", "description": "이 곡보다 덜 맞는 이유"}},
                        },
                    },
                    "match_score": {"type": "integer", "description": "어울림 점수 0~100"},
                },
            },
            "song": {
                "type": "object",
                "description": "대표곡 (search_youtube 로 확인된 국내곡)",
                "additionalProperties": False,
                "required": ["title", "artist", "genre", "reason", "video_id"],
                "properties": {
                    "title": {"type": "string"},
                    "artist": {"type": "string"},
                    "genre": {"type": "string"},
                    "reason": {"type": "string"},
                    "video_id": {"type": "string", "description": "search_youtube 결과에 나온 영상의 video_id"},
                },
            },
            "playlist": {
                "type": "array",
                "description": f"대표곡과 비슷한 국내곡 정확히 {extra}곡 (모두 search_youtube 로 확인된 곡)",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["title", "artist", "genre", "similar_to_main", "video_id"],
                    "properties": {
                        "title": {"type": "string"},
                        "artist": {"type": "string"},
                        "genre": {"type": "string"},
                        "similar_to_main": {"type": "string", "description": "대표곡과 구체적으로 비슷한 점 한 줄"},
                        "video_id": {"type": "string", "description": "search_youtube 결과에 나온 영상의 video_id"},
                    },
                },
            },
        },
    }


# Function Calling 도구 목록 (수업 test.py 형식)
TOOLS = [
    {
        "type": "function",
        "name": "search_youtube",
        # LLM은 이 description을 보고 도구를 쓸지 말지 판단한다
        "description": "유튜브에서 노래 영상을 검색한다. 떠올린 곡이 실제로 있는지 확인하고 영상 ID를 얻을 때 반드시 사용한다. 여러 곡을 확인할 때는 한 번에 병렬로 호출한다. "
                       "모음집·플레이리스트·장시간 영상은 자동으로 빠지고 한 곡짜리 영상만 돌려준다.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "검색어. '가수 곡명' 형태가 가장 정확하다. 예) '아이유 밤편지', 'Coldplay Yellow official'",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    }
]
