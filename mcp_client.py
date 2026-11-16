"""
MCP 클라이언트 테스트 (수업 client.py 형식)
도구 목록 확인 → search_youtube → find_youtube_video 순서로 체인 호출
"""
import asyncio
import json
import os

from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport

# 서버 프로세스에 현재 환경변수를 넘겨준다 (회사·학교 프록시 환경에서도 인터넷 검색이 되도록)
client = Client(PythonStdioTransport("mcp_server.py", env=dict(os.environ)))


def to_list(result):
    """도구 결과를 파이썬 리스트로 (버전에 따라 data 또는 content 로 옴)"""
    if getattr(result, "data", None) is not None:
        return result.data
    return json.loads(result.content[0].text) if result.content else []


async def main():
    async with client:
        tools = await client.list_tools()
        print("사용 가능한 도구:", [t.name for t in tools])

        # 1. 노래 검색
        result = await client.call_tool("search_youtube", {"query": "DAY6 한 페이지가 될 수 있게", "n": 3})
        videos = to_list(result)
        if not videos:
            print("검색 결과가 없어요 (인터넷 연결 확인)")
        for v in videos:
            print(f"- {v['title']} | {v['channel']} | {v['length']} | {v['views']}")

        # 2. 바로 재생할 영상 주소 찾기
        link = await client.call_tool("find_youtube_video", {"artist": "DAY6", "title": "한 페이지가 될 수 있게"})
        print("재생 주소:", link.content[0].text)


asyncio.run(main())
