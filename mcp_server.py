"""
MCP 서버 (수업 server.py 형식)
app.py 에서 AI가 쓰는 유튜브 노래 검색 도구를 MCP 도구로도 공개한다.
→ Claude Desktop, Cursor 같은 다른 AI 앱도 이 도구를 쓸 수 있게 됨

실행 테스트:  python mcp_client.py
"""
from fastmcp import FastMCP

from youtube import find_videos
from youtube import search_youtube as yt_search
from youtube import youtube_link as make_link

mcp = FastMCP("music mirror")


@mcp.tool
def search_youtube(query: str, n: int = 5) -> list[dict]:
    """유튜브에서 노래 영상을 검색할 때 사용합니다. '가수 곡명'으로 검색하면 실제로 있는 곡인지 확인할 수 있습니다."""
    return yt_search(query, n=n)


@mcp.tool
def youtube_link(artist: str, title: str) -> str:
    """가수와 곡 제목이 주어지면 유튜브 검색 링크를 만들어 줄 때 사용합니다."""
    return make_link(artist, title)


@mcp.tool
def find_youtube_video(artist: str, title: str) -> str:
    """가수와 곡 제목으로 실제 재생 가능한 유튜브 영상 주소를 찾을 때 사용합니다."""
    ids = find_videos(artist, title, n=1)
    return f"https://www.youtube.com/watch?v={ids[0]}" if ids else "영상을 찾지 못했습니다."


if __name__ == "__main__":
    mcp.run()
