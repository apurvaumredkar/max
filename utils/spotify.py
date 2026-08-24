"""
Standalone Spotify playback helper. Not wired into the agent's tool-calling loop —
Max invokes this directly via `_execute_bash`, e.g.:

    python utils/spotify.py --resume
    python utils/spotify.py --pause
    python utils/spotify.py --next
    python utils/spotify.py --prev

The functions here are also imported directly by the FastAPI backend to power the
web UI's Spotify card (now-playing state + playback buttons).

Requires SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, and SPOTIFY_REFRESH_TOKEN in
secrets/.env (refresh token needs the user-modify-playback-state and
user-read-currently-playing scopes).
"""

import argparse
import base64
import json
import os
import urllib.parse

import requests
from dotenv import load_dotenv, set_key
from fastapi import APIRouter, Body
from fastapi.responses import RedirectResponse

load_dotenv("secrets/.env")

router = APIRouter()

TOKEN_URL = "https://accounts.spotify.com/api/token"
AUTHORIZE_URL = "https://accounts.spotify.com/authorize"
SEARCH_URL = "https://api.spotify.com/v1/search"
PLAYER_URL = "https://api.spotify.com/v1/me/player"
REDIRECT_URI = "https://spotify.apurvau.dev/callback"
SCOPES = (
    "user-modify-playback-state user-read-currently-playing "
    "streaming user-read-email user-read-private"
)


def get_authorize_url():
    params = {
        "client_id": os.getenv("SPOTIFY_CLIENT_ID"),
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPES,
    }
    return f"{AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"


def exchange_code_for_token(code):
    credentials = base64.b64encode(
        f"{os.getenv('SPOTIFY_CLIENT_ID')}:{os.getenv('SPOTIFY_CLIENT_SECRET')}".encode()
    ).decode()
    response = requests.post(
        TOKEN_URL,
        headers={"Authorization": f"Basic {credentials}"},
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": REDIRECT_URI,
        },
    )
    response.raise_for_status()
    refresh_token = response.json()["refresh_token"]
    set_key("secrets/.env", "SPOTIFY_REFRESH_TOKEN", refresh_token)
    os.environ["SPOTIFY_REFRESH_TOKEN"] = refresh_token
    return refresh_token


def get_access_token():
    credentials = base64.b64encode(
        f"{os.getenv('SPOTIFY_CLIENT_ID')}:{os.getenv('SPOTIFY_CLIENT_SECRET')}".encode()
    ).decode()
    response = requests.post(
        TOKEN_URL,
        headers={"Authorization": f"Basic {credentials}"},
        data={
            "grant_type": "refresh_token",
            "refresh_token": os.getenv("SPOTIFY_REFRESH_TOKEN"),
        },
    )
    response.raise_for_status()
    return response.json()["access_token"]


def _auth_headers():
    return {"Authorization": f"Bearer {get_access_token()}"}


def transfer_playback(device_id):
    # play=False — claim the device on page load without starting playback. Whatever was
    # already playing keeps its state; a paused session stays paused.
    response = requests.put(
        PLAYER_URL,
        headers=_auth_headers(),
        json={"device_ids": [device_id], "play": False},
    )
    return {"status_code": response.status_code}


def resume():
    response = requests.put(f"{PLAYER_URL}/play", headers=_auth_headers())
    return {"status_code": response.status_code}


def pause():
    response = requests.put(f"{PLAYER_URL}/pause", headers=_auth_headers())
    return {"status_code": response.status_code}


def next_track():
    response = requests.post(f"{PLAYER_URL}/next", headers=_auth_headers())
    return {"status_code": response.status_code}


def previous_track():
    response = requests.post(f"{PLAYER_URL}/previous", headers=_auth_headers())
    return {"status_code": response.status_code}


def search_track(query, limit=5):
    """
    Search Spotify for tracks matching a query.

    Args:
        query: Free-text search, e.g. "paper rings taylor swift".
        limit: Maximum number of results to return.
    """
    response = requests.get(
        SEARCH_URL,
        headers=_auth_headers(),
        params={"q": query, "type": "track", "limit": limit},
    )
    response.raise_for_status()
    tracks = response.json().get("tracks", {}).get("items", [])
    return [
        {
            "uri": track["uri"],
            "name": track["name"],
            "artists": [artist["name"] for artist in track.get("artists", [])],
            "album": track.get("album", {}).get("name"),
        }
        for track in tracks
    ]


def play_track(uri):
    """
    Start playback of a specific track by its Spotify URI.

    Args:
        uri: The track's Spotify URI, e.g. "spotify:track:3JTLIzNfTYNPqOc7ZzrO4A" — get this
            from search_track first if you only have a song name.
    """
    response = requests.put(
        f"{PLAYER_URL}/play",
        headers=_auth_headers(),
        json={"uris": [uri]},
    )
    return {"status_code": response.status_code}


def add_to_queue(uri):
    """
    Add a track to the end of the current playback queue.

    Args:
        uri: The track's Spotify URI, e.g. "spotify:track:3JTLIzNfTYNPqOc7ZzrO4A" — get this
            from search_track first if you only have a song name.
    """
    response = requests.post(
        f"{PLAYER_URL}/queue",
        headers=_auth_headers(),
        params={"uri": uri},
    )
    return {"status_code": response.status_code}


def now_playing():
    response = requests.get(f"{PLAYER_URL}/currently-playing", headers=_auth_headers())
    if response.status_code == 204 or not response.content:
        return {"is_playing": False, "track": None}
    response.raise_for_status()
    data = response.json()
    item = data.get("item") or {}
    return {
        "is_playing": data.get("is_playing", False),
        "track": item.get("name"),
        "artists": [artist["name"] for artist in item.get("artists", [])],
        "album_art": (item.get("album", {}).get("images") or [{}])[0].get("url"),
        "progress_ms": data.get("progress_ms"),
        "duration_ms": item.get("duration_ms"),
    }


@router.get("/spotify/now-playing")
async def route_now_playing():
    try:
        return now_playing()
    except Exception as e:
        print(f"Error fetching Spotify now-playing: {e}")
        return {"is_playing": False, "track": None}


@router.post("/spotify/resume")
async def route_resume():
    return resume()


@router.post("/spotify/pause")
async def route_pause():
    return pause()


@router.post("/spotify/next")
async def route_next():
    return next_track()


@router.post("/spotify/prev")
async def route_prev():
    return previous_track()


@router.get("/spotify/authenticate")
async def route_authenticate():
    return RedirectResponse(get_authorize_url())


@router.get("/spotify/token")
async def route_token():
    return {"access_token": get_access_token()}


@router.post("/spotify/transfer")
async def route_transfer(device_id: str = Body(embed=True)):
    return transfer_playback(device_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pause", action="store_true")
    parser.add_argument("--next", action="store_true")
    parser.add_argument("--prev", action="store_true")
    parser.add_argument("--now-playing", action="store_true")
    parser.add_argument("--search", metavar="QUERY", help="Search for a track")
    parser.add_argument("--play", metavar="URI", help="Play a track by its Spotify URI")
    parser.add_argument("--queue", metavar="URI", help="Add a track to the queue by its Spotify URI")
    args = parser.parse_args()

    if args.resume:
        print(json.dumps(resume()))
    elif args.pause:
        print(json.dumps(pause()))
    elif args.next:
        print(json.dumps(next_track()))
    elif args.prev:
        print(json.dumps(previous_track()))
    elif args.now_playing:
        print(json.dumps(now_playing()))
    elif args.search:
        print(json.dumps(search_track(args.search)))
    elif args.play:
        print(json.dumps(play_track(args.play)))
    elif args.queue:
        print(json.dumps(add_to_queue(args.queue)))
    else:
        parser.print_help()
