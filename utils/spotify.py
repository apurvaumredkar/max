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
import time
import urllib.parse

import requests
from dotenv import load_dotenv, set_key
from fastapi import APIRouter, Body
from fastapi.responses import RedirectResponse

from utils.logging_config import get_logger

load_dotenv("secrets/.env")

log = get_logger(__name__)

# No Spotify call may block forever: this module runs in the same process as the agent
# loop, and an unanswered socket used to stall chat, TTS and the Discord gateway with it.
SPOTIFY_TIMEOUT = 10

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


_token_cache = {"access_token": None, "expires_at": 0}


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
        timeout=SPOTIFY_TIMEOUT,
    )
    response.raise_for_status()
    # Re-authenticating must invalidate the cached access token. Without this, get_access_token
    # keeps serving the pre-auth token until it expires — so the re-auth you just did to fix a
    # bad grant appears to do nothing for up to an hour.
    _token_cache["expires_at"] = 0
    refresh_token = response.json()["refresh_token"]
    set_key("secrets/.env", "SPOTIFY_REFRESH_TOKEN", refresh_token)
    os.environ["SPOTIFY_REFRESH_TOKEN"] = refresh_token
    return refresh_token


# In-process cache for the access token — the web UI polls now-playing every 5s, and
# re-running the refresh-token exchange (a round trip to Spotify's auth server) on every
# poll was the actual source of the card's lag. Access tokens are valid ~1hr; refresh
# a bit early to stay clear of the exact expiry edge.


def get_access_token():
    if _token_cache["access_token"] and time.time() < _token_cache["expires_at"]:
        return _token_cache["access_token"]
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
        timeout=SPOTIFY_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    _token_cache["access_token"] = data["access_token"]
    _token_cache["expires_at"] = time.time() + data.get("expires_in", 3600) - 60
    return _token_cache["access_token"]


def _auth_headers():
    return {"Authorization": f"Bearer {get_access_token()}"}


def list_devices():
    response = requests.get(f"{PLAYER_URL}/devices", headers=_auth_headers(), timeout=SPOTIFY_TIMEOUT)
    response.raise_for_status()
    return response.json().get("devices", [])


def available_devices():
    """
    List the Spotify devices currently available to play on, e.g. to honour "play X on my iPad".

    Each result has the device id, name, type, whether it is currently active, and whether it is
    restricted (restricted devices cannot be controlled through the API). Match the user's wording
    against the name/type, then pass that device's id to play_track.
    """
    return [
        {
            "id": device.get("id"),
            "name": device.get("name"),
            "type": device.get("type"),
            "is_active": device.get("is_active", False),
            "is_restricted": device.get("is_restricted", False),
        }
        for device in list_devices()
    ]


def transfer_playback(device_id):
    """
    Claim playback for the web player, but only if no other device already holds it.

    A real device (phone, desktop app) takes priority: if one is active, leave it alone so
    opening the page never yanks playback away from whatever Apurva is already using. Only
    when nothing is active does the web player take over — and with play=False, so claiming
    the device doesn't start music on its own.

    Stale web-player devices from earlier page loads also report as "Max", so our own device
    is identified by the device_id the SDK handed us, never by name.
    """
    try:
        devices = list_devices()
    except Exception as e:
        # Can't tell who's active — err on the side of not stealing playback.
        log.error("Failed to list Spotify devices: %s", e)
        return {"status": "skipped", "reason": "device-list-failed"}

    active = next((d for d in devices if d.get("is_active")), None)
    if active and active.get("id") != device_id:
        log.info(
            "Transfer skipped: %r is active, leaving playback there",
            active.get("name"),
        )
        return {
            "status": "skipped",
            "reason": "another-device-active",
            "active_device": active.get("name"),
        }
    if active and active.get("id") == device_id:
        log.info("Transfer skipped: web player is already the active device")
        return {"status": "already-active", "active_device": active.get("name")}

    log.info("No active device — claiming playback for the web player")
    response = requests.put(
        PLAYER_URL,
        headers=_auth_headers(),
        json={"device_ids": [device_id], "play": False},
        timeout=SPOTIFY_TIMEOUT,
    )
    if response.status_code >= 400:
        log.error(
            "Playback transfer failed (%s): %s",
            response.status_code,
            response.text[:200],
        )
    return {"status": "transferred", "status_code": response.status_code}


def resume():
    response = requests.put(f"{PLAYER_URL}/play", headers=_auth_headers(), timeout=SPOTIFY_TIMEOUT)
    return {"status_code": response.status_code}


def pause():
    response = requests.put(f"{PLAYER_URL}/pause", headers=_auth_headers(), timeout=SPOTIFY_TIMEOUT)
    return {"status_code": response.status_code}


def next_track():
    response = requests.post(f"{PLAYER_URL}/next", headers=_auth_headers(), timeout=SPOTIFY_TIMEOUT)
    return {"status_code": response.status_code}


def previous_track():
    response = requests.post(f"{PLAYER_URL}/previous", headers=_auth_headers(), timeout=SPOTIFY_TIMEOUT)
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
        timeout=SPOTIFY_TIMEOUT,
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


def play_track(uri, device_id=None):
    """
    Start playback of a specific track by its Spotify URI.

    Args:
        uri: The track's Spotify URI, e.g. "spotify:track:3JTLIzNfTYNPqOc7ZzrO4A" — get this
            from search_track first if you only have a song name.
        device_id: Optional device to play on, from available_devices. Omit to use whatever
            device is currently active.
    """
    response = requests.put(
        f"{PLAYER_URL}/play",
        headers=_auth_headers(),
        params={"device_id": device_id} if device_id else None,
        json={"uris": [uri]},
        timeout=SPOTIFY_TIMEOUT,
    )
    return {"status_code": response.status_code}


def add_to_queue(uri, device_id=None):
    """
    Add a track to the end of the current playback queue.

    Args:
        uri: The track's Spotify URI, e.g. "spotify:track:3JTLIzNfTYNPqOc7ZzrO4A" — get this
            from search_track first if you only have a song name.
        device_id: Optional device whose queue to add to, from available_devices. Omit to use
            whatever device is currently active.
    """
    params = {"uri": uri}
    if device_id:
        params["device_id"] = device_id
    response = requests.post(
        f"{PLAYER_URL}/queue",
        headers=_auth_headers(),
        params=params,
        timeout=SPOTIFY_TIMEOUT,
    )
    return {"status_code": response.status_code}


def now_playing():
    response = requests.get(f"{PLAYER_URL}/currently-playing", headers=_auth_headers(), timeout=SPOTIFY_TIMEOUT)
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
def route_now_playing():
    try:
        return now_playing()
    except Exception as e:
        log.error("Failed to fetch Spotify now-playing: %s", e)
        return {"is_playing": False, "track": None}


@router.post("/spotify/resume")
def route_resume():
    return resume()


@router.post("/spotify/pause")
def route_pause():
    return pause()


@router.post("/spotify/next")
def route_next():
    return next_track()


@router.post("/spotify/prev")
def route_prev():
    return previous_track()


@router.get("/spotify/authenticate")
def route_authenticate():
    return RedirectResponse(get_authorize_url())


@router.get("/spotify/token")
def route_token():
    return {"access_token": get_access_token()}


@router.post("/spotify/transfer")
def route_transfer(device_id: str = Body(embed=True)):
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
    parser.add_argument("--devices", action="store_true", help="List available Spotify devices")
    parser.add_argument("--device-id", metavar="ID", help="Target a specific device for --play/--queue")
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
        print(json.dumps(play_track(args.play, args.device_id)))
    elif args.queue:
        print(json.dumps(add_to_queue(args.queue, args.device_id)))
    elif args.devices:
        print(json.dumps(available_devices(), indent=2))
    else:
        parser.print_help()
