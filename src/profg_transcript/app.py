from __future__ import annotations

import argparse
import base64
import copy
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Iterable, Sequence


FEED_URL = "https://feeds.megaphone.fm/profgmarkets"
PODCAST_NAME = "Prof G Markets"
USER_AGENT = "profg-transcript/0.1"
GROQ_MODEL = "whisper-large-v3"
RESEND_EMAIL_URL = "https://api.resend.com/emails"
DEFAULT_STATE_PATH = Path("sent_episodes.json")
GROQ_MAX_AUDIO_BYTES = 100 * 1024 * 1024
GROQ_TARGET_AUDIO_BYTES = 95 * 1024 * 1024
GROQ_REQUEST_TARGET_BYTES = 20 * 1024 * 1024
CHUNK_SEGMENT_SECONDS = (900, 600, 300)
TRANSCODE_BITRATES = ("128k", "96k", "64k", "48k", "32k")
REQUIRED_ENV_VARS = (
    "GROQ_API_KEY",
    "RESEND_API_KEY",
    "RESEND_FROM_EMAIL",
    "RESEND_TO_EMAIL",
)


@dataclass(frozen=True)
class Config:
    groq_api_key: str
    resend_api_key: str
    resend_from_email: str
    resend_to_email: str

    @classmethod
    def from_env(cls) -> "Config":
        missing = [name for name in REQUIRED_ENV_VARS if not os.environ.get(name)]
        if missing:
            joined = ", ".join(missing)
            raise RuntimeError(f"Missing required environment variables: {joined}")

        return cls(
            groq_api_key=os.environ["GROQ_API_KEY"],
            resend_api_key=os.environ["RESEND_API_KEY"],
            resend_from_email=os.environ["RESEND_FROM_EMAIL"],
            resend_to_email=os.environ["RESEND_TO_EMAIL"],
        )


@dataclass(frozen=True)
class Episode:
    guid: str
    title: str
    publication_datetime: datetime
    description: str
    audio_url: str

    @property
    def publication_date(self) -> date:
        return self.publication_datetime.date()


@dataclass(frozen=True)
class RunResult:
    status: str
    episode_guid: str | None = None
    email_id: str | None = None
    state_changed: bool = False


Transcriber = Callable[[Episode, Config], str]
EmailSender = Callable[[Episode, str, Config], str]


class _HTMLToTextParser(HTMLParser):
    block_tags = {"address", "article", "br", "div", "li", "p", "section"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in self.block_tags:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self.block_tags:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def fetch_feed_xml() -> str:
    request = urllib.request.Request(
        FEED_URL,
        headers={"User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


def parse_feed(xml_text: str) -> list[Episode]:
    root = ET.fromstring(xml_text)
    items = root.findall("./channel/item")
    episodes: list[Episode] = []

    for item in items:
        title = _required_text(item, "title")
        pub_date = _parse_pub_date(_required_text(item, "pubDate"))
        guid = _guid_text(item)
        description = description_to_plain_text(item.findtext("description") or "")
        enclosure = item.find("enclosure")
        audio_url = enclosure.attrib.get("url", "").strip() if enclosure is not None else ""
        if not audio_url:
            raise ValueError(f"Episode is missing enclosure URL: {title}")

        episodes.append(
            Episode(
                guid=guid,
                title=clean_inline_text(title),
                publication_datetime=pub_date,
                description=description,
                audio_url=audio_url,
            )
        )

    return sorted(episodes, key=lambda episode: episode.publication_datetime, reverse=True)


def _required_text(item: ET.Element, tag: str) -> str:
    value = item.findtext(tag)
    if value is None or not value.strip():
        raise ValueError(f"RSS item is missing required tag: {tag}")
    return value.strip()


def _guid_text(item: ET.Element) -> str:
    guid = item.findtext("guid")
    if guid and guid.strip():
        return guid.strip()

    enclosure = item.find("enclosure")
    if enclosure is not None and enclosure.attrib.get("url"):
        return enclosure.attrib["url"].strip()

    raise ValueError("RSS item is missing guid and enclosure URL")


def _parse_pub_date(value: str) -> datetime:
    parsed = parsedate_to_datetime(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def description_to_plain_text(description: str) -> str:
    parser = _HTMLToTextParser()
    parser.feed(html.unescape(description))
    parser.close()
    return clean_multiline_text(parser.get_text(), preserve_blank_lines=False)


def clean_inline_text(value: str) -> str:
    value = html.unescape(value).replace("\xa0", " ")
    return re.sub(r"\s+", " ", value).strip()


def clean_multiline_text(value: str, *, preserve_blank_lines: bool = True) -> str:
    value = html.unescape(value).replace("\xa0", " ")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    lines = [line.strip() for line in value.split("\n")]
    if not preserve_blank_lines:
        lines = [line for line in lines if line]
    value = "\n".join(lines).strip()
    return re.sub(r"\n{3,}", "\n\n", value).strip()


def load_state(path: Path = DEFAULT_STATE_PATH) -> dict:
    if not path.exists():
        return {"version": 1, "initialized": False, "episodes": {}}

    state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(state.get("episodes"), dict):
        raise ValueError("State file must contain an episodes object")
    state.setdefault("version", 1)
    state.setdefault("initialized", False)
    return state


def save_state(state: dict, path: Path = DEFAULT_STATE_PATH) -> None:
    path.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def transcribe_episode(episode: Episode, config: Config) -> str:
    try:
        from groq import Groq
    except ImportError as exc:
        raise RuntimeError("The groq package is required. Run: python -m pip install -e .") from exc

    client = Groq(api_key=config.groq_api_key)
    audio_url = resolve_audio_url(episode.audio_url)

    with tempfile.TemporaryDirectory(prefix="profg-transcript-") as temp_dir:
        audio_path = prepare_audio_for_groq(audio_url, Path(temp_dir))
        audio_parts = split_audio_for_groq_requests(audio_path, Path(temp_dir))
        return transcribe_audio_files(client, audio_parts)


def transcribe_audio_files(client: object, audio_paths: Sequence[Path]) -> str:
    transcripts: list[str] = []
    for index, audio_path in enumerate(audio_paths, start=1):
        print(
            f"Transcribing audio part {index}/{len(audio_paths)} "
            f"({format_bytes(audio_path.stat().st_size)})."
        )
        with audio_path.open("rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=GROQ_MODEL,
                file=audio_file,
                language="en",
                response_format="text",
                temperature=0.0,
                timeout=300,
            )

        transcripts.append(extract_transcript_text(transcription))

    return clean_multiline_text("\n\n".join(transcripts))


def extract_transcript_text(transcription: object) -> str:
    if isinstance(transcription, str):
        transcript_text = transcription
    else:
        transcript_text = getattr(transcription, "text", "")

    if not transcript_text:
        raise RuntimeError("Groq returned an empty transcript")
    return clean_multiline_text(transcript_text)


def prepare_audio_for_groq(audio_url: str, work_dir: Path) -> Path:
    original_path = work_dir / "episode-original.mp3"
    download_audio(audio_url, original_path)
    original_size = original_path.stat().st_size

    if original_size <= GROQ_REQUEST_TARGET_BYTES:
        print(f"Audio is {format_bytes(original_size)}; uploading without transcoding.")
        return original_path

    print(
        f"Audio is {format_bytes(original_size)}; transcoding below "
        f"{format_bytes(GROQ_MAX_AUDIO_BYTES)} for Groq."
    )
    compressed_path = transcode_audio_to_limit(original_path, work_dir)
    print(f"Compressed audio is {format_bytes(compressed_path.stat().st_size)}.")
    return compressed_path


def split_audio_for_groq_requests(audio_path: Path, work_dir: Path) -> list[Path]:
    audio_size = audio_path.stat().st_size
    if audio_size <= GROQ_REQUEST_TARGET_BYTES:
        return [audio_path]

    print(
        f"Audio upload is {format_bytes(audio_size)}; splitting into parts below "
        f"{format_bytes(GROQ_REQUEST_TARGET_BYTES)}."
    )
    ffmpeg_path = get_ffmpeg_path()
    last_error = ""

    for segment_seconds in CHUNK_SEGMENT_SECONDS:
        chunks_dir = work_dir / f"chunks-{segment_seconds}"
        chunks_dir.mkdir(exist_ok=True)
        output_pattern = chunks_dir / "chunk-%03d.mp3"
        command = [
            ffmpeg_path,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(audio_path),
            "-map",
            "0:a:0",
            "-c",
            "copy",
            "-f",
            "segment",
            "-segment_time",
            str(segment_seconds),
            "-reset_timestamps",
            "1",
            str(output_pattern),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            last_error = result.stderr.strip() or result.stdout.strip()
            continue

        chunks = sorted(chunks_dir.glob("chunk-*.mp3"))
        if not chunks:
            last_error = f"ffmpeg created no chunks with segment_time={segment_seconds}"
            continue

        oversized = [
            chunk for chunk in chunks if chunk.stat().st_size > GROQ_REQUEST_TARGET_BYTES
        ]
        empty = [chunk for chunk in chunks if chunk.stat().st_size == 0]
        if not oversized and not empty:
            print(f"Split audio into {len(chunks)} parts.")
            return chunks

        last_error = (
            f"segment_time={segment_seconds} produced "
            f"{len(oversized)} oversized and {len(empty)} empty chunks"
        )

    raise RuntimeError(f"Could not split audio into upload-safe chunks. {last_error}")


def download_audio(audio_url: str, output_path: Path) -> None:
    request = urllib.request.Request(
        audio_url,
        headers={"User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            with output_path.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        raise RuntimeError(f"Failed to download episode audio: {exc}") from exc

    if output_path.stat().st_size == 0:
        raise RuntimeError("Downloaded episode audio is empty")


def transcode_audio_to_limit(input_path: Path, work_dir: Path) -> Path:
    ffmpeg_path = get_ffmpeg_path()
    last_error = ""

    for bitrate in TRANSCODE_BITRATES:
        output_path = work_dir / f"episode-{bitrate}.mp3"
        command = [
            ffmpeg_path,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-b:a",
            bitrate,
            "-codec:a",
            "libmp3lame",
            str(output_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            last_error = result.stderr.strip() or result.stdout.strip()
            continue

        if output_path.stat().st_size <= GROQ_TARGET_AUDIO_BYTES:
            return output_path

        last_error = (
            f"{bitrate} output was {format_bytes(output_path.stat().st_size)}, "
            f"above target {format_bytes(GROQ_TARGET_AUDIO_BYTES)}"
        )

    raise RuntimeError(f"Could not transcode audio below Groq size limit. {last_error}")


def get_ffmpeg_path() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path
    raise RuntimeError(
        "ffmpeg is required for audio compression. Run: python -m pip install -e ."
    )


def format_bytes(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} B"
        size /= 1024
    return f"{size:.1f} GB"


def resolve_audio_url(audio_url: str) -> str:
    request = urllib.request.Request(
        audio_url,
        headers={
            "User-Agent": USER_AGENT,
            "Range": "bytes=0-0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.geturl()
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        print(
            f"Warning: could not resolve audio URL redirects; using RSS enclosure URL. {exc}",
            file=sys.stderr,
        )
        return audio_url


def send_transcript_email(episode: Episode, markdown: str, config: Config) -> str:
    payload = {
        "from": config.resend_from_email,
        "to": parse_recipient_list(config.resend_to_email),
        "subject": build_email_subject(episode),
        "text": build_email_body(episode),
        "attachments": [
            {
                "filename": build_attachment_filename(episode),
                "content": base64.b64encode(markdown.encode("utf-8")).decode("ascii"),
            }
        ],
    }
    request = urllib.request.Request(
        RESEND_EMAIL_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.resend_api_key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
            "Idempotency-Key": f"profgmkts-transcript/{episode.guid}",
        },
        method="POST",
    )
    response_text = _open_request(request, timeout=60)
    response = json.loads(response_text)
    email_id = response.get("id")
    if not email_id:
        raise RuntimeError(f"Resend response did not include an email id: {response_text}")
    return email_id


def parse_recipient_list(value: str) -> str | list[str]:
    recipients = [part.strip() for part in value.split(",") if part.strip()]
    if not recipients:
        raise RuntimeError("RESEND_TO_EMAIL must contain at least one recipient")
    if len(recipients) == 1:
        return recipients[0]
    return recipients


def build_email_subject(episode: Episode) -> str:
    return f"{episode.publication_date:%Y-%m-%d} ProfGMkts Transcript: {episode.title}"


def build_email_body(episode: Episode) -> str:
    return "\n\n".join(
        [
            f"Title: {episode.title}",
            f"Publication Date: {episode.publication_date:%Y-%m-%d}",
            f"Short Description: {episode.description}",
            "The Markdown transcript is attached.",
        ]
    )


def build_markdown_attachment(episode: Episode, transcript: str) -> str:
    return "\n\n".join(
        [
            f"# {episode.title}",
            f"**Podcast:** {PODCAST_NAME}\n"
            f"**Publication Date:** {episode.publication_date:%Y-%m-%d}\n"
            f"**Episode ID:** {episode.guid}\n"
            f"**Audio URL:** {episode.audio_url}",
            "## Short Description",
            episode.description,
            "## Transcript",
            clean_multiline_text(transcript),
        ]
    ).rstrip() + "\n"


def build_attachment_filename(episode: Episode) -> str:
    return f"{episode.publication_date:%Y-%m-%d}-profgmkts-transcript-{slugify(episode.title)}.md"


def slugify(value: str, max_length: int = 80) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized)
    normalized = normalized.strip("-")
    if len(normalized) > max_length:
        normalized = normalized[:max_length].rstrip("-")
    return normalized or "episode"


def _open_request(request: urllib.request.Request, timeout: int) -> str:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {request.full_url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Request failed for {request.full_url}: {exc.reason}") from exc


def run(
    *,
    state_path: Path = DEFAULT_STATE_PATH,
    dry_run: bool = False,
    feed_xml: str | None = None,
    config: Config | None = None,
    transcriber: Transcriber = transcribe_episode,
    email_sender: EmailSender = send_transcript_email,
    now: Callable[[], datetime] | None = None,
) -> RunResult:
    now = now or (lambda: datetime.now(UTC))
    state = load_state(state_path)
    episodes = parse_feed(feed_xml if feed_xml is not None else fetch_feed_xml())
    if not episodes:
        raise RuntimeError("No episodes found in RSS feed")

    latest = episodes[0]
    if latest.guid in state["episodes"]:
        print(f"No new episode. Latest already recorded: {latest.title}")
        return RunResult(status="noop", episode_guid=latest.guid, state_changed=False)

    if dry_run:
        print(f"Dry run: would transcribe and email: {latest.title}")
        return RunResult(status="dry-run", episode_guid=latest.guid, state_changed=False)

    load_dotenv()
    config = config or Config.from_env()
    transcript = transcriber(latest, config)
    markdown = build_markdown_attachment(latest, transcript)
    email_id = email_sender(latest, markdown, config)

    updated_state = copy.deepcopy(state)
    was_initialized = bool(updated_state.get("initialized"))
    updated_state["version"] = 1
    updated_state["initialized"] = True
    updated_state["episodes"][latest.guid] = _episode_state_entry(
        latest,
        status="sent",
        timestamp_key="sent_at",
        timestamp=now(),
        email_id=email_id,
    )

    if not was_initialized:
        for episode in episodes[1:]:
            updated_state["episodes"].setdefault(
                episode.guid,
                _episode_state_entry(
                    episode,
                    status="skipped_initial_backfill",
                    timestamp_key="skipped_at",
                    timestamp=now(),
                ),
            )

    save_state(updated_state, state_path)
    print(f"Sent transcript email for: {latest.title}")
    return RunResult(
        status="sent",
        episode_guid=latest.guid,
        email_id=email_id,
        state_changed=True,
    )


def _episode_state_entry(
    episode: Episode,
    *,
    status: str,
    timestamp_key: str,
    timestamp: datetime,
    email_id: str | None = None,
) -> dict[str, str]:
    entry = {
        "status": status,
        "title": episode.title,
        "publication_date": f"{episode.publication_date:%Y-%m-%d}",
        "guid": episode.guid,
        "audio_url": episode.audio_url,
        timestamp_key: _utc_isoformat(timestamp),
    }
    if email_id:
        entry["resend_email_id"] = email_id
    return entry


def _utc_isoformat(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prof G Markets transcript emailer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Check the feed and send the latest unsent transcript")
    run_parser.add_argument("--dry-run", action="store_true", help="Do not transcribe, email, or update state")
    run_parser.add_argument(
        "--state-path",
        default=str(DEFAULT_STATE_PATH),
        help="Path to sent episode state JSON",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "run":
        try:
            result = run(state_path=Path(args.state_path), dry_run=args.dry_run)
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        return 0 if result.status in {"sent", "noop", "dry-run"} else 1

    parser.error(f"Unknown command: {args.command}")
    return 2
