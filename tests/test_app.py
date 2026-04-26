from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from profg_transcript.app import (
    Config,
    build_attachment_filename,
    build_email_body,
    build_email_subject,
    build_markdown_attachment,
    extract_transcript_text,
    parse_feed,
    run,
    send_transcript_email,
    split_audio_for_groq_requests,
    transcribe_audio_files,
)


SAMPLE_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Prof G Markets</title>
    <item>
      <title>Latest &amp; Greatest Episode</title>
      <description><![CDATA[<p>Latest intro.</p><p>More details &amp; context.</p>]]></description>
      <pubDate>Fri, 24 Apr 2026 08:15:00 -0000</pubDate>
      <guid isPermaLink="false"><![CDATA[latest-guid]]></guid>
      <enclosure url="https://example.com/latest.mp3" length="0" type="audio/mpeg"/>
    </item>
    <item>
      <title>Older Episode</title>
      <description>Older description.</description>
      <pubDate>Thu, 23 Apr 2026 08:15:00 -0000</pubDate>
      <guid isPermaLink="false">older-guid</guid>
      <enclosure url="https://example.com/older.mp3" length="0" type="audio/mpeg"/>
    </item>
  </channel>
</rss>
"""


class ProfGTranscriptTests(unittest.TestCase):
    def test_parse_feed_orders_latest_first_and_cleans_description(self) -> None:
        episodes = parse_feed(SAMPLE_FEED)

        self.assertEqual([episode.guid for episode in episodes], ["latest-guid", "older-guid"])
        self.assertEqual(episodes[0].title, "Latest & Greatest Episode")
        self.assertEqual(episodes[0].description, "Latest intro.\nMore details & context.")
        self.assertEqual(episodes[0].audio_url, "https://example.com/latest.mp3")

    def test_first_run_sends_latest_and_skips_older_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "sent_episodes.json"
            calls: dict[str, str] = {}

            def transcriber(episode, config):
                calls["transcribed"] = episode.guid
                return "Transcript text."

            def sender(episode, markdown, config):
                calls["sent"] = episode.guid
                calls["markdown"] = markdown
                return "resend-email-id"

            result = run(
                state_path=state_path,
                feed_xml=SAMPLE_FEED,
                config=fake_config(),
                transcriber=transcriber,
                email_sender=sender,
                now=lambda: datetime(2026, 4, 25, 12, 30, tzinfo=UTC),
            )

            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(result.status, "sent")
            self.assertEqual(calls["transcribed"], "latest-guid")
            self.assertEqual(calls["sent"], "latest-guid")
            self.assertTrue(state["initialized"])
            self.assertEqual(state["episodes"]["latest-guid"]["status"], "sent")
            self.assertEqual(state["episodes"]["latest-guid"]["resend_email_id"], "resend-email-id")
            self.assertEqual(state["episodes"]["older-guid"]["status"], "skipped_initial_backfill")

    def test_noop_when_latest_episode_already_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "sent_episodes.json"
            state_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "initialized": True,
                        "episodes": {"latest-guid": {"status": "sent"}},
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            before = state_path.read_text(encoding="utf-8")

            def transcriber(episode, config):
                raise AssertionError("Transcriber should not be called")

            def sender(episode, markdown, config):
                raise AssertionError("Email sender should not be called")

            result = run(
                state_path=state_path,
                feed_xml=SAMPLE_FEED,
                config=fake_config(),
                transcriber=transcriber,
                email_sender=sender,
            )

            self.assertEqual(result.status, "noop")
            self.assertEqual(state_path.read_text(encoding="utf-8"), before)

    def test_email_and_markdown_content(self) -> None:
        episode = parse_feed(SAMPLE_FEED)[0]
        subject = build_email_subject(episode)
        body = build_email_body(episode)
        markdown = build_markdown_attachment(episode, "Line one.\n\nLine two.")
        filename = build_attachment_filename(episode)

        self.assertEqual(
            subject,
            "2026-04-24 ProfGMkts Transcript: Latest & Greatest Episode",
        )
        self.assertIn("Title: Latest & Greatest Episode", body)
        self.assertIn("Publication Date: 2026-04-24", body)
        self.assertIn("Short Description: Latest intro.\nMore details & context.", body)
        self.assertEqual(
            filename,
            "2026-04-24-profgmkts-transcript-latest-greatest-episode.md",
        )
        self.assertIn("# Latest & Greatest Episode", markdown)
        self.assertIn("**Podcast:** Prof G Markets", markdown)
        self.assertIn("**Publication Date:** 2026-04-24", markdown)
        self.assertIn("**Episode ID:** latest-guid", markdown)
        self.assertIn("**Audio URL:** https://example.com/latest.mp3", markdown)
        self.assertIn("## Short Description", markdown)
        self.assertIn("## Transcript", markdown)
        self.assertIn("Line one.\n\nLine two.", markdown)

    def test_failure_does_not_update_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "sent_episodes.json"

            def transcriber(episode, config):
                return "Transcript text."

            def sender(episode, markdown, config):
                raise RuntimeError("Resend failed")

            with self.assertRaises(RuntimeError):
                run(
                    state_path=state_path,
                    feed_xml=SAMPLE_FEED,
                    config=fake_config(),
                    transcriber=transcriber,
                    email_sender=sender,
                )

            self.assertFalse(state_path.exists())

    def test_small_audio_file_is_single_upload_part(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            audio_path = work_dir / "small.mp3"
            audio_path.write_bytes(b"fake audio")

            self.assertEqual(split_audio_for_groq_requests(audio_path, work_dir), [audio_path])

    def test_transcribe_audio_files_combines_parts_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            first = work_dir / "chunk-001.mp3"
            second = work_dir / "chunk-002.mp3"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            client = FakeGroqClient(["First transcript.", "Second transcript."])

            transcript = transcribe_audio_files(client, [first, second])

            self.assertEqual(transcript, "First transcript.\n\nSecond transcript.")
            self.assertEqual(client.calls, ["chunk-001.mp3", "chunk-002.mp3"])

    def test_empty_transcript_response_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            extract_transcript_text(SimpleNamespace(text=""))

    def test_resend_request_includes_user_agent(self) -> None:
        episode = parse_feed(SAMPLE_FEED)[0]
        with patch("profg_transcript.app._open_request", return_value='{"id":"email-id"}') as opener:
            email_id = send_transcript_email(episode, "# Transcript", fake_config())

        request = opener.call_args.args[0]
        self.assertEqual(email_id, "email-id")
        self.assertEqual(request.get_header("User-agent"), "profg-transcript/0.1")
        self.assertEqual(request.get_header("Content-type"), "application/json")


def fake_config() -> Config:
    return Config(
        groq_api_key="groq-key",
        resend_api_key="resend-key",
        resend_from_email="from@example.com",
        resend_to_email="to@example.com",
    )


class FakeGroqClient:
    def __init__(self, responses: list[str]) -> None:
        self.calls: list[str] = []
        self.audio = SimpleNamespace(
            transcriptions=SimpleNamespace(create=self._create),
        )
        self._responses = responses

    def _create(self, **kwargs):
        self.calls.append(Path(kwargs["file"].name).name)
        return self._responses.pop(0)


if __name__ == "__main__":
    unittest.main()
