# Prof G Markets Transcript Emailer

Small standalone Python project that checks the Prof G Markets RSS feed, transcribes the latest unsent episode with Groq, and sends the transcript as a Markdown attachment with Resend.

## Local Setup

Use Python 3.11 or newer.

The project depends on `imageio-ffmpeg`, which provides the ffmpeg binary used to compress long episodes for `whisper-large-v3`. Large compressed files are split into smaller upload parts before transcription to avoid request-size limits.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

Create `.env` with:

```text
GROQ_API_KEY=
RESEND_API_KEY=
RESEND_FROM_EMAIL=
RESEND_TO_EMAIL=
```

Run a dry check without sending email or updating state:

```powershell
.\.venv\Scripts\python.exe -m profg_transcript run --dry-run
```

Run for real:

```powershell
.\.venv\Scripts\python.exe -m profg_transcript run
```

## GitHub Actions

The workflow in `.github/workflows/profg-transcript.yml` runs daily at 8:30 AM ET and can also be triggered manually. Add these repository secrets before enabling it:

- `GROQ_API_KEY`
- `RESEND_API_KEY`
- `RESEND_FROM_EMAIL`
- `RESEND_TO_EMAIL`

The workflow commits `sent_episodes.json` back to the repo when a new episode is processed.

## First Run Behavior

The first real run sends only the latest episode currently visible in the RSS feed, then records older visible episodes as `skipped_initial_backfill` so they are not emailed later.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```
