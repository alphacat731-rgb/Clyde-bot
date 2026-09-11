# Clyde Bot

Clyde is a Discord-native helper bot built from the Token project but kept as a separate application and memory database.

## What Clyde can do

- AI chat through Groq with Gemini fallback.
- Image understanding through Groq's vision model with Gemini vision fallback.
- GIF/WebP conversion to a representative PNG frame for vision.
- Persistent SQLite conversation/user memory.
- `/clyde` AI command.
- `/help`, `/ping`, `/serverinfo`, `/userinfo`, `/avatar`.
- `/poll` for quick reaction polls.
- `/clear` with Manage Messages permission checks.
- `/timeout` and `/untimeout` with moderation and role-hierarchy checks.
- `/kick` and `/ban` with moderation and role-hierarchy checks.
- `/announce` for a simple embed announcement with Manage Server permission.
- `/forget` to clear channel conversation memory.

## Personality

Clyde uses she/her pronouns and behaves as a capable, witty, Discord-focused helper rather than a Token-style character. Her wording is intentionally more natural and candid than a corporate support bot, while Discord actions remain permission-aware.

## Environment

Create `.env` from `.env.example`:

```env
DISCORD_TOKEN=your_clyde_discord_bot_token
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
GROQ_VISION_MODEL=qwen/qwen3.6-27b
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.7-flash
CLYDE_MEMORY_DB=clyde_memory.db
```

You can reuse the same Groq and Gemini API keys used by another bot. `DISCORD_TOKEN` must belong to Clyde's own Discord application.

## Raspberry Pi setup

```bash
git clone https://github.com/alphacat731-rgb/Clyde-bot.git
cd Clyde-bot
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
nano .env
python clyde.py
```

For 24/7 startup, install `clyde-bot.service.example` as `/etc/systemd/system/clyde-bot.service`, replace `YOUR_LINUX_USERNAME`, then run:

```bash
sudo systemctl daemon-reload
sudo systemctl enable clyde-bot.service
sudo systemctl start clyde-bot.service
sudo journalctl -u clyde-bot.service -f
```

## Discord setup

Create a separate Discord application/bot for Clyde. Enable **Message Content Intent** because Clyde uses message content for mention/DM conversations. Give the bot only the permissions needed for the features you enable. Moderation commands also depend on Discord's normal permission and role hierarchy rules.

Do not commit `.env` or the SQLite database to GitHub.
