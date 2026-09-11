import asyncio
import base64
import io
import os
import random
import re
from collections import defaultdict, deque
from datetime import timedelta

import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv
from groq import AsyncGroq
from google import genai
from google.genai import types
from PIL import Image

from memory import ClydeMemory

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.6-27b")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")
MEMORY_DB = os.getenv("CLYDE_MEMORY_DB", "clyde_memory.db")

if not DISCORD_TOKEN:
    raise RuntimeError("DISCORD_TOKEN is missing.")

groq = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
gemini = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
memory = ClydeMemory(MEMORY_DB)

CLYDE_PERSONALITY = r"""
You are Clyde, the Discord mascot and a female Discord-native helper. Use she/her for yourself.

ROLE:
- You are Clyde, a capable Discord assistant who understands Discord deeply.
- You help people use, configure, troubleshoot, and manage Discord servers.
- You are helpful first, but you have personality: witty, dry, playful, sarcastic, and occasionally chaotic.
- You are not a corporate customer-support bot.
- Never pretend a Discord action happened unless the bot actually performed it.

VOICE:
- Natural Discord speech. Avoid corporate filler.
- Mild profanity, slang, teasing, and blunt wording are okay when they fit.
- Be less sterile than a generic assistant, but stay appropriate.
- CAPS are occasional emphasis, not constant screaming.
- Text emoticons like :3 are okay sparingly.
- Do not use Unicode emoji in generated prose. Discord reactions and embeds may use emoji separately.

DISCORD EXPERTISE:
- Understand roles, permissions, channels, threads, categories, webhooks, bots, slash commands, intents, permissions hierarchy, timeouts, moderation, embeds, reactions, and polls.
- When troubleshooting permissions, name the exact Discord permission that matters.
- Remember that role hierarchy and channel overwrites can still block an otherwise correct permission.
- Never suggest bypassing permissions.

CAPABILITIES:
- You receive basic server, channel, user, role, and attachment context when relevant.
- Slash commands can perform Discord actions when their permission checks pass.
- Images are routed to a vision-capable model.
- GIFs are converted to a representative still frame for vision.
- Persistent SQLite memory stores safe conversational context.

MEMORY:
- Use supplied memory as real continuity.
- Do not invent memories.
- Remember useful names, preferences, recurring topics, and safe technical context.

VISION:
- When VISUAL INPUT is supplied, inspect the actual image data before answering.
- Base descriptions on visible evidence, not filenames or guesses.
- If something is unclear, say what is uncertain.

SAFETY:
- Do not encourage real-world violence, self-harm, dangerous behavior, or damaging real systems.
- For moderation features, obey Discord permissions and role hierarchy.
- Confirm what an action actually did instead of claiming success beforehand.

RESPONSE STYLE:
- Simple questions: short Discord-style answers.
- Troubleshooting and tutorials: useful detail without bloated essays.
- Be candid and funny when appropriate.
- The bot is Clyde, not Token. Never describe yourself as Token.
"""

HELP_TEXT = (
    "**Clyde** — your Discord helper.\n\n"
    "`/clyde` ask me anything\n"
    "`/help` show this menu\n"
    "`/ping` check latency\n"
    "`/serverinfo` server details\n"
    "`/userinfo` inspect a member\n"
    "`/avatar` show a member's avatar\n"
    "`/poll` create a poll\n"
    "`/clear` delete recent messages (Manage Messages)\n"
    "`/timeout` timeout a member\n"
    "`/untimeout` remove a timeout\n"
    "`/kick` kick a member\n"
    "`/ban` ban a member\n"
    "`/announce` send an announcement\n"
    "`/forget` clear Clyde's channel memory\n\n"
    "Mention Clyde or DM her for normal conversation."
)

FALLBACKS = [
    "one of my AI backends just tripped over a cable. try that again.",
    "backend hiccup. i'm still online though.",
    "my brain engine sneezed. give me another shot.",
    "provider failure. annoy me again in a second.",
]

conversation_history = defaultdict(lambda: deque(maxlen=40))
channel_locks = defaultdict(asyncio.Lock)


def is_quota_error(exc):
    text = str(exc).lower()
    return any(term in text for term in (
        "429", "quota", "rate limit", "rate_limit", "resource_exhausted",
        "too many requests", "tokens per minute", "requests per day"
    ))


def remember(channel_id, user_id, username, role, content, display_name=""):
    conversation_history[channel_id].append({"role": role, "content": content})
    memory.remember_message(
        channel_id,
        user_id,
        username,
        role,
        content,
        display_name,
    )


def load_history(channel_id):
    if not conversation_history[channel_id]:
        conversation_history[channel_id].extend(
            memory.load_history(channel_id, limit=40)
        )
    return conversation_history[channel_id]


def user_context(message):
    user = message.author
    lines = [
        f"CURRENT USER: {user.display_name} (username: {user.name}, id: {user.id})"
    ]

    if message.guild:
        guild = message.guild
        lines.extend([
            f"SERVER: {guild.name} (id: {guild.id}, members: {guild.member_count})",
            f"CHANNEL: #{getattr(message.channel, 'name', 'DM')}",
        ])
        roles = [role.name for role in getattr(user, "roles", []) if role.name != "@everyone"]
        if roles:
            lines.append("USER ROLES: " + ", ".join(roles[:15]))

        permissions = getattr(user, "guild_permissions", None)
        if permissions:
            important = []
            for name in ("administrator", "manage_guild", "manage_messages", "moderate_members", "kick_members", "ban_members"):
                if getattr(permissions, name, False):
                    important.append(name)
            if important:
                lines.append("IMPORTANT PERMISSIONS: " + ", ".join(important))

    recent = memory.recent_users(message.channel.id, limit=10)
    if recent:
        lines.append("RECENT USERS:")
        for item in recent:
            lines.append(
                f"- {item['display_name']} / {item['username']} / {item['user_id']}"
            )

    return "\n".join(lines)


def prepare_visual(data, mime_type, filename):
    if mime_type == "image/gif":
        try:
            image = Image.open(io.BytesIO(data))
            image.seek(0)
            output = io.BytesIO()
            image.convert("RGB").save(output, format="PNG")
            return output.getvalue(), "image/png"
        except Exception as exc:
            print(f"Clyde GIF conversion failed for {filename}: {exc}")
            return None, None

    if mime_type == "image/webp":
        try:
            image = Image.open(io.BytesIO(data))
            output = io.BytesIO()
            image.convert("RGB").save(output, format="PNG")
            return output.getvalue(), "image/png"
        except Exception as exc:
            print(f"Clyde WebP conversion failed for {filename}: {exc}")
            return None, None

    return data, mime_type


async def get_attachments(message):
    context = []
    visuals = []

    for attachment in message.attachments:
        mime = attachment.content_type or "application/octet-stream"
        try:
            data = await attachment.read()
        except Exception as exc:
            context.append(
                f"Attachment {attachment.filename} could not be read: {exc}"
            )
            continue

        context.append(
            f"Attachment: {attachment.filename} ({mime}, {len(data)} bytes)"
        )

        if mime.startswith("image/"):
            visual_data, visual_mime = prepare_visual(
                data,
                mime,
                attachment.filename,
            )
            if visual_data:
                visuals.append({
                    "data": visual_data,
                    "mime_type": visual_mime,
                    "filename": attachment.filename,
                })

    return "\n".join(context), visuals


def groq_messages(hist, instruction, visuals=None):
    result = [{
        "role": "system",
        "content": CLYDE_PERSONALITY,
    }]

    result.extend({
        "role": "assistant" if item["role"] == "assistant" else "user",
        "content": item["content"],
    } for item in hist)

    if visuals:
        parts = [{
            "type": "text",
            "text": instruction,
        }]

        for visual in visuals:
            encoded = base64.b64encode(visual["data"]).decode("ascii")
            parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{visual['mime_type']};base64,{encoded}"
                },
            })

        result.append({
            "role": "user",
            "content": parts,
        })
    else:
        result.append({
            "role": "user",
            "content": instruction,
        })

    return result


def gemini_contents(hist, instruction, visuals=None):
    result = []

    for item in hist:
        result.append(
            types.Content(
                role="model" if item["role"] == "assistant" else "user",
                parts=[types.Part(text=item["content"])],
            )
        )

    parts = [types.Part(text=instruction)]

    for visual in visuals or []:
        parts.append(
            types.Part.from_bytes(
                data=visual["data"],
                mime_type=visual["mime_type"],
            )
        )

    result.append(
        types.Content(
            role="user",
            parts=parts,
        )
    )

    return result


async def ask_groq(hist, instruction, visuals=None):
    model = GROQ_VISION_MODEL if visuals else GROQ_MODEL

    kwargs = {
        "model": model,
        "messages": groq_messages(hist, instruction, visuals),
        "max_completion_tokens": 900,
        "temperature": 0.75 if visuals else 0.85,
    }

    if not visuals:
        kwargs["reasoning_effort"] = "low"

    return await groq.chat.completions.create(**kwargs)


async def ask_gemini(hist, instruction, visuals=None):
    return await asyncio.to_thread(
        gemini.models.generate_content,
        model=GEMINI_MODEL,
        contents=gemini_contents(hist, instruction, visuals),
        config=types.GenerateContentConfig(
            system_instruction=CLYDE_PERSONALITY,
            max_output_tokens=900,
            thinking_config=types.ThinkingConfig(
                thinking_level="low"
            ),
        ),
    )


def extract_groq(response):
    return (response.choices[0].message.content or "").strip()


def extract_gemini(response):
    return (response.text or "").strip()


def clean_reply(text):
    text = re.sub(r"\[PING:(\d+)\]", r"<@\1>", text or "")
    return text.strip()


async def clyde_reply(message, prompt, attachment_context="", visuals=None):
    visuals = visuals or []
    hist = load_history(message.channel.id)

    user_content = f"{message.author.display_name}: {prompt}"
    if attachment_context:
        user_content += f"\n[ATTACHMENTS]\n{attachment_context}"

    remember(
        message.channel.id,
        message.author.id,
        message.author.name,
        "user",
        user_content,
        message.author.display_name,
    )

    instruction = (
        f"{user_context(message)}\n"
        "Respond to the latest user message naturally as Clyde."
    )

    if visuals:
        instruction += (
            "\nVISUAL INPUT: Actual image data is attached. "
            "Inspect it carefully and base the response on what is visible."
        )

    providers = []

    if groq:
        providers.append((
            "Groq Vision" if visuals else "Groq",
            ask_groq,
            extract_groq,
        ))

    if gemini:
        providers.append((
            "Gemini Vision" if visuals else "Gemini",
            ask_gemini,
            extract_gemini,
        ))

    for provider_name, ask, extract in providers:
        try:
            response = await ask(hist, instruction, visuals)
            reply = clean_reply(extract(response))

            if reply:
                remember(
                    message.channel.id,
                    None,
                    "Clyde",
                    "assistant",
                    reply,
                    "Clyde",
                )
                print(f"Clyde AI provider: {provider_name}")
                return reply

            print(f"{provider_name} returned an empty response; trying next provider.")

        except Exception as exc:
            print(f"{provider_name} error: {exc}")
            if is_quota_error(exc):
                print(f"{provider_name} quota/rate limit reached; trying next provider.")

    reply = (
        "i can't inspect that image right now because both vision backends failed."
        if visuals
        else random.choice(FALLBACKS)
    )

    remember(
        message.channel.id,
        None,
        "Clyde",
        "assistant",
        reply,
        "Clyde",
    )

    return reply


def can_act(interaction, permission_name):
    if not interaction.guild:
        return False
    permissions = interaction.user.guild_permissions
    return bool(getattr(permissions, permission_name, False) or permissions.administrator)


def bot_can_target(interaction, member):
    if not interaction.guild or not member:
        return False
    me = interaction.guild.me
    if me is None:
        return False
    return me.top_role > member.top_role and member != interaction.guild.owner


def role_error(interaction, action):
    return interaction.response.send_message(
        f"I can't {action} that member because of Discord's role hierarchy.",
        ephemeral=True,
    )


@bot.event
async def setup_hook():
    synced = await bot.tree.sync()
    print(f"Clyde synced {len(synced)} slash command(s).")


@bot.event
async def on_ready():
    print("=" * 52)
    print("CLYDE ONLINE")
    print("=" * 52)
    print(f"Account: {bot.user}")
    print(f"Guilds: {len(bot.guilds)}")
    print(f"Text model: {GROQ_MODEL if groq else 'disabled'}")
    print(f"Vision model: {GROQ_VISION_MODEL if groq else 'disabled'}")
    print(f"Gemini fallback: {GEMINI_MODEL if gemini else 'disabled'}")
    print("Memory: PERSISTENT SQLITE")
    print("Vision: ONLINE" if (groq or gemini) else "Vision: OFFLINE")
    print("Discord tools: ONLINE")
    print("=" * 52)


@bot.event
async def on_message(message):
    if message.author.bot:
        return

    content = message.content.strip()
    if not content and not message.attachments:
        return

    memory.remember_user(
        message.author.id,
        message.author.name,
        message.author.display_name,
    )

    mentioned = bot.user is not None and bot.user in message.mentions
    is_dm = isinstance(message.channel, discord.DMChannel)

    attachment_context = ""
    visuals = []

    if message.attachments and (mentioned or is_dm):
        attachment_context, visuals = await get_attachments(message)

    if not (mentioned or is_dm):
        await bot.process_commands(message)
        return

    prompt = content
    if bot.user:
        prompt = prompt.replace(f"<@{bot.user.id}>", "")
        prompt = prompt.replace(f"<@!{bot.user.id}>", "")
        prompt = prompt.strip()

    if not prompt:
        prompt = "hello Clyde"

    async with channel_locks[message.channel.id]:
        reply = await clyde_reply(
            message,
            prompt,
            attachment_context,
            visuals,
        )
        await send_reply(message, reply)

    await bot.process_commands(message)


async def send_reply(message, text):
    text = text.strip()
    if not text:
        return

    while text:
        chunk = text[:1900]

        if len(text) > 1900:
            cut = max(
                chunk.rfind("\n"),
                chunk.rfind(" "),
            )
            if cut < 700:
                cut = 1900
            chunk = text[:cut].rstrip()

        text = text[len(chunk):].lstrip()

        async with message.channel.typing():
            await asyncio.sleep(
                min(
                    2.5,
                    max(0.25, len(chunk) * 0.012),
                )
            )

        await message.reply(
            chunk,
            mention_author=False,
        )


@bot.tree.command(name="help", description="Show Clyde's Discord tools.")
async def help_command(interaction: discord.Interaction):
    await interaction.response.send_message(
        HELP_TEXT,
        ephemeral=True,
    )


@bot.tree.command(name="clyde", description="Ask Clyde something.")
@app_commands.describe(prompt="What do you want to ask Clyde?")
async def clyde_command(interaction: discord.Interaction, prompt: str):
    await interaction.response.defer(thinking=True)

    class CommandMessage:
        pass

    pseudo = CommandMessage()
    pseudo.channel = interaction.channel
    pseudo.author = interaction.user
    pseudo.guild = interaction.guild
    pseudo.content = prompt
    pseudo.mentions = []
    pseudo.attachments = []

    reply = await clyde_reply(pseudo, prompt)
    chunks = []
    remaining = reply.strip()

    while remaining:
        chunk = remaining[:1900]
        if len(remaining) > 1900:
            cut = max(chunk.rfind("\n"), chunk.rfind(" "))
            if cut < 700:
                cut = 1900
            chunk = remaining[:cut].rstrip()
        chunks.append(chunk)
        remaining = remaining[len(chunk):].lstrip()

    await interaction.followup.send(chunks[0] if chunks else "...")
    for chunk in chunks[1:]:
        await interaction.channel.send(chunk)


@bot.tree.command(name="ping", description="Check Clyde's latency.")
async def ping_command(interaction: discord.Interaction):
    ms = round(bot.latency * 1000)
    await interaction.response.send_message(
        f"pong — {ms} ms",
        ephemeral=True,
    )


@bot.tree.command(name="serverinfo", description="Show server information.")
async def serverinfo_command(interaction: discord.Interaction):
    if not interaction.guild:
        await interaction.response.send_message(
            "this command only works in a server.",
            ephemeral=True,
        )
        return

    guild = interaction.guild
    text = (
        f"**{guild.name}**\n"
        f"Owner: <@{guild.owner_id}>\n"
        f"Members: {guild.member_count}\n"
        f"Channels: {len(guild.channels)}\n"
        f"Roles: {len(guild.roles)}\n"
        f"Created: {discord.utils.format_dt(guild.created_at, 'F')}"
    )

    await interaction.response.send_message(text)


@bot.tree.command(name="userinfo", description="Show information about a member.")
@app_commands.describe(member="Member to inspect")
async def userinfo_command(
    interaction: discord.Interaction,
    member: discord.Member,
):
    roles = [role.mention for role in member.roles if role.name != "@everyone"]
    text = (
        f"**{member.display_name}** (@{member.name})\n"
        f"ID: `{member.id}`\n"
        f"Joined: {discord.utils.format_dt(member.joined_at, 'F') if member.joined_at else 'unknown'}\n"
        f"Account created: {discord.utils.format_dt(member.created_at, 'F')}\n"
        f"Roles: {', '.join(roles[-10:]) if roles else 'none'}"
    )

    await interaction.response.send_message(text)


@bot.tree.command(name="avatar", description="Show a member's avatar.")
@app_commands.describe(member="Member whose avatar you want")
async def avatar_command(
    interaction: discord.Interaction,
    member: discord.Member | None = None,
):
    target = member or interaction.user
    embed = discord.Embed(title=f"{target.display_name}'s avatar")
    embed.set_image(url=target.display_avatar.url)
    await interaction.response.send_message(embed=embed)


@bot.tree.command(name="poll", description="Create a simple poll.")
@app_commands.describe(
    question="Poll question",
    option1="First option",
    option2="Second option",
    option3="Optional third option",
    option4="Optional fourth option",
    option5="Optional fifth option",
)
async def poll_command(
    interaction: discord.Interaction,
    question: str,
    option1: str,
    option2: str,
    option3: str | None = None,
    option4: str | None = None,
    option5: str | None = None,
):
    options = [option1, option2]
    for option in (option3, option4, option5):
        if option:
            options.append(option)

    symbols = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣"]
    lines = [f"**{question}**"]
    for index, option in enumerate(options):
        lines.append(f"{symbols[index]} {option}")

    message = await interaction.channel.send("\n".join(lines))
    for index in range(len(options)):
        try:
            await message.add_reaction(symbols[index])
        except discord.HTTPException:
            break

    await interaction.response.send_message("poll created.", ephemeral=True)


@bot.tree.command(name="clear", description="Delete recent messages from this channel.")
@app_commands.describe(amount="Number of recent messages to delete (1-100)")
async def clear_command(interaction: discord.Interaction, amount: app_commands.Range[int, 1, 100]):
    if not can_act(interaction, "manage_messages"):
        await interaction.response.send_message(
            "you need Manage Messages to do that.",
            ephemeral=True,
        )
        return

    if not interaction.channel or not hasattr(interaction.channel, "purge"):
        await interaction.response.send_message(
            "this channel doesn't support message cleanup.",
            ephemeral=True,
        )
        return

    await interaction.response.defer(ephemeral=True)

    try:
        deleted = await interaction.channel.purge(limit=amount)
        await interaction.followup.send(
            f"deleted {len(deleted)} message(s).",
            ephemeral=True,
        )
    except discord.Forbidden:
        await interaction.followup.send(
            "Discord denied that action. Check my Manage Messages permission and channel overrides.",
            ephemeral=True,
        )


@bot.tree.command(name="timeout", description="Timeout a member.")
@app_commands.describe(
    member="Member to timeout",
    minutes="Timeout duration in minutes (1-10080)",
    reason="Reason for the timeout",
)
async def timeout_command(
    interaction: discord.Interaction,
    member: discord.Member,
    minutes: app_commands.Range[int, 1, 10080],
    reason: str | None = None,
):
    if not can_act(interaction, "moderate_members"):
        await interaction.response.send_message(
            "you need Moderate Members to do that.",
            ephemeral=True,
        )
        return

    if not bot_can_target(interaction, member):
        await role_error(interaction, "timeout")
        return

    try:
        await member.timeout(
            timedelta(minutes=minutes),
            reason=reason or f"Clyde timeout by {interaction.user}",
        )
    except discord.Forbidden:
        await interaction.response.send_message(
            "Discord denied the timeout. Check my Moderate Members permission and role hierarchy.",
            ephemeral=True,
        )
        return

    await interaction.response.send_message(
        f"timed out {member.mention} for {minutes} minute(s).",
    )


@bot.tree.command(name="untimeout", description="Remove a member's timeout.")
@app_commands.describe(member="Member to untimeout")
async def untimeout_command(
    interaction: discord.Interaction,
    member: discord.Member,
):
    if not can_act(interaction, "moderate_members"):
        await interaction.response.send_message(
            "you need Moderate Members to do that.",
            ephemeral=True,
        )
        return

    if not bot_can_target(interaction, member):
        await role_error(interaction, "remove that timeout from")
        return

    try:
        await member.timeout(
            None,
            reason=f"Clyde timeout removal by {interaction.user}",
        )
    except discord.Forbidden:
        await interaction.response.send_message(
            "Discord denied that action.",
            ephemeral=True,
        )
        return

    await interaction.response.send_message(
        f"removed {member.mention}'s timeout.",
    )


@bot.tree.command(name="kick", description="Kick a member.")
@app_commands.describe(member="Member to kick", reason="Reason for the kick")
async def kick_command(
    interaction: discord.Interaction,
    member: discord.Member,
    reason: str | None = None,
):
    if not can_act(interaction, "kick_members"):
        await interaction.response.send_message(
            "you need Kick Members to do that.",
            ephemeral=True,
        )
        return

    if not bot_can_target(interaction, member):
        await role_error(interaction, "kick")
        return

    try:
        await member.kick(reason=reason or f"Clyde kick by {interaction.user}")
    except discord.Forbidden:
        await interaction.response.send_message(
            "Discord denied the kick. Check role hierarchy and Kick Members permission.",
            ephemeral=True,
        )
        return

    await interaction.response.send_message(
        f"kicked `{member}`.",
    )


@bot.tree.command(name="ban", description="Ban a member.")
@app_commands.describe(member="Member to ban", reason="Reason for the ban")
async def ban_command(
    interaction: discord.Interaction,
    member: discord.Member,
    reason: str | None = None,
):
    if not can_act(interaction, "ban_members"):
        await interaction.response.send_message(
            "you need Ban Members to do that.",
            ephemeral=True,
        )
        return

    if not bot_can_target(interaction, member):
        await role_error(interaction, "ban")
        return

    try:
        await member.ban(reason=reason or f"Clyde ban by {interaction.user}")
    except discord.Forbidden:
        await interaction.response.send_message(
            "Discord denied the ban. Check role hierarchy and Ban Members permission.",
            ephemeral=True,
        )
        return

    await interaction.response.send_message(
        f"banned `{member}`.",
    )


@bot.tree.command(name="announce", description="Send an announcement embed.")
@app_commands.describe(title="Announcement title", message="Announcement body")
async def announce_command(
    interaction: discord.Interaction,
    title: str,
    message: str,
):
    if not can_act(interaction, "manage_guild"):
        await interaction.response.send_message(
            "you need Manage Server to do that.",
            ephemeral=True,
        )
        return

    embed = discord.Embed(
        title=title[:256],
        description=message[:4096],
        timestamp=discord.utils.utcnow(),
    )
    embed.set_footer(text=f"Announcement by {interaction.user.display_name}")

    await interaction.channel.send(embed=embed)
    await interaction.response.send_message(
        "announcement sent.",
        ephemeral=True,
    )


@bot.tree.command(name="forget", description="Clear Clyde's saved conversation for this channel.")
async def forget_command(interaction: discord.Interaction):
    conversation_history[interaction.channel_id].clear()
    memory.forget_channel(interaction.channel_id)
    await interaction.response.send_message(
        "channel memory cleared.",
        ephemeral=True,
    )


bot.run(DISCORD_TOKEN)
