#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat Personal Assistant Bot.

A personal assistant that can check Google Calendar, check Gmail, and send reminders
via WhatsApp using OpenAI Function Calling.

Required AI services:
- Deepgram (Speech-to-Text)
- OpenAI (LLM)
- Cartesia (Text-to-Speech)
- Tavus (Video Avatar)

Required API keys:
- DEEPGRAM_API_KEY
- OPENAI_API_KEY
- CARTESIA_API_KEY
- TAVUS_API_KEY
- TAVUS_REPLICA_ID
- TWILIO_ACCOUNT_SID
- TWILIO_AUTH_TOKEN
- TWILIO_WHATSAPP_NUMBER
- RECIPIENT_NUMBER
- GOOGLE_CREDENTIALS_PATH (path to Google OAuth credentials JSON file)
- GOOGLE_TOKEN_PATH (optional, defaults to token.json)

Run the bot using::

    uv run bot.py
"""
import os

import aiohttp
from dotenv import load_dotenv
from loguru import logger

print("🚀 Starting Pipecat bot...")
print("⏳ Loading models and imports (20 seconds, first run only)\n")

logger.info("Loading Local Smart Turn Analyzer V3...")
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

logger.info("✅ Local Smart Turn Analyzer V3 loaded")
logger.info("Loading Silero VAD model...")
from pipecat.audio.vad.silero import SileroVADAnalyzer

logger.info("✅ Silero VAD model loaded")

from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.filters.stt_mute_filter import STTMuteConfig, STTMuteFilter, STTMuteStrategy
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.tavus.video import TavusVideoService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams

logger.info("✅ All components loaded successfully!")

load_dotenv(override=True)

# Import tool functions
from functions import get_calendar_events, get_gmail_emails, send_whatsapp_reminder


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")
    async with aiohttp.ClientSession() as session:
        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            model_id="sonic-3",  # Use the newest, highest-performing model
            voice_id="f786b574-daa5-4673-aa0c-cbe3e8534c02",  # Recommended Sonic 3 Voice: Katie
        )

        # Define the Calendar function schema for the LLM
        calendar_tool_definition = {
            "type": "function",
            "function": {
                "name": "get_calendar_events",
                "description": "Get calendar events for TODAY. Use this when the user asks about their agenda, schedule, meetings, or what's on their calendar for today.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }
        
        # Define the Gmail function schema for the LLM
        gmail_tool_definition = {
            "type": "function",
            "function": {
                "name": "get_gmail_emails",
                "description": "Get the 2 most recent Gmail emails. Use this when the user asks about their emails, messages, or wants to check their inbox.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }
        
        # Define the WhatsApp reminder function schema for the LLM
        whatsapp_tool_definition = {
            "type": "function",
            "function": {
                "name": "send_whatsapp_reminder",
                "description": "Send a reminder message via WhatsApp. Use this when the user asks you to send them a text reminder, summary, or message with information they need to remember.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reminder_text": {
                            "type": "string",
                            "description": "The text content of the reminder message to send. This should include all the important information the user wants to be reminded of (e.g., calendar events, tasks, notes).",
                        }
                    },
                    "required": ["reminder_text"],
                },
            },
        }

        # Initialize the LLM
        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

        # Register the function handlers
        # Note: First parameter must match tool definition name
        # Second parameter can be any function name
        llm.register_function("get_calendar_events", get_calendar_events)
        llm.register_function("get_gmail_emails", get_gmail_emails)
        llm.register_function("send_whatsapp_reminder", send_whatsapp_reminder)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful and friendly personal assistant named James. "
                    "You help manage calendar, emails, and can send reminders via WhatsApp.\n\n"
                    "Your capabilities:\n"
                    "1. Check calendar events for TODAY using the 'get_calendar_events' function when asked about agenda, schedule, meetings, or what's on their calendar for today\n"
                    "2. Check Gmail emails using the 'get_gmail_emails' function when asked about emails, messages, or inbox. This returns the 2 most recent emails.\n"
                    "3. Send reminders via WhatsApp using the 'send_whatsapp_reminder' function when the user asks you to send them a text reminder or summary\n\n"
                    "Be conversational and natural. When the user asks about their agenda or calendar, use the calendar function. "
                    "When they ask about emails or messages, use the Gmail function. "
                    "When they ask you to send a reminder or text, gather the information they want included and use the WhatsApp function. "
                    "Keep responses concise and helpful.\n\n"
                    "When the user first greets you, respond with: 'Good morning! Are you ready to start the day?' This should be your first response after they greet you.\n\n"
                    "IMPORTANT: When responding about emails, be casual and human-like. Don't list emails formally with subjects. "
                    "Instead, speak naturally like: 'yeah, you got one from a colleague talking about the livestream' or "
                    "'someone gave you a lighthearted update about genai trends.' Use the snippet and subject to understand what each email is about, "
                    "then summarize it casually in your own words. Keep it conversational, not robotic."
                ),
            },
        ]

        # Initialize the LLM context with tools (per docs, tools go in the context, not the LLM service)
        context = OpenAILLMContext(
            messages,
            tools=[calendar_tool_definition, gmail_tool_definition, whatsapp_tool_definition]
        )
        
        # Create context aggregator using the LLM service method (per docs)
        context_aggregator = llm.create_context_aggregator(context)

        tavus = TavusVideoService(
            api_key=os.getenv("TAVUS_API_KEY"),
            replica_id=os.getenv("TAVUS_REPLICA_ID"),
            persona_id="pipecat-stream",  # Uses your bot's TTS voice (Cartesia) instead of Tavus persona voice
            session=session,
        )

        rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

        # Configure STT mute filter to mute during function calls (prevents awkward silence)
        stt_mute_filter = STTMuteFilter(config=STTMuteConfig(strategies={STTMuteStrategy.FUNCTION_CALL}))

        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                rtvi,
                stt_mute_filter,  # Mute STT during function calls
                stt,
                context_aggregator.user(),  # User responses
                llm,  # LLM
                tts,  # TTS
                tavus,  # Tavus output layer
                transport.output(),  # Transport bot output
                context_aggregator.assistant(),  # Assistant spoken responses
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
            observers=[RTVIObserver(rtvi)],
        )

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info(f"Client connected")
            # Kick off the conversation so the bot greets the user
            messages.append(
                {
                    "role": "system",
                    "content": "Start by greeting the user with: 'Good morning! Are you ready to start the day?'"
                }
            )
            await task.queue_frames([LLMRunFrame()])

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info(f"Client disconnected")
            await task.cancel()

        runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

        await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point for the bot starter."""

    transport_params = {
        "daily": lambda: DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_out_enabled=True,
            video_out_is_live=True,
            video_out_width=1280,
            video_out_height=720,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            turn_analyzer=LocalSmartTurnAnalyzerV3(),
        ),
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_out_enabled=True,
            video_out_is_live=True,
            video_out_width=1280,
            video_out_height=720,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            turn_analyzer=LocalSmartTurnAnalyzerV3(),
        ),
    }

    transport = await create_transport(runner_args, transport_params)
    
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
