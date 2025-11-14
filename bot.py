#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat Order Taker Bot with WhatsApp Confirmation.

The bot takes an order and uses OpenAI Function Calling to trigger a custom
Python function that sends the final order confirmation via the Twilio WhatsApp API.

Required AI services:
- Deepgram (Speech-to-Text)
- OpenAI (LLM)
- Cartesia (Text-to-Speech)

Required API keys:
- DEEPGRAM_API_KEY
- OPENAI_API_KEY
- CARTESIA_API_KEY
- TWILIO_ACCOUNT_SID
- TWILIO_AUTH_TOKEN
- TWILIO_WHATSAPP_NUMBER
- RECIPIENT_NUMBER

Run the bot using::

    uv run bot.py
"""

import os

from dotenv import load_dotenv
from loguru import logger
from twilio.rest import Client

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
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams

logger.info("✅ All components loaded successfully!")

load_dotenv(override=True)


# Custom WhatsApp function to send order confirmations
async def handle_whatsapp_order_confirmation(params: FunctionCallParams):
    """Send order confirmation via Twilio WhatsApp.
    
    Args:
        params: FunctionCallParams containing the order_summary and phone_number in arguments
        
    Returns:
        str: Confirmation message
    """
    try:
        order_summary = params.arguments.get("order_summary", "")
        # Get the phone number passed by the LLM (should already be normalized per prompt instructions)
        recipient_number = params.arguments.get("phone_number", "")
        to_number = f"whatsapp:{recipient_number}"  # Format for Twilio
        
        # Get Twilio credentials from environment
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        from_number = os.getenv("TWILIO_WHATSAPP_NUMBER")  # Should be: +14155238886 (just the number, no whatsapp: prefix)
        from_number = f"whatsapp:{from_number}"  # Add whatsapp: prefix for Twilio
        
        logger.info(f"📤 Preparing WhatsApp message:")
        logger.info(f"   From: {from_number}")
        logger.info(f"   To: {to_number}")
        
        # Initialize Twilio client
        client = Client(account_sid, auth_token)
        
        # Format the message
        message_body = (
            f"📦 Order Confirmed!\n\n"
            f"Your order:\n{order_summary}\n\n"
            f"Thank you for your order! We'll send you updates shortly."
        )
        
        message = client.messages.create(
            from_=from_number,
            body=message_body,
            to=to_number
        )
        
        logger.info(f"✅ WhatsApp message sent successfully. SID: {message.sid}")
        logger.info(f"   From: {from_number}")
        logger.info(f"   To: {to_number}")
        logger.info(f"   Message Status: {message.status}")
        logger.info(f"   Full response: {message}")
        
        result = f"Order confirmation sent to WhatsApp number {recipient_number} successfully! Order: {order_summary}"
        await params.result_callback(result)
        return result
        
    except Exception as e:
        logger.error(f"❌ Failed to send WhatsApp message: {e}")
        error_result = f"Error sending WhatsApp message: {str(e)}"
        await params.result_callback(error_result)
        return error_result


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        model_id="sonic-3", # <--- ADD THIS LINE
        voice_id="f786b574-daa5-4673-aa0c-cbe3e8534c02",  # Recommended Sonic 3 Voice: Katie (American Female)
    )

    # Define the WhatsApp function schema for the LLM
    whatsapp_tool_definition = {
        "type": "function",
        "function": {
            "name": "send_whatsapp_message",
            "description": "Send an order confirmation message to the customer via WhatsApp. Call this function when the user confirms their order and is ready to proceed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_summary": {
                        "type": "string",
                        "description": "A detailed summary of the customer's complete order including all items, quantities, and any special instructions.",
                    },
                    "phone_number": {
                        "type": "string",
                        "description": "The customer's phone number, including the country code, formatted with only digits and a plus sign (e.g., +16507303690). No spaces, parentheses, or dashes. This is the number to send the WhatsApp confirmation to.",
                    }
                },
                "required": ["order_summary", "phone_number"],
            },
        },
    }

    # Initialize the LLM
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Register the function handler
    # Note: First parameter ("send_whatsapp_message") must match tool definition name
    # Second parameter (handle_whatsapp_order_confirmation) can be any function name
    llm.register_function("send_whatsapp_message", handle_whatsapp_order_confirmation)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a friendly and efficient order taker for a restaurant or store. "
                "Your job is to:\n"
                "1. Greet the customer warmly\n"
                "2. Ask what they would like to order\n"
                "3. Collect all items, quantities, and any special instructions\n"
                "4. Repeat back the complete order and ask them to confirm the order is correct (food and toppings only)\n"
                "5. Once the order is confirmed, THEN ask for their phone number (including country code, e.g., +1...) for the WhatsApp confirmation\n"
                "6. Repeat back the phone number and ask them to confirm the phone number is correct\n"
                "7. Once both the order and phone number are confirmed, ask if they're ready to proceed with the WhatsApp confirmation\n"
                "8. When they confirm they're ready to proceed (say yes, confirm, sounds good, etc.), you MUST immediately call the 'send_whatsapp_message' function with the complete order summary AND the collected phone number. IMPORTANT: Format the phone_number parameter with only digits and a plus sign (e.g., +16507303690), removing any spaces, parentheses, or dashes the user may have provided.\n"
                "9. After the function is called, thank them and let them know the confirmation has been sent\n\n"
                "Be conversational, friendly, and make sure you have all the details before confirming."
            ),
        },
    ]

    # Initialize the LLM context with tools (per docs, tools go in the context, not the LLM service)
    context = OpenAILLMContext(
        messages=messages,
        tools=[whatsapp_tool_definition]
    )
    
    # Create context aggregator using the LLM service method (per docs)
    context_aggregator = llm.create_context_aggregator(context)

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    # Configure STT mute filter to mute during function calls (prevents awkward silence)
    stt_mute_filter = STTMuteFilter(
        config=STTMuteConfig(strategies={STTMuteStrategy.FUNCTION_CALL})
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            rtvi,
            stt_mute_filter,  # Mute STT during function calls
            stt,
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
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
        # Kick off the conversation.
        messages.append({"role": "system", "content": "Greet the customer warmly and ask what they would like to order today."})
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
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            turn_analyzer=LocalSmartTurnAnalyzerV3(),
        ),
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            turn_analyzer=LocalSmartTurnAnalyzerV3(),
        ),
    }

    transport = await create_transport(runner_args, transport_params)

    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()