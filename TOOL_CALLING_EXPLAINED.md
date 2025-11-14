# Function Calling Flow - Step by Step

## Setup Phase (One Time)

```
1. Tool Definition Created
   whatsapp_tool_definition = {
       "name": "send_whatsapp_message",
       "description": "...",
       "parameters": {
           "order_summary": {"type": "string", ...},
           "phone_number": {"type": "string", ...}  // ← Dynamic phone number!
       },
       "required": ["order_summary", "phone_number"]
   }
   ↓
   [Stored in Context]

2. Handler Registered
   llm.register_function("send_whatsapp_message", handle_whatsapp_order_confirmation)
   ↓
   [Stored in LLM Service: llm._functions = {"send_whatsapp_message": handler}]
```

## Runtime Phase (Every LLM Request)

### Step 1: Context with Tools → OpenAI API

```
Context (has tools) → LLM Service → OpenAI API
                         ↓
        "Here are the functions available: [tool definition]"
```

**What happens:**
- `context.tools` is included in the API request
- OpenAI's LLM sees: "You have a function called `send_whatsapp_message`"
- LLM can decide to call it

**If you only had handler registration:**
- OpenAI would never know the function exists
- LLM would never call it

### Step 2: LLM Decides to Call Function

During conversation, the LLM collects:
- Order items (from user's spoken order)
- Phone number (user provides during conversation)

When user confirms, LLM calls the function:

```
LLM Response: {
    "tool_calls": [{
        "id": "call_123",
        "function": {
            "name": "send_whatsapp_message",
            "arguments": '{"order_summary": "1 large pepperoni pizza, 2 cokes", "phone_number": "+15551234567"}'
        }
    }]
}
```

### Step 3: LLM Service Receives Function Call

```
LLM Service: "LLM wants to call 'send_whatsapp_message'"
    ↓
Check: Is it registered? → llm._functions["send_whatsapp_message"]
    ↓
If NOT registered: Warning + Skip execution
If registered: Execute handler(params)
```

**If you only had tool definition:**
- LLM would call the function
- But Python wouldn't know what code to run
- Function call would fail silently or error

### Step 4: Execute Handler

```
Registered Handler Found → handle_whatsapp_order_confirmation(params)
    ↓
Extract arguments:
  - order_summary = params.arguments.get("order_summary")
  - phone_number = params.arguments.get("phone_number")  // ← Collected from conversation!
    ↓
Send WhatsApp message to phone_number
    ↓
params.result_callback(result) → Result back to LLM
```

**Key Point:** The LLM constructs both `order_summary` and `phone_number` from the conversation context. The phone number is collected dynamically during the conversation, not hardcoded!

## Why Both Are Essential

**Tool Definition → Enables LLM to CALL the function**
- Without it: LLM never knows function exists
- Result: Function never gets called

**Handler Registration → Enables Python to EXECUTE the function**
- Without it: Function call happens but no code runs
- Result: "Function not registered" error

## They Work Together:

```
Tool Definition (Context)
    ↓
[LLM sees it in API request]
    ↓
LLM decides to call function
    ↓
LLM Service receives function call
    ↓
[Looks up handler registration]
    ↓
Handler (Python Function)
    ↓
Code executes!
```

