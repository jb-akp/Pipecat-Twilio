# Function Calling Flow - Step by Step

## Setup Phase (One Time)

```
1. Tool Definition Created
   whatsapp_tool_definition = {
       "name": "send_whatsapp_message",
       "description": "...",
       "parameters": {...}
   }
   ↓
   [Stored in Context]

2. Handler Registered
   llm.register_function("send_whatsapp_message", send_whatsapp_message)
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

```
LLM Response: {
    "tool_calls": [{
        "id": "call_123",
        "function": {
            "name": "send_whatsapp_message",
            "arguments": '{"order_summary": "1 pizza"}'
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
Registered Handler Found → send_whatsapp_message(params)
    ↓
Your Python code runs
    ↓
params.result_callback(result) → Result back to LLM
```

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

