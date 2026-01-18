# ResearchGravity MCP Integration

**Model Context Protocol (MCP) Server for ResearchGravity**

---

## Overview

This MCP server exposes ResearchGravity's research context, learnings, and context packs to **any MCP-compatible client**. MCP is an open protocol created by Anthropic that enables AI applications to access tools and context.

**Universal Protocol:** Works with Claude Desktop, custom clients, or any application that implements the MCP protocol.

---

## What is MCP?

**Model Context Protocol (MCP)** is an open standard for connecting AI assistants to tools and data sources. Think of it like USB-C for AI context:

- **Universal:** Works with any MCP client (Claude Desktop, custom apps, etc.)
- **Bidirectional:** AI can read data AND perform actions
- **Secure:** Runs locally, no cloud intermediary
- **Extensible:** Easy to add new tools and resources

### MCP Architecture

```
┌─────────────────┐
│   MCP Client    │  (Claude Desktop, custom app, etc.)
│  (AI Assistant) │
└────────┬────────┘
         │ stdio/HTTP
         │
┌────────▼────────┐
│   MCP Server    │  (ResearchGravity)
│   (This code)   │
└────────┬────────┘
         │
┌────────▼────────┐
│ Data Sources    │  (~/.agent-core/, sessions, learnings)
│  & Actions      │
└─────────────────┘
```

---

## Tools Provided

The ResearchGravity MCP server exposes 8 tools:

### 1. get_session_context
Get active or specific research session information.

**Inputs:**
- `session_id` (optional): Specific session ID, or omit for active session

**Returns:** Session topic, URLs, findings, status

**Example:**
```json
{
  "name": "get_session_context",
  "arguments": {}
}
```

---

### 2. search_learnings
Search archived learnings from past research sessions.

**Inputs:**
- `query` (required): Search keywords or phrases
- `limit` (optional): Max results (default: 10)

**Returns:** Relevant concepts, findings, and papers

**Example:**
```json
{
  "name": "search_learnings",
  "arguments": {
    "query": "multi-agent consensus",
    "limit": 5
  }
}
```

---

### 3. get_project_research
Load research files for a specific project.

**Inputs:**
- `project_name` (required): Project name (os-app, careercoach, metaventions)

**Returns:** All research markdown files for the project

**Example:**
```json
{
  "name": "get_project_research",
  "arguments": {
    "project_name": "os-app"
  }
}
```

---

### 4. log_finding
Record a finding to the active research session.

**Inputs:**
- `finding` (required): Finding text
- `type` (optional): Finding type (general, implementation, metrics, innovation)

**Returns:** Success confirmation

**Example:**
```json
{
  "name": "log_finding",
  "arguments": {
    "finding": "Context Packs V2 achieves 99%+ token reduction",
    "type": "metrics"
  }
}
```

---

### 5. select_context_packs
Select relevant context packs using Context Packs V2 (7-layer system).

**Inputs:**
- `query` (required): Query for pack selection
- `budget` (optional): Token budget (default: 50000)
- `use_v1` (optional): Force V1 engine (default: false)

**Returns:** Selected packs with metadata

**Example:**
```json
{
  "name": "select_context_packs",
  "arguments": {
    "query": "debugging React performance",
    "budget": 50000
  }
}
```

---

### 6. get_research_index
Get the unified cross-project research index.

**Returns:** Complete research index with papers and concepts

---

### 7. list_projects
List all tracked projects with metadata, tech stack, and status.

**Returns:** Project list with details

---

### 8. get_session_stats
Get statistics about sessions, papers, concepts, and cognitive wallet value.

**Returns:** ResearchGravity metrics

---

## Resources Provided

The server exposes resources that clients can read directly:

### Session Resources
- `session://active` - Active session data (JSON)
- `session://{id}` - Specific session by ID (JSON)

### Memory Resources
- `learnings://all` - All archived learnings (Markdown)
- `research://index` - Unified research index (Markdown)

### Project Resources
- `project://{name}/research` - Project research files (Markdown)
  - Example: `project://os-app/research`
  - Example: `project://careercoach/research`

---

## Setup for Any MCP Client

### Universal Setup (Stdio Mode)

The server uses stdio (standard input/output) for communication. Any MCP client can connect:

```bash
# Start server (for testing/manual use)
python3 /Users/dicoangelo/researchgravity/mcp_server.py
```

The server will:
1. Initialize MCP protocol
2. Listen on stdin for MCP messages
3. Send responses to stdout
4. Log errors to stderr

### Client Configuration Template

```json
{
  "servers": {
    "researchgravity": {
      "command": "python3",
      "args": ["/absolute/path/to/researchgravity/mcp_server.py"],
      "env": {}
    }
  }
}
```

Replace `/absolute/path/to/` with your actual path.

---

## Setup for Claude Desktop

### 1. Copy Configuration

Copy `claude_desktop_config.json` to Claude Desktop's config directory:

**macOS:**
```bash
# Create directory if needed
mkdir -p ~/Library/Application\ Support/Claude/

# Copy config (merge with existing if you have one)
cp ~/researchgravity/claude_desktop_config.json ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

**Windows:**
```cmd
mkdir %APPDATA%\Claude
copy researchgravity\claude_desktop_config.json %APPDATA%\Claude\claude_desktop_config.json
```

**Linux:**
```bash
mkdir -p ~/.config/Claude
cp ~/researchgravity/claude_desktop_config.json ~/.config/Claude/claude_desktop_config.json
```

### 2. Restart Claude Desktop

Restart the Claude Desktop app for changes to take effect.

### 3. Verify Connection

In Claude Desktop, you should see:
- 🔌 "ResearchGravity" in the server list
- 8 tools available
- Resources accessible

### 4. Test Tools

Try these commands in Claude Desktop:

```
Get my active session context
```

```
Search learnings for "multi-agent orchestration"
```

```
Select context packs for "debugging React performance"
```

---

## Building Custom MCP Clients

### Python Client Example

```python
import asyncio
import json
from mcp.client import ClientSession, stdio_client

async def main():
    # Connect to ResearchGravity MCP server
    async with stdio_client(
        command="python3",
        args=["/Users/dicoangelo/researchgravity/mcp_server.py"]
    ) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize
            await session.initialize()

            # List available tools
            tools = await session.list_tools()
            print(f"Available tools: {[t.name for t in tools]}")

            # Call a tool
            result = await session.call_tool(
                "get_session_context",
                arguments={}
            )
            print(f"Session context: {result.content}")

            # Read a resource
            resource = await session.read_resource(
                "session://active"
            )
            print(f"Active session: {resource.contents}")

if __name__ == "__main__":
    asyncio.run(main())
```

### JavaScript/TypeScript Client Example

```typescript
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

async function main() {
  // Create transport
  const transport = new StdioClientTransport({
    command: "python3",
    args: ["/Users/dicoangelo/researchgravity/mcp_server.py"]
  });

  // Connect client
  const client = new Client({
    name: "researchgravity-client",
    version: "1.0.0"
  }, {
    capabilities: {}
  });

  await client.connect(transport);

  // List tools
  const tools = await client.listTools();
  console.log("Available tools:", tools.tools.map(t => t.name));

  // Call tool
  const result = await client.callTool({
    name: "search_learnings",
    arguments: {
      query: "multi-agent consensus",
      limit: 5
    }
  });

  console.log("Search results:", result.content);

  // Close connection
  await client.close();
}

main();
```

---

## API Reference

### Tool Call Format

All tools use this JSON format:

```json
{
  "method": "tools/call",
  "params": {
    "name": "tool_name",
    "arguments": {
      "param1": "value1",
      "param2": "value2"
    }
  }
}
```

### Resource Read Format

```json
{
  "method": "resources/read",
  "params": {
    "uri": "resource://uri"
  }
}
```

### Response Format

```json
{
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Response text here"
      }
    ]
  }
}
```

---

## Use Cases

### 1. IDE Integration

Connect ResearchGravity to your IDE (VSCode, JetBrains) via custom MCP extension:

```javascript
// VSCode extension example
const client = createMCPClient("researchgravity");

// Get context for current file
const projectContext = await client.callTool("get_project_research", {
  project_name: detectProject()
});

// Inject into completion context
editor.setContext(projectContext);
```

### 2. CLI Automation

```bash
# Get session stats in terminal
echo '{"method":"tools/call","params":{"name":"get_session_stats","arguments":{}}}' | \
  python3 ~/researchgravity/mcp_server.py | \
  jq '.result.content[0].text'
```

### 3. Team Dashboard

Build a web dashboard that uses MCP to display research metrics:

```python
# Flask/FastAPI server
@app.get("/api/stats")
async def get_stats():
    result = await mcp_client.call_tool("get_session_stats", {})
    return jsonify(result)

@app.get("/api/learnings/search")
async def search_learnings(query: str):
    result = await mcp_client.call_tool("search_learnings", {
        "query": query,
        "limit": 20
    })
    return jsonify(result)
```

### 4. Research Assistant Bot

Create a Slack/Discord bot that accesses ResearchGravity:

```python
@bot.command()
async def research(ctx, query):
    """Search learnings"""
    result = await mcp_client.call_tool("search_learnings", {
        "query": query,
        "limit": 5
    })
    await ctx.send(result)

@bot.command()
async def session(ctx):
    """Get active session"""
    result = await mcp_client.call_tool("get_session_context", {})
    await ctx.send(result)
```

---

## Troubleshooting

### Server Won't Start

**Error:** `ModuleNotFoundError: No module named 'mcp'`

**Solution:**
```bash
pip3 install mcp --break-system-packages
```

---

### Tools Not Appearing in Claude Desktop

**Solution:**
1. Check config location:
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`
   - Linux: `~/.config/Claude/claude_desktop_config.json`

2. Verify path in config is absolute:
   ```json
   {
     "mcpServers": {
       "researchgravity": {
         "command": "python3",
         "args": ["/Users/your-username/researchgravity/mcp_server.py"]
       }
     }
   }
   ```

3. Restart Claude Desktop completely (quit and reopen)

---

### Permission Errors

**Error:** `Permission denied: ~/.agent-core/session_tracker.json`

**Solution:**
```bash
# Check permissions
ls -la ~/.agent-core/session_tracker.json

# Fix if needed
chmod 644 ~/.agent-core/session_tracker.json
```

---

### No Active Session

**Error:** Tool returns "No active session"

**Solution:**
```bash
# Start a research session
cd ~/researchgravity
python3 init_session.py "Your Research Topic"
```

---

## Development

### Adding New Tools

1. Define tool in `list_tools()`:
```python
Tool(
    name="your_tool",
    description="What it does",
    inputSchema={
        "type": "object",
        "properties": {
            "param": {"type": "string", "description": "Parameter"}
        },
        "required": ["param"]
    }
)
```

2. Handle in `call_tool()`:
```python
elif name == "your_tool":
    param = arguments["param"]
    result = your_function(param)
    return [TextContent(type="text", text=result)]
```

3. Test:
```bash
python3 mcp_server.py
# Send test request via stdio
```

### Adding New Resources

1. Add to `list_resources()`:
```python
{
    "uri": "your://resource",
    "name": "Your Resource",
    "description": "What it provides",
    "mimeType": "text/markdown"
}
```

2. Handle in `read_resource()`:
```python
elif uri == "your://resource":
    return load_your_data()
```

---

## MCP Protocol Specification

**Official Spec:** https://spec.modelcontextprotocol.io/

**Key Concepts:**
- **Tools:** Functions the AI can call (like `get_session_context`)
- **Resources:** Data the AI can read (like `session://active`)
- **Prompts:** Pre-defined prompt templates (not used in ResearchGravity yet)
- **Sampling:** AI can request completions (not used in ResearchGravity yet)

**Transport:**
- **Stdio:** Communication via stdin/stdout (what we use)
- **HTTP/SSE:** Communication via HTTP with Server-Sent Events (alternative)

---

## Security Considerations

### Local-Only by Default

The MCP server:
- ✅ Runs locally (no cloud intermediary)
- ✅ Accesses only `~/.agent-core/` data
- ✅ Requires file system permissions
- ✅ No network access required

### Recommended Practices

1. **File Permissions:** Ensure `~/.agent-core/` has proper permissions
2. **No Secrets:** Don't log API keys or credentials
3. **Sandboxing:** Consider running in a container for production
4. **Audit Logging:** Monitor tool calls for unexpected behavior

---

## Roadmap

### v4.1 (Planned)
- HTTP/SSE transport (for web clients)
- Authentication & authorization
- Rate limiting
- Prompt templates
- Sampling support (AI-to-AI context)

### v4.2 (Planned)
- Browser extension MCP client
- Team collaboration via shared MCP server
- Cloud-hosted option for teams

---

## Examples Repository

See `examples/` directory for:
- Python client examples
- JavaScript/TypeScript client examples
- Custom integration examples
- Test scripts

---

## Contributing

To contribute to MCP integration:

1. Follow MCP protocol specification
2. Add tests for new tools/resources
3. Update this documentation
4. Submit PR with examples

---

## License

MIT License - See LICENSE file for details

---

## Support

**Issues:** https://github.com/Dicoangelo/ResearchGravity/issues
**MCP Spec:** https://spec.modelcontextprotocol.io/
**MCP SDK:** https://github.com/modelcontextprotocol/

---

**ResearchGravity MCP Server** - Universal context access via Model Context Protocol

**Status:** v4.0 Complete | Universal MCP Support ✅
**Protocol:** MCP 1.0 (compatible with any MCP client)
**Last Updated:** 2026-01-18
