import asyncio
import os
import logging
from typing import List, Annotated

from dotenv import load_dotenv
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client

from a2a.types import Message
from a2a.utils.message import get_message_text
from beeai_sdk.server import Server
from beeai_sdk.a2a.types import AgentMessage
from beeai_sdk.a2a.extensions import LLMServiceExtensionServer, LLMServiceExtensionSpec
from beeai_framework.tools.mcp import MCPTool
from beeai_framework.agents.tool_calling import ToolCallingAgent
from beeai_framework.adapters.openai import OpenAIChatModel
from beeai_framework.backend.types import ChatModelParameters
from beeai_framework.memory import UnconstrainedMemory

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

server = Server()

# Create server parameters for MCP Everything server
server_params = StdioServerParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-everything"],
    env={
        "PATH": os.getenv("PATH", default=""),
    },
)

# Global variable to store MCP tools
mcp_tools: List[MCPTool] = []

async def initialize_selected_mcp_tools():
    """Initialize all MCP tools from everything server"""
    try:
        logger.info("Connecting to MCP Everything server...")
        client = stdio_client(server_params)
        
        # Get all tools at once
        logger.info("Loading all MCP tools...")
        all_tools = await MCPTool.from_client(client)
        
        # Use all available tools
        for tool in all_tools:
            logger.info(f"Loaded MCP tool: {tool.name} - {tool.description}")
        
        logger.info(f"Successfully loaded {len(all_tools)} MCP tools")
        return all_tools
    except Exception as e:
        logger.error(f"Failed to initialize MCP tools: {e}")
        return []

@server.agent()
async def everything_agent(
    input: Message,
    llm: Annotated[
        LLMServiceExtensionServer,
        LLMServiceExtensionSpec.single_demand(suggested=("ibm/granite-3-3-8b-instruct", "llama3.1", "gpt-4o-mini"))
    ],
):
    """Advanced RequirementAgent using all MCP everything server tools with LLM integration"""
    
    # Initialize all MCP tools
    all_tools = await initialize_selected_mcp_tools()
    if not all_tools:
        yield AgentMessage(text="❌ Failed to connect to MCP Everything server")
        return
    
    user_message = get_message_text(input)
    
    try:
        if not llm:
            yield AgentMessage(text="❌ No LLM service available")
            return
            
        # Get LLM configuration from BeeAI platform
        llm_config = llm.data.llm_fulfillments.get("default")
        
        # Create OpenAI chat model instance with platform configuration
        chat_model = OpenAIChatModel(
            model_id=llm_config.api_model,
            base_url=llm_config.api_base,
            api_key=llm_config.api_key,
            parameters=ChatModelParameters(temperature=0.0),
            #tool_choice_support=set()
        )
        print(f"🔧 Loaded {len(all_tools)} MCP tools")
        
        # Create ToolCallingAgent with all MCP tools and custom system prompt
        tool_calling_agent = ToolCallingAgent(
            llm=chat_model,
            memory=UnconstrainedMemory(),
            tools=all_tools,  # All MCP tools from everything server
            templates={
                "system": lambda template: template.update(
                    defaults={
                        "instructions": f"""You are an advanced agent with access to {len(all_tools)} MCP Everything server tools.

Available tools:
- echo: Repeat messages back to the user
- add: Perform mathematical addition of two numbers
- printEnv: Show environment variables for debugging
- listRoots: Display available MCP roots
- sampleLLM: Generate AI responses using MCP's LLM sampling
- longRunningOperation: Demonstrate progress updates for long tasks
- getTinyImage: Generate small test images
- annotatedMessage: Create messages with metadata annotations
- getResourceReference: Retrieve resource references
- startElicitation: Interactive user preference collection
- structuredContent: Return structured data with schemas

IMPORTANT: Always select and use the most appropriate MCP tools based on the user's request. For math problems, use 'add'. For echoing text, use 'echo'. For images, use 'getTinyImage'. Be intelligent about tool selection.""",
                    }
                )
            }
        )
        
        # Run the tool calling agent - it will intelligently select MCP tools
        yield AgentMessage(text="🤖 **ToolCallingAgent with MCP Everything Server**")
        response = await tool_calling_agent.run(user_message)
        
        # Stream the response from ToolCallingAgent
        if hasattr(response, 'last_message') and hasattr(response.last_message, 'text'):
            yield AgentMessage(text=response.last_message.text)
        elif hasattr(response, 'text'):
            yield AgentMessage(text=response.text)
        else:
            yield AgentMessage(text=f"Agent response: {str(response)}")
        
        # Show available tools info
        tools_info = f"\n\n**Available MCP Tools ({len(all_tools)}):**\n" + \
                    "\n".join([f"• {tool.name}: {tool.description}" for tool in all_tools])
        yield AgentMessage(text=tools_info)
        
    except Exception as e:
        import traceback
        logger.error(f"Error in everything_agent: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        yield AgentMessage(text=f"❌ Error: {str(e)}")
        yield AgentMessage(text=f"Details: {traceback.format_exc()}")

def run():
    server.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", 8000)))

if __name__ == "__main__":
    run()