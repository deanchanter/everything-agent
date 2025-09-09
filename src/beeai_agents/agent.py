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
from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.adapters.openai import OpenAIChatModel
from beeai_framework.backend.types import ChatModelParameters

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

server = Server()

# Create server parameters for MCP Everything server
server_params = StdioServerParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-everything"],
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
        LLMServiceExtensionSpec.single_demand()
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
            
        print(f"🔧 Loaded {len(all_tools)} MCP tools")
        
        # Get LLM configuration from BeeAI platform
        llm_config = llm.data.llm_fulfillments.get("default")
        
        # Create OpenAI chat model instance with platform configuration
        chat_model = OpenAIChatModel(
            model_id=llm_config.api_model,
            base_url=llm_config.api_base,
            api_key=llm_config.api_key,
            parameters=ChatModelParameters(temperature=0.0),
            tool_choice_support=set()
        )
        
        # Create RequirementAgent with all MCP tools
        requirement_agent = RequirementAgent(
            llm=chat_model,
            tools=all_tools,  # All MCP tools from everything server
        )
        
        # Run the requirement agent - it will intelligently select MCP tools
        yield AgentMessage(text="🤖 **RequirementAgent with MCP Everything Server**")
        response = await requirement_agent.run(user_message)
        
        # Stream the response from RequirementAgent
        if hasattr(response, 'answer') and hasattr(response.answer, 'text'):
            yield AgentMessage(text=response.answer.text)
        elif hasattr(response, 'text'):
            yield AgentMessage(text=response.text)
        else:
            yield AgentMessage(text=f"Agent response: {str(response)}")
        
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