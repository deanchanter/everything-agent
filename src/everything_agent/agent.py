import asyncio
import base64
import logging
import os
import traceback
import uuid
from textwrap import dedent
from typing import Annotated, List

from dotenv import load_dotenv
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client

from a2a.types import Message, TaskState, TaskStatus, TextPart
from a2a.utils.message import get_message_text
from beeai_framework.adapters.openai import OpenAIChatModel
from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.agents.experimental.requirements.conditional import ConditionalRequirement
from beeai_framework.agents.types import AgentExecutionConfig
from beeai_framework.backend.types import ChatModelParameters
from beeai_framework.tools.mcp import MCPTool
from beeai_framework.tools.think import ThinkTool
from beeai_sdk.a2a.extensions import (
    AgentDetail,
    AgentDetailTool,
    AgentDetailContributor,
    LLMServiceExtensionServer,
    LLMServiceExtensionSpec,
    TrajectoryExtensionServer,
    TrajectoryExtensionSpec,
)
from beeai_sdk.a2a.types import AgentMessage
from beeai_sdk.platform.file import File
from beeai_sdk.server import Server
from beeai_sdk.server.agent import AgentSkill

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
        
        logger.info("Loading all MCP tools...")
        all_tools = await MCPTool.from_client(client)
        
        for tool in all_tools:
            logger.info(f"Loaded MCP tool: {tool.name} - {tool.description}")
        
        logger.info(f"Successfully loaded {len(all_tools)} MCP tools")
        return all_tools
    except Exception as e:
        logger.error(f"Failed to initialize MCP tools: {e}")
        return []



async def create_file_from_data(data, filename: str, mime_type: str):
    """Generic helper to create downloadable files from various data formats"""
    try:
        binary_data = base64.b64decode(data) if isinstance(data, str) else data
        file = await File.create(filename=filename, content_type=mime_type, content=binary_data)
        return file.to_file_part()
    except Exception as e:
        logger.error(f"Error creating file {filename}: {e}")
        return None


async def handle_mcp_content(item, tool_name: str):
    """Handle various MCP content types and create downloadable files"""
    item_type = str(type(item))
    
    # Handle ImageContent
    if item_type == "<class 'mcp.types.ImageContent'>":
        image_data = getattr(item, 'data', None)
        if not image_data:
            return None
        mime_type = getattr(item, 'mimeType', None) or getattr(item, 'mime_type', 'image/png')
        filename = f"{tool_name}_tiny_image.png"
        return await create_file_from_data(image_data, filename, mime_type)
    
    # Handle EmbeddedResource
    elif item_type == "<class 'mcp.types.EmbeddedResource'>":
        resource_data = getattr(item, 'resource', None)
        if not resource_data:
            return None
        resource_text = getattr(resource_data, 'text', '')
        if not resource_text:
            return None
        
        resource_name = getattr(resource_data, 'name', f"{tool_name}_resource")
        mime_type = getattr(resource_data, 'mimeType', 'text/plain')
        extension = ".txt" if mime_type == 'text/plain' else ".bin"
        filename = f"{resource_name.replace(' ', '_')}{extension}"
        
        return await create_file_from_data(resource_text.encode('utf-8'), filename, mime_type)
    
    return None

@server.agent(
    name="Everything Agent",
    default_input_modes=["text",],
    default_output_modes=["text", "text/plain", "image/png"],
    detail=AgentDetail(
        interaction_mode="multi-turn",
        user_greeting="Hi! I'm your Everything Agent powered by MCP Everything Server tools. I can help you with calculations, file operations, and much more!",
        version="0.1.0",
        tools=[
            AgentDetailTool(
                name="Think", 
                description="Advanced reasoning and analysis to think through problems step-by-step before taking action."
            ),
            AgentDetailTool(
                name="Echo", 
                description="Echo back messages to test communication and validate input."
            ),
            AgentDetailTool(
                name="Add", 
                description="Perform mathematical addition operations on numbers."
            ),
            AgentDetailTool(
                name="Long Running Operation", 
                description="Execute long-running tasks with progress notifications and status updates."
            ),
            AgentDetailTool(
                name="Sample LLM", 
                description="Generate sample LLM responses for testing and demonstration purposes."
            ),
            AgentDetailTool(
                name="Get Tiny Image", 
                description="Retrieve small test images for visual content demonstration."
            ),
            AgentDetailTool(
                name="Annotated Message", 
                description="Create messages with content annotations and structured formatting."
            ),
            AgentDetailTool(
                name="Resource Reference", 
                description="Get resource references by ID for content management and linking."
            ),
            AgentDetailTool(
                name="Start Elicitation", 
                description="Initiate interactive elicitation processes for gathering user input."
            )
        ],
        framework="BeeAI",
        source_code_url="https://github.com/deanchanter/everything-agent",
        container_image_url="ghcr.io/deanchanter/everything-agent/my-agent",
        author=AgentDetailContributor(
            name="Dean Chanter",
            email="dean.chanter@ibm.com",
        ),
    ),
    skills=[
        AgentSkill(
            id="everything",
            name="Everything Operations",
            description=dedent(
                """\
                A comprehensive agent that leverages MCP Everything Server tools to perform a wide variety of tasks
                including mathematical operations, content processing, file handling, and interactive operations.
                The agent thinks through problems systematically and uses appropriate tools to accomplish tasks efficiently.
                """
            ),
            tags=["MCP", "Math", "Content", "Files", "Interactive", "Testing"],
            examples=[
                "Add 5 and 3 together",
                "Echo back my message: Hello World",
                "Run a long running operation and show me progress",
                "Get a tiny test image",
                "Get me a sample text resource to download",
                "Show me a sample LLM response",
                "Create an annotated message with examples",
                "Start an interactive elicitation process",
                "Get a resource reference for ID 42",
                "Test the simple prompt feature",
                "Try the complex prompt with temperature 0.7 and casual style",
                "Demonstrate the capabilities of the Everything Server"
            ]
        )
    ]
)
async def everything_agent(
    input: Message,
    llm: Annotated[
        LLMServiceExtensionServer,
        LLMServiceExtensionSpec.single_demand(suggested=("granite-3-3-8b", "openai/gpt-4o", "anthropic/claude-4-sonnet"))
    ],
    trajectory: Annotated[
        TrajectoryExtensionServer,
        TrajectoryExtensionSpec()
    ],
):
    """Everything Agent that uses MCP Everything Server tools to perform various tasks"""
    
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
                
                
        # Use native tool calling for Claude, response format fallback for others
        use_response_format_fallback = 'claude' not in llm_config.api_model.lower()
        
        chat_model = OpenAIChatModel(
            model_id=llm_config.api_model,
            base_url=llm_config.api_base,
            api_key=llm_config.api_key,
            tool_call_fallback_via_response_format=use_response_format_fallback,
            parameters=ChatModelParameters(temperature=0.1),
            tool_choice_support=set(),  # Enable tool choice support
        )

        # Create conditional requirements
        requirements = []
        if 'claude' not in llm_config.api_model.lower():
            # Create think tool and combine with MCP tools
            think_tool = ThinkTool()
            all_tools.append(think_tool)

            # Force think tool to be called first
            think_requirement = ConditionalRequirement(
                target=think_tool,
                name="think_first",
                min_invocations=1,
                priority=100,
                force_at_step=1,  # Force it to be the first step
            )
            requirements.append(think_requirement)
        
        # Find specific tools for special requirements
        resource_links_tool = None
        resource_reference_tool = None
        final_answer_tool = None
        other_tools = []
        
        for mcp_tool in all_tools:
            if "getResourceLinks" in mcp_tool.name:
                resource_links_tool = mcp_tool
            elif "getResourceReference" in mcp_tool.name:
                resource_reference_tool = mcp_tool
            elif mcp_tool.name == "final_answer":
                final_answer_tool = mcp_tool
            elif mcp_tool.name != "think":  # Exclude think tool from other tools
                other_tools.append(mcp_tool)

        # Requirement: getResourceReference runs after getResourceLinks
        if resource_links_tool and resource_reference_tool:
            resource_reference_requirement = ConditionalRequirement(
                target=resource_reference_tool,
                name="resource_reference_after_links",
                force_after=[resource_links_tool],
                priority=99
            )
            requirements.append(resource_reference_requirement)
        
        # Requirement: All tools other than getResourceLinks and getResourceReference should call final_answer after
        # if final_answer_tool and other_tools:
        #     for tool in other_tools:
        #         # Skip getResourceLinks and getResourceReference tools
        #         if ("getResourceLinks" not in tool.name and 
        #             "getResourceReference" not in tool.name):
        #             final_answer_requirement = ConditionalRequirement(
        #                 target=final_answer_tool,
        #                 name=f"final_answer_after_{tool.name}",
        #                 force_after=[tool],
        #                 priority=98
        #             )
        #             requirements.append(final_answer_requirement)
        
        # Create RequirementAgent with conditional requirements
        requirement_agent = RequirementAgent(
            llm=chat_model,
            tools=all_tools,
            requirements=requirements,
            instructions= dedent("""\
                You are an Everything Agent that uses MCP Everything Server tools to perform various tasks.
                Think through problems step-by-step and use the appropriate tools to accomplish tasks efficiently.
                Use the tools provided to you, and follow the requirements specified.
                If you need to think, use the Think tool first.
                If you need to get resource links, use the getResourceLinks tool.
                If you need to reference a resource, use the getResourceReference tool.
                Valide Resources are 1-100.
                Always provide a final answer using the final_answer tool when available.
            """),
        )
        
        # Run the requirement agent with yielding pattern
        yield AgentMessage(text="🤖 **RequirementAgent with MCP Everything Server**")
        
        response_text = ""
        
        # Stream execution with tool updates
        async for event, meta in requirement_agent.run(
            user_message,
            execution=AgentExecutionConfig(max_iterations=20, max_retries_per_step=2, total_max_retries=5)
        ):
            if meta.name == "success" and event.state.steps:
                step = event.state.steps[-1]
                if not step.tool:
                    continue
                
                tool_name = step.tool.name
                
                
                # Handle final answer
                if tool_name == "final_answer":
                    response_text += step.input.get("response", "")
                           
                # Handle thinking steps
                elif tool_name == "think":
                    thoughts = step.input.get("thoughts", "")
                    if thoughts:
                        yield trajectory.trajectory_metadata(
                            title="💭 Thinking",
                            content=thoughts
                        )
                
                # Handle resource tools
                elif "resource" in tool_name.lower():
                    yield trajectory.trajectory_metadata(
                        title=f"📄 {tool_name}",
                        content=f"Processing resource with {tool_name}..."
                    )
                    
                    if hasattr(step, 'output') and step.output and hasattr(step.output, 'result') and isinstance(step.output.result, list):
                        for item in step.output.result:
                            # Only create files from embedded resources (actual content), not resource links (metadata)
                            if "getResourceLinks" in tool_name:
                                # Skip creating files for getResourceLinks - it just lists resources
                                continue
                            else:
                                # Handle any MCP content type
                                file_part = await handle_mcp_content(item, tool_name)
                                if file_part:
                                    yield file_part

                # Handle image tools
                elif "image" in tool_name.lower() or "tinyimage" in tool_name.lower():
                    yield trajectory.trajectory_metadata(
                        title=f"📁 {tool_name}",
                        content=f"Generating file/image with {tool_name}..."
                    )
                    
                    # Check if the tool output has files to download
                    if hasattr(step, 'output') and step.output:
                        # Handle JSONToolOutput with result array (MCP Everything Server format)
                        if hasattr(step.output, 'result') and isinstance(step.output.result, list):
                            for item in step.output.result:
                                # Handle MCP content objects - create downloadable files
                                file_part = await handle_mcp_content(item, tool_name)
                                if file_part:
                                    yield file_part
                     
                # Handle long running operations with task status updates
                elif "longrunningoperation" in tool_name.lower() or "long_running" in tool_name.lower():
                    yield trajectory.trajectory_metadata(
                        title="⏳ Long Running Operation",
                        content="Starting long running operation..."
                    )
                    
                    # Provide task status updates for long running operations
                    yield TaskStatus(
                        message=Message(
                            message_id=str(uuid.uuid4()),
                            role="agent", 
                            parts=[TextPart(text="\n\n🔄 Long running operation in progress... Please wait.")]
                        ),
                        state=TaskState.working,
                    )
                
                # Handle other tool calls
                else:
                    yield trajectory.trajectory_metadata(
                        title=f"🔧 {tool_name}",
                        content=f"Executing {tool_name}..."
                    )
        
        # Yield final response
        if response_text:
            yield AgentMessage(text=f"\n\n{response_text}")
        else:
            yield AgentMessage(text="\n\n✅ Task completed")
        
    except Exception as e:
        logger.error(f"Error in everything_agent: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        yield AgentMessage(text=f"❌ Error: {str(e)}")
        yield AgentMessage(text=f"Details: {traceback.format_exc()}")

def run():
    server.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", 8000)))

if __name__ == "__main__":
    run()