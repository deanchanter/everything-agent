import asyncio
import base64
import logging
import os
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
    LLMServiceExtensionServer,
    LLMServiceExtensionSpec,
    TrajectoryExtensionServer,
    TrajectoryExtensionSpec,
)
from beeai_sdk.a2a.types import AgentArtifact, AgentMessage, FilePart, FileWithBytes
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


def create_file_artifact(filename: str, content: str, mime_type: str) -> AgentArtifact:
    """Create a downloadable file artifact from content"""
    try:
        # Convert content to base64
        content_bytes = content.encode('utf-8')
        base64_content = base64.b64encode(content_bytes).decode('utf-8')
        
        file_part = FilePart(
            file=FileWithBytes(
                name=filename,
                bytes=base64_content,
                mime_type=mime_type
            )
        )
        return AgentArtifact(name=filename, parts=[file_part])
    except Exception as e:
        logger.error(f"Error creating file artifact {filename}: {e}")
        return None


def handle_mcp_image_content(item, tool_name: str):
    """Handle MCP ImageContent objects and create downloadable artifacts"""
    if str(type(item)) != "<class 'mcp.types.ImageContent'>":
        return None
        
    print(f"🖼️ DEBUG - Found MCP ImageContent object!")
    image_data = getattr(item, 'data', None)
    mime_type = getattr(item, 'mimeType', None) or getattr(item, 'mime_type', 'image/png')
    
    print(f"🖼️ DEBUG - Image data length: {len(image_data) if image_data else 'None'}")
    print(f"🖼️ DEBUG - MIME type: {mime_type}")
    
    if image_data:
        try:
            # Use base64 decode to get binary data, then create file artifact
            import base64
            binary_data = base64.b64decode(image_data)
            text_data = binary_data.decode('latin-1')  # Use latin-1 to preserve bytes
            
            filename = f"{tool_name}_output.png"
            artifact = create_file_artifact(filename, text_data, mime_type)
            if artifact:
                print(f"🖼️ DEBUG - Created artifact using create_file_artifact: {filename}")
                return artifact
            else:
                print(f"🖼️ DEBUG - Failed to create artifact using create_file_artifact")
        except Exception as e:
            print(f"🖼️ DEBUG - Error creating artifact: {e}")
    
    return None


def handle_mcp_embedded_resource(item, tool_name: str):
    """Handle MCP EmbeddedResource objects and create downloadable artifacts"""
    if str(type(item)) != "<class 'mcp.types.EmbeddedResource'>":
        return None
        
    print(f"📄 DEBUG - Found MCP EmbeddedResource object!")
    resource_data = getattr(item, 'resource', None)
    
    if not resource_data:
        print(f"📄 DEBUG - No resource data found in EmbeddedResource")
        return None
    
    resource_text = getattr(resource_data, 'text', '')
    resource_name = getattr(resource_data, 'name', f"{tool_name}_resource")
    mime_type = getattr(resource_data, 'mimeType', 'text/plain')
    
    print(f"📄 DEBUG - Resource name: {resource_name}")
    print(f"📄 DEBUG - Resource text length: {len(resource_text)}")
    print(f"📄 DEBUG - MIME type: {mime_type}")
    
    if resource_text:
        filename = f"{resource_name.replace(' ', '_')}.txt" if mime_type == 'text/plain' else f"{resource_name.replace(' ', '_')}.bin"
        artifact = create_file_artifact(filename, resource_text, mime_type)
        if artifact:
            print(f"📄 DEBUG - Created artifact: {filename}")
        return artifact
    
    print(f"📄 DEBUG - No resource text found")
    return None


def handle_mcp_resource_link(item, tool_name: str):
    """Handle MCP ResourceLink objects and create metadata files"""
    if str(type(item)) != "<class 'mcp.types.ResourceLink'>":
        return None
        
    print(f"📄 DEBUG - Found MCP ResourceLink object!")
    resource_name = getattr(item, 'name', f"{tool_name}_link")
    resource_uri = getattr(item, 'uri', '')
    resource_description = getattr(item, 'description', '')
    mime_type = getattr(item, 'mimeType', 'text/plain')
    
    print(f"📄 DEBUG - Resource name: {resource_name}")
    print(f"📄 DEBUG - Resource URI: {resource_uri}")
    
    # Create metadata content
    resource_content = f"Resource Link: {resource_name}\nURI: {resource_uri}\nDescription: {resource_description}\nMIME Type: {mime_type}"
    filename = f"{resource_name.replace(' ', '_')}_metadata.txt"
    
    artifact = create_file_artifact(filename, resource_content, 'text/plain')
    if artifact:
        print(f"📄 DEBUG - Created metadata artifact: {filename}")
    return artifact

@server.agent(
    name="Everything Agent",
    default_input_modes=["text", "text/plain"],
    default_output_modes=["text", "text/plain"],
    detail=AgentDetail(
        interaction_mode="multi-turn",
        user_greeting="Hi! I'm your Everything Agent powered by MCP Everything Server tools. I can help you with calculations, file operations, web requests, and much more!",
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
        framework="BeeAI + MCP",
        author={
            "name": "Everything Agent Developer"
        }
    ),
    skills=[
        AgentSkill(
            id="everything",
            name="Everything Operations",
            description=dedent(
                """\
                A comprehensive agent that leverages MCP Everything Server tools to perform a wide variety of tasks
                including mathematical operations, content processing, file handling, web requests, and interactive operations.
                The agent thinks through problems systematically and uses appropriate tools to accomplish tasks efficiently.
                """
            ),
            tags=["Math", "Content", "Files", "Web", "Interactive", "Testing"],
            examples=[
                "Add 5 and 3 together",
                "Echo back my message: Hello World",
                "Run a long running operation and show me progress",
                "Get a tiny test image",
                "Show me a sample LLM response",
                "Create an annotated message with examples",
                "Start an interactive elicitation process",
                "Get a resource reference for ID 42",
                "Test the simple prompt feature",
                "Try the complex prompt with temperature 0.7 and casual style",
                "Help me understand how MCP tools work",
                "Process some data and show me the results",
                "Demonstrate the capabilities of the Everything Server"
            ]
        )
    ]
)
async def everything_agent(
    input: Message,
    llm: Annotated[
        LLMServiceExtensionServer,
        LLMServiceExtensionSpec.single_demand()
    ],
    trajectory: Annotated[
        TrajectoryExtensionServer,
        TrajectoryExtensionSpec()
    ],
):
    """Advanced RequirementAgent using all MCP everything server tools with LLM integration
    
    IMPORTANT VALIDATION RULES:
    - For getResourceReference tool: resourceId must be between 1 and 100 (inclusive)
    - For all resource tools: use valid resource IDs (1-5 are typically available)
    - Always validate tool parameters match the expected schema before calling
    """
    
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
        
        # Create OpenAI chat model instance with platform configuration and validation instructions
        chat_model = OpenAIChatModel(
            model_id=llm_config.api_model,
            base_url=llm_config.api_base,
            api_key=llm_config.api_key,
            parameters=ChatModelParameters(
                temperature=0.0,
                system_message="You are an intelligent agent using MCP Everything Server tools. IMPORTANT: When using getResourceReference tool, resourceId must be between 1-100. When using other resource tools, use valid resource IDs (1-5 are typically available). Always validate tool parameters before making calls."
            ),
            tool_choice_support=set()
        )
        
        # Create think tool and combine with MCP tools
        think_tool = ThinkTool()
        all_agent_tools = [think_tool] + all_tools
        
        print(f"🔧 Loaded {len(all_agent_tools)} tools (1 think tool + {len(all_tools)} MCP tools)")
        
        # Create conditional requirements
        requirements = []
        
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
        
        for mcp_tool in all_tools:
            if "getResourceLinks" in mcp_tool.name:
                resource_links_tool = mcp_tool
            elif "getResourceReference" in mcp_tool.name:
                resource_reference_tool = mcp_tool

        # Requirement: getResourceReference runs after getResourceLinks
        if resource_links_tool and resource_reference_tool:
            resource_reference_requirement = ConditionalRequirement(
                target=resource_reference_tool,
                name="resource_reference_after_links",
                force_after=[resource_links_tool],
                priority=75
            )
            requirements.append(resource_reference_requirement)
            print(f"🔧 DEBUG - Added requirement: {resource_reference_tool.name} must run after {resource_links_tool.name}")
        else:
            print(f"🔧 DEBUG - Could not find required tools: getResourceLinks={resource_links_tool is not None}, getResourceReference={resource_reference_tool is not None}")
        
        # Create RequirementAgent with conditional requirements
        requirement_agent = RequirementAgent(
            llm=chat_model,
            tools=all_agent_tools,  # Think tool + all MCP tools from everything server
            requirements=requirements
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
                
                # Debug: Print tool results for debugging
                print(f"🔍 DEBUG - Tool: {tool_name}")
                print(f"🔍 DEBUG - Step input: {step.input}")
                print(f"🔍 DEBUG - Step output: {step.output}")
                print(f"🔍 DEBUG - Step output type: {type(step.output)}")
                if hasattr(step.output, '__dict__'):
                    print(f"🔍 DEBUG - Step output attributes: {list(step.output.__dict__.keys())}")
                print("=" * 50)
                
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
                                # Only handle embedded resources for file creation
                                artifact = handle_mcp_embedded_resource(item, tool_name)
                                if artifact:
                                    yield artifact

                # Handle image and file-generating tools
                elif "image" in tool_name.lower() or "file" in tool_name.lower():
                    print(f"🖼️ DEBUG - Handling image/file tool: {tool_name}")
                    
                    yield trajectory.trajectory_metadata(
                        title=f"📁 {tool_name}",
                        content=f"Generating file/image with {tool_name}..."
                    )
                    
                    # Check if the tool output has files to download
                    if hasattr(step, 'output') and step.output:
                        print(f"🖼️ DEBUG - Tool has output, checking for file content...")
                        
                        # Handle JSONToolOutput with result array (MCP Everything Server format)
                        if hasattr(step.output, 'result') and isinstance(step.output.result, list):
                            print(f"🖼️ DEBUG - Found result array with {len(step.output.result)} items")
                            for i, item in enumerate(step.output.result):
                                print(f"🖼️ DEBUG - Item {i}: {type(item)} - {item if isinstance(item, dict) else 'Non-dict item'}")
                                
                                # Handle MCP ImageContent objects - create downloadable files only
                                artifact = handle_mcp_image_content(item, tool_name)
                                if artifact:
                                    yield artifact
                                
                                
                        
                        # Handle file creation tool output (original format)
                        elif hasattr(step.output, 'result') and hasattr(step.output.result, 'files'):
                            print(f"🖼️ DEBUG - Found files in result")
                            result = step.output.result
                            for file_info in result.files:
                                print(f"🖼️ DEBUG - Processing file: {file_info.display_filename}")
                                part = file_info.file.to_file_part()
                                part.file.name = file_info.display_filename
                                yield AgentArtifact(name=file_info.display_filename, parts=[part])
                        
                        # Handle direct content (fallback)
                        elif hasattr(step.output, 'content'):
                            print(f"🖼️ DEBUG - Found content in output")
                            content = step.output.content
                            print(f"🖼️ DEBUG - Content type: {type(content)}")
                            
                            if hasattr(content, 'data') or hasattr(content, 'bytes'):
                                print(f"🖼️ DEBUG - Found binary data, creating artifact")
                                from beeai_sdk.a2a.types import FilePart, FileWithBytes
                                file_data = content.data if hasattr(content, 'data') else content.bytes
                                filename = f"{tool_name}_output.png" if "image" in tool_name.lower() else f"{tool_name}_output.bin"
                                
                                file_part = FilePart(
                                    file=FileWithBytes(
                                        name=filename,
                                        bytes=file_data,
                                        mime_type="image/png" if "image" in tool_name.lower() else "application/octet-stream"
                                    )
                                )
                                yield AgentArtifact(name=filename, parts=[file_part])
                                print(f"🖼️ DEBUG - Created artifact: {filename}")
                            else:
                                print(f"🖼️ DEBUG - No binary data found in content")
                        else:
                            print(f"🖼️ DEBUG - No recognized file format found in output")
                    else:
                        print(f"🖼️ DEBUG - Tool has no output")
                
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
        import traceback
        logger.error(f"Error in everything_agent: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        yield AgentMessage(text=f"❌ Error: {str(e)}")
        yield AgentMessage(text=f"Details: {traceback.format_exc()}")

def run():
    server.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", 8000)))

if __name__ == "__main__":
    run()