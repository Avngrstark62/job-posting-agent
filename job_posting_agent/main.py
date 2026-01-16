# |---------------------------------------------------------|
# |                                                         |
# |                 Give Feedback / Get Help                |
# | https://github.com/getbindu/Bindu/issues/new/choose    |
# |                                                         |
# |---------------------------------------------------------|
#
#  Thank you users! We ❤️ you! - 🌻

"""job-posting-agent - A Bindu Agent for creating compelling job postings."""

import argparse
import asyncio
import json
import os
import re
import sys
import traceback
from pathlib import Path
from textwrap import dedent
from typing import Any

from bindu.penguin.bindufy import bindufy
from crewai import LLM, Agent, Crew, Process, Task
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Global variables
crew: Crew | None = None
_initialized = False
_init_lock = asyncio.Lock()


class CrewInitializationError(Exception):
    """Exception raised when crew initialization fails."""

    pass


class CrewExecutionError(Exception):
    """Exception raised when crew execution fails."""

    CREW_NOT_INITIALIZED = "Crew not initialized"


def _raise_api_key_error() -> None:
    """Raise an error when API keys are not configured."""
    error_msg = (
        "No API key provided. Set OPENAI_API_KEY or OPENROUTER_API_KEY environment variable.\n"
        "For OpenRouter: https://openrouter.ai/keys\n"
        "For OpenAI: https://platform.openai.com/api-keys"
    )
    raise CrewInitializationError(error_msg)


def _is_complete_posting(result_str: str, markers: list[str]) -> bool:
    """Check if result string contains a complete job posting."""
    return len(result_str) > 200 and any(m in result_str for m in markers)


def _extract_task_outputs(result: Any) -> list[Any]:
    """Extract task outputs from CrewAI result."""
    if hasattr(result, "tasks_output") and result.tasks_output:
        return result.tasks_output
    if hasattr(result, "raw") and hasattr(result.raw, "tasks_output") and result.raw.tasks_output:
        return result.raw.tasks_output
    if isinstance(result, list):
        return result
    return []


def _find_clean_output(task_outputs: list[Any], markers: list[str]) -> str | None:
    """Find clean output from task outputs."""
    for output in reversed(task_outputs):
        if isinstance(output, str) and len(output) > 100:
            clean_output = output.strip()
            if clean_output.startswith("```"):
                clean_output = re.sub(r"^```\w*\n?|\n?```$", "", clean_output, flags=re.MULTILINE)
            if any(m in clean_output for m in markers) and len(clean_output) > 200:
                return clean_output
    return None


def _extract_code_block(result_str: str, markers: list[str]) -> str | None:
    """Extract code block from result string."""
    code_blocks = re.findall(r"```(?:.*?\n)?(.*?)```", result_str, re.DOTALL)
    for block in code_blocks:
        clean_block = block.strip()
        if len(clean_block) > 200 and any(m in clean_block for m in markers):
            return clean_block
    return None


def _extract_from_outputs_or_blocks(result: Any, result_str: str, markers: list[str]) -> str:
    """Extract from task outputs or code blocks, with fallback."""
    task_outputs = _extract_task_outputs(result)
    clean_output = _find_clean_output(task_outputs, markers)
    if clean_output:
        return clean_output

    code_block = _extract_code_block(result_str, markers)
    if code_block:
        return code_block

    return result_str


def load_config() -> dict[str, Any]:
    """Load agent configuration from project root."""
    possible_paths = [
        Path(__file__).parent.parent / "agent_config.json",
        Path(__file__).parent / "agent_config.json",
        Path.cwd() / "agent_config.json",
    ]

    for config_path in possible_paths:
        if config_path.exists():
            try:
                with open(config_path) as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  Error reading {config_path}: {type(e).__name__}")
                continue

    print("⚠️  No agent_config.json found, using default configuration")
    return {
        "name": "job-posting-agent",
        "description": "AI job posting agent for creating compelling job descriptions",
        "version": "1.0.0",
        "deployment": {
            "url": "http://127.0.0.1:3773",
            "expose": True,
            "protocol_version": "1.0.0",
            "proxy_urls": ["127.0.0.1"],
            "cors_origins": ["*"],
        },
        "environment_variables": [
            {"key": "OPENAI_API_KEY", "description": "OpenAI API key for LLM calls", "required": False},
            {"key": "OPENROUTER_API_KEY", "description": "OpenRouter API key for LLM calls", "required": True},
            {"key": "MEM0_API_KEY", "description": "Mem0 API key for memory operations", "required": False},
        ],
    }


async def initialize_crew() -> None:
    """Initialize the job posting crew with model and agents."""
    global crew

    openai_api_key = os.getenv("OPENAI_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    model_name = os.getenv("MODEL_NAME", "openai/gpt-4o")

    llm: Any
    try:
        if openai_api_key and not openrouter_api_key:
            llm = LLM(model="gpt-4o", api_key=openai_api_key, temperature=0.7)
            print("✅ Using OpenAI GPT-4o directly")

        elif openrouter_api_key:
            llm = LLM(
                model=model_name,
                api_key=openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
                temperature=0.7,
            )
            print(f"✅ Using OpenRouter via CrewAI LLM: {model_name}")

            if not os.getenv("OPENAI_API_KEY"):
                os.environ["OPENAI_API_KEY"] = openrouter_api_key

        else:
            _raise_api_key_error()

    except Exception as e:
        print(f"❌ LLM initialization error: {e}")
        print("⚠️ Using mock LLM for testing only")

        class MockLLM:
            def __call__(self, *args: Any, **kwargs: Any) -> str:
                return "Mock response for testing"

        llm = MockLLM()

    # Define Agents for job posting workflow
    research_agent = Agent(
        role="Research Analyst",
        goal="Analyze the company website and description to extract insights on culture, values, and specific needs",
        backstory=dedent(
            """
            You are an expert in analyzing company cultures and identifying key values
            and needs from various sources, including websites and brief descriptions.
            You excel at understanding what makes a company unique and attractive to potential candidates.
            """
        ),
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )

    writer_agent = Agent(
        role="Job Description Writer",
        goal="Use insights from research to create detailed, engaging, and enticing job postings",
        backstory=dedent(
            """
            You are skilled in crafting compelling job descriptions that resonate
            with the company's values and attract the right candidates. You know how
            to highlight key responsibilities, requirements, and benefits in a way
            that makes the position irresistible to top talent.
            """
        ),
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )

    review_agent = Agent(
        role="Review and Editing Specialist",
        goal="Review job postings for clarity, engagement, grammatical accuracy, and alignment with company values",
        backstory=dedent(
            """
            You are a meticulous editor with a keen eye for detail and a deep understanding
            of effective communication. You ensure job postings are polished, professional,
            and perfectly aligned with company culture while being attractive to candidates.
            """
        ),
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )

    # Define Tasks
    research_company_culture_task = Task(
        description=dedent(
            """
            Analyze the company information provided: {company_description}
            Company domain: {company_domain}

            Focus on understanding:
            1. Company culture, values, and mission
            2. Unique selling points and achievements
            3. What makes this company attractive to candidates

            Compile a comprehensive report summarizing these insights, specifically
            how they can be leveraged in a job posting to attract the right candidates.

            IMPORTANT: Return only the research insights.
            """
        ),
        expected_output="A comprehensive report detailing the company's culture, values, mission, and unique selling points.",
        agent=research_agent,
    )

    research_role_requirements_task = Task(
        description=dedent(
            """
            Based on the hiring needs: {hiring_needs}, identify the key skills,
            experiences, and qualities the ideal candidate should possess.

            Consider:
            1. Technical skills and qualifications required
            2. Soft skills and personal qualities
            3. Experience level and background
            4. Industry-specific knowledge needed

            Prepare a detailed list of recommended job requirements and qualifications
            that align with the company's needs and values.

            IMPORTANT: Return only the requirements analysis.
            """
        ),
        expected_output="A detailed list of recommended skills, experiences, and qualities for the ideal candidate.",
        agent=research_agent,
        context=[research_company_culture_task],
    )

    draft_job_posting_task = Task(
        description=dedent(
            """
            Draft a comprehensive job posting for the role: {hiring_needs}

            Structure the posting with:
            1. Compelling introduction about the company
            2. Detailed role description and key responsibilities
            3. Required skills and qualifications
            4. Company benefits and unique opportunities: {specific_benefits}
            5. Application instructions

            Ensure the tone aligns with the company culture and incorporates
            the insights from the research. Make it engaging and attractive
            to top talent.

            IMPORTANT: Return the complete draft job posting in markdown format.
            """
        ),
        expected_output="A detailed, engaging job posting with introduction, role description, responsibilities, requirements, and benefits.",
        agent=writer_agent,
        context=[research_role_requirements_task],
    )

    review_and_edit_job_posting_task = Task(
        description=dedent(
            """
            Review and refine the draft job posting for the role: {hiring_needs}

            Check for:
            1. Clarity and readability
            2. Grammatical accuracy and professional tone
            3. Alignment with company culture and values
            4. Engagement and appeal to target candidates
            5. Completeness of information

            Edit and polish the content to ensure it speaks directly to the
            desired candidates and accurately reflects the role's unique
            benefits and opportunities.

            IMPORTANT: Return the final, polished job posting in markdown format.
            """
        ),
        expected_output="A polished, error-free job posting that is clear, engaging, and perfectly aligned with company culture and values. Formatted in markdown.",
        agent=review_agent,
        context=[draft_job_posting_task],
    )

    # Create the crew with the review task as the final output
    crew = Crew(
        agents=[research_agent, writer_agent, review_agent],
        tasks=[
            research_company_culture_task,
            research_role_requirements_task,
            draft_job_posting_task,
            review_and_edit_job_posting_task,
        ],
        verbose=True,
        process=Process.sequential,
        memory=False,
    )

    print("✅ Job Posting Crew initialized")


def extract_job_posting(result: Any) -> str:
    """Extract job posting from CrewAI result formats."""
    result_str = str(result)
    markers = ["Job Title", "Position", "Role", "Responsibilities", "Requirements", "Qualifications", "Benefits", "About"]

    # If looks like a complete job posting, return it
    if _is_complete_posting(result_str, markers):
        return result_str

    # Try task outputs and code blocks
    return _extract_from_outputs_or_blocks(result, result_str, markers)


async def run_crew(company_description: str, company_domain: str, hiring_needs: str, specific_benefits: str) -> str:
    """Run the crew and return the final job posting."""
    global crew

    if not crew:
        raise CrewExecutionError(CrewExecutionError.CREW_NOT_INITIALIZED)

    print(f"📝 Running job posting crew for: {hiring_needs}")
    try:
        result = crew.kickoff(
            inputs={
                "company_description": company_description,
                "company_domain": company_domain,
                "hiring_needs": hiring_needs,
                "specific_benefits": specific_benefits,
            }
        )
    except Exception as e:
        error_msg = f"Crew execution failed: {e!s}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        return f"## Error\n\n{error_msg}\n\nPlease try again with different input."

    job_posting = extract_job_posting(result)
    job_posting = job_posting.strip()
    print(f"📊 Final job posting length: {len(job_posting)} chars")
    return job_posting


async def handler(messages: list[dict[str, str]]) -> str:
    """Handle incoming agent messages."""
    global _initialized

    # Lazy initialization
    async with _init_lock:
        if not _initialized:
            print("🔧 Initializing Job Posting Crew...")
            await initialize_crew()
            _initialized = True

    # Extract user input
    user_input = ""
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                user_input = msg.get("content", "").strip()
                break

    if not user_input:
        return (
            "Please provide job posting details in the following format:\n\n"
            "Company Description: [Brief description of the company]\n"
            "Company Domain: [company website or domain]\n"
            "Hiring Needs: [Role title and key requirements]\n"
            "Specific Benefits: [Unique benefits and perks]\n\n"
            "Example:\n"
            "Company Description: Fast-growing tech startup focused on AI solutions\n"
            "Company Domain: example.com\n"
            "Hiring Needs: Senior Software Engineer with Python and ML experience\n"
            "Specific Benefits: Remote work, equity, unlimited PTO"
        )

    print(f"✅ Processing: {user_input}")

    # Parse the input to extract components
    company_description = "Innovative technology company"
    company_domain = "example.com"
    hiring_needs = user_input
    specific_benefits = "Competitive salary, health benefits, professional development opportunities"

    # Try to parse structured input if provided
    lines = user_input.split("\n")
    for line in lines:
        if "company description:" in line.lower():
            company_description = line.split(":", 1)[1].strip()
        elif "company domain:" in line.lower():
            company_domain = line.split(":", 1)[1].strip()
        elif "hiring needs:" in line.lower():
            hiring_needs = line.split(":", 1)[1].strip()
        elif "specific benefits:" in line.lower() or "benefits:" in line.lower():
            specific_benefits = line.split(":", 1)[1].strip()

    try:
        job_posting = await run_crew(company_description, company_domain, hiring_needs, specific_benefits)
        if job_posting and len(job_posting) > 100:
            print(f"✅ Success! Generated job posting ({len(job_posting)} chars)")
            return job_posting
        else:
            print("⚠️ Generated content may be incomplete")
            return "I couldn't generate a complete job posting. Please try providing more detailed information."
    except Exception as e:
        error_msg = f"Handler error: {e!s}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        return f"## Error\n\n{error_msg}\n\nPlease check your API keys and try again."


async def cleanup() -> None:
    """Clean up resources."""
    global crew
    print("🧹 Cleaning up Job Posting Crew resources...")
    if crew:
        crew = None
    print("✅ Cleanup complete")


def main():
    """Run the main entry point for the Job Posting Agent."""
    parser = argparse.ArgumentParser(description="Bindu Job Posting Agent")
    parser.add_argument(
        "--openrouter-api-key",
        type=str,
        default=os.getenv("OPENROUTER_API_KEY"),
        help="OpenRouter API key (env: OPENROUTER_API_KEY)",
    )
    parser.add_argument(
        "--mem0-api-key",
        type=str,
        default=os.getenv("MEM0_API_KEY"),
        help="Mem0 API key (env: MEM0_API_KEY)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("MODEL_NAME", "openai/gpt-4o-mini"),
        help="Model ID for OpenRouter (env: MODEL_NAME)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to agent_config.json (optional)",
    )
    args = parser.parse_args()

    # Set environment variables from CLI
    if args.openrouter_api_key:
        os.environ["OPENROUTER_API_KEY"] = args.openrouter_api_key
        if not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = args.openrouter_api_key
    if args.mem0_api_key:
        os.environ["MEM0_API_KEY"] = args.mem0_api_key
    if args.model:
        os.environ["MODEL_NAME"] = args.model

    print("🤖 Job Posting Agent - Creating compelling job descriptions")
    print("📝 Capabilities: Company research, role analysis, job posting creation, review & editing")
    print("⚙️ Process: 3-agent crew with sequential workflow")

    # Load configuration
    config = load_config()

    try:
        print("🚀 Starting Bindu Job Posting Agent server...")
        print(f"🌐 Server will run on: {config.get('deployment', {}).get('url', 'http://127.0.0.1:3773')}")
        bindufy(config, handler)
    except KeyboardInterrupt:
        print("\n🛑 Job Posting Agent stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        asyncio.run(cleanup())


if __name__ == "__main__":
    main()
