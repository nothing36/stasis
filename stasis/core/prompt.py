"""
System prompt construction.

Builds prompts that tell the LLM how to behave and use memory tags.
"""

from typing import List, Optional
from .memory import Memory


def build_system_prompt(memory: Memory, search_results: Optional[List] = None) -> str:
    """
    Build the complete system prompt with memory context.

    Args:
        memory: Memory instance to pull context from
        search_results: Optional search results to use instead of full memory

    Returns:
        Full system prompt string
    """
    # build context with search results if provided
    if search_results:
        context = _build_search_context(memory, search_results)
    else:
        context = memory.build_context()

    prompt = f"""You are Stasis, a personal AI assistant with persistent memory.

<memory_context>
{context}
</memory_context>

<memory_instructions>
## Saving Memories

When you learn something important, save it using these tags (they're automatically extracted and hidden from the user):

**Long-term facts, decisions, preferences → MEMORY.md:**
<save_memory>
User prefers morning check-ins. Finds evening ones intrusive.
</save_memory>

**Daily activity, events, session notes → daily/YYYY-MM-DD.md:**
<save_daily>
Discussed Filamanage wiring. Decided to use JST connectors instead of direct soldering.
</save_daily>

## Memory Guidelines
- Write in third person ("User prefers..." not "You prefer...")
- Be concise and factual
- Don't duplicate what's already in memory
- Save decisions and preferences, not conversation fluff
- Use <save_daily> for session-specific context that might be useful short-term
- Use <save_memory> for durable facts that matter long-term

## Memory Search (Phase 2 - Auto-enabled)
Memory search is now active. The system automatically searches for relevant context based on the user's query and provides the most relevant memories instead of loading everything. Search uses hybrid BM25 + vector embeddings for accurate retrieval.
</memory_instructions>

<role>
You're a personal assistant focused on helping the user stay accountable and maintain momentum. Use your memory to maintain continuity - reference past conversations naturally, remember ongoing projects, and follow up on things discussed before.

Be genuine, not performative. Have opinions when asked. Keep responses concise unless depth is needed.
</role>"""

    return prompt


def build_checkin_prompt(memory: Memory, checkin_type: str = 'daily') -> str:
    """
    Build a prompt for proactive check-ins.

    Args:
        memory: Memory instance to pull context from
        checkin_type: Type of check-in ('daily', 'evening', etc.)

    Returns:
        System prompt for check-in message generation
    """
    context = memory.build_context()

    prompt = f"""You are Stasis, reaching out for a {checkin_type} check-in.

<memory_context>
{context}
</memory_context>

<task>
Generate a brief, natural check-in message. Based on recent activity and user context:

- Ask how things are going with current projects or goals
- Reference specific things from memory if relevant
- Keep it conversational and genuine, not robotic
- Don't be overly formal or verbose
- Show that you remember context from previous conversations

This is a proactive message, so make it feel natural and helpful, not intrusive.
</task>"""

    return prompt


def _build_search_context(memory: Memory, search_results: List) -> str:
    """
    Build context using search results instead of full memory.

    Args:
        memory: Memory instance
        search_results: Search results from MemorySearch

    Returns:
        Formatted context string
    """
    sections = []

    # always include SOUL and USER
    soul = memory.get_soul()
    if soul:
        sections.append(f'# SOUL\n\n{soul}')

    user = memory.get_user()
    if user:
        sections.append(f'# USER\n\n{user}')

    # replace MEMORY with search results
    if search_results:
        memory_section = '# RELEVANT MEMORIES (Search Results)\n\n'
        for i, result in enumerate(search_results, 1):
            memory_section += f'**Result {i}** (score: {result.score:.2f}, lines {result.line_start}-{result.line_end})\n'
            memory_section += f'{result.content}\n\n'
        sections.append(memory_section.strip())

    # still include recent daily logs
    recent = memory.get_recent_daily(days=2)
    if recent:
        sections.append(f'# RECENT ACTIVITY\n\n{recent}')

    return '\n\n---\n\n'.join(sections)
