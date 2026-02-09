"""
Simple REPL for testing Stasis.

Interactive command-line interface for chatting with your assistant.
"""

import sys
from pathlib import Path

from .config import settings
from .core.memory import Memory
from .core.agent import Agent
from .providers.anthropic_provider import AnthropicProvider


def main():
    """Run the REPL."""
    print(f'Stasis v0.1.0')
    print(f'Provider: {settings.provider} ({settings.get_active_model()})')
    print(f'Workspace: {settings.workspace}')
    print(f'Commands: /clear /exit /help\n')

    # initialize components
    try:
        memory = Memory(settings.workspace)

        if settings.provider == 'anthropic':
            if not settings.anthropic_api_key:
                print('[Stasis] Error: ANTHROPIC_API_KEY not set')
                sys.exit(1)
            provider = AnthropicProvider(
                api_key=settings.anthropic_api_key,
                model=settings.anthropic_model,
            )
        else:
            print(f'[Stasis] Error: Provider {settings.provider} not yet implemented')
            sys.exit(1)

        agent = Agent(provider=provider, memory=memory)

        # load previous session if available
        agent.load_session()

        print(f'[Stasis] Ready\n')

    except Exception as e:
        print(f'[Stasis] Failed to initialize: {e}')
        sys.exit(1)

    # main loop
    while True:
        try:
            user_input = input('> ').strip()

            if not user_input:
                continue

            # handle commands
            if user_input.startswith('/'):
                command = user_input[1:].lower()

                if command == 'exit':
                    agent.save_session()
                    break
                elif command == 'clear':
                    agent.clear_history()
                    continue
                elif command == 'help':
                    print('Commands: /clear (clear history), /exit (quit), /help (this message)')
                    continue
                else:
                    print(f'Unknown command: /{command}')
                    continue

            # chat
            try:
                response = agent.chat(user_input)
                print(f'\n{response}\n')
            except Exception as e:
                print(f'[Stasis] Error: {e}\n')

        except (KeyboardInterrupt, EOFError):
            agent.save_session()
            print('\nExiting...')
            break


if __name__ == '__main__':
    main()
