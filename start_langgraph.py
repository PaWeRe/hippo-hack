#!/usr/bin/env python3
"""
Startup script for LangGraph development server with proper environment loading
"""

import os
import subprocess
import sys
from dotenv import load_dotenv


def main():
    print("🏥 Starting Hippo Medical Agent Development Server...")

    # Load environment variables from .env file
    if not os.path.exists(".env"):
        print("❌ Error: .env file not found. Please create one based on .env.example")
        sys.exit(1)

    load_dotenv()
    print("✅ Environment variables loaded from .env")

    # Verify critical environment variables are loaded
    required_vars = {
        "API_KEY_MISTRAL": "Mistral AI API key",
        "LANGSMITH_API_KEY": "LangSmith API key for tracing",
    }

    missing_vars = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"  - {var}: {description}")
        else:
            print(f"✅ {var}: Loaded")

    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(var)
        print("\nPlease add these to your .env file.")
        sys.exit(1)

    print("\n🚀 Starting LangGraph development server...")
    print("   - API will be available at: http://127.0.0.1:2024")
    print("   - Studio UI will open in your browser")
    print("   - Press Ctrl+C to stop\n")

    # Run langgraph dev with the loaded environment
    try:
        subprocess.run(["uv", "run", "langgraph", "dev"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running langgraph dev: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down development server...")
        sys.exit(0)
    except FileNotFoundError:
        print("❌ Error: 'uv' command not found. Please install uv package manager.")
        print("   Visit: https://docs.astral.sh/uv/")
        sys.exit(1)


if __name__ == "__main__":
    main()
