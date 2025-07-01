"""
Regression test for issue #552: Windows 11 hanging on MCP client initialization.

The bug: Windows-specific process creation code in win32.py causes the client
to hang indefinitely during initialization.

The fix: Use the generic anyio.open_process for all platforms instead of
custom Windows-specific code.
"""

import sys
import textwrap

import anyio
import pytest

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific regression test")
@pytest.mark.anyio
async def test_issue_552_windows_no_hang():
    """
    Test that stdio_client doesn't hang on Windows during initialization.

    Issue #552: The Windows-specific process creation code caused hanging.
    This test verifies that using a Python subprocess completes without hanging.
    """
    # Minimal Python script that responds to initialize request
    server_script = textwrap.dedent("""
        import sys, json
        request = json.loads(sys.stdin.readline())
        if request.get("method") == "initialize":
            response = {"jsonrpc": "2.0", "id": request.get("id"), "result": {"protocolVersion": "0.1.0", "capabilities": {}}}
            print(json.dumps(response))
            sys.stdout.flush()
    """)

    params = StdioServerParameters(command=sys.executable, args=["-c", server_script])

    # Should complete without hanging (issue #552 would hang here)
    with anyio.fail_after(5):
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                result = await session.initialize()
                assert result is not None
                assert result.protocolVersion == "0.1.0"
