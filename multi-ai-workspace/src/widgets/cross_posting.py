"""Cross-Posting Panel - Share AI responses across platforms.

The Cross-Posting Panel enables easy sharing of AI responses to:
- Clipboard
- Files (txt, md, json)
- Social media formats
- Email drafts
- Code snippets
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path
import json

from ..core.backend import Response
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ExportFormat:
    """Export format configuration."""
    name: str
    extension: str
    mime_type: str
    formatter: str  # Function name to call


class CrossPostingPanel:
    """
    Cross-Posting Panel Widget.

    Enables sharing AI responses in various formats:
    - Plain text
    - Markdown
    - JSON
    - Code blocks
    - Social media posts
    """

    SUPPORTED_FORMATS = {
        "text": ExportFormat(
            name="Plain Text",
            extension=".txt",
            mime_type="text/plain",
            formatter="format_text"
        ),
        "markdown": ExportFormat(
            name="Markdown",
            extension=".md",
            mime_type="text/markdown",
            formatter="format_markdown"
        ),
        "json": ExportFormat(
            name="JSON",
            extension=".json",
            mime_type="application/json",
            formatter="format_json"
        ),
        "code": ExportFormat(
            name="Code Snippet",
            extension=".txt",
            mime_type="text/plain",
            formatter="format_code"
        ),
        "tweet": ExportFormat(
            name="Twitter/X Post",
            extension=".txt",
            mime_type="text/plain",
            formatter="format_tweet"
        ),
        "email": ExportFormat(
            name="Email Draft",
            extension=".txt",
            mime_type="text/plain",
            formatter="format_email"
        )
    }

    def __init__(self, output_dir: str | Path = "exports"):
        """
        Initialize Cross-Posting Panel.

        Args:
            output_dir: Directory for exported files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"CrossPostingPanel initialized: {self.output_dir}")

    def export_response(
        self,
        response: Response,
        format_type: str = "text",
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Export AI response in specified format.

        Args:
            response: AI response to export
            format_type: Export format (text, markdown, json, etc.)
            filename: Optional custom filename
            metadata: Additional metadata to include

        Returns:
            Export result with file path and content
        """
        if format_type not in self.SUPPORTED_FORMATS:
            logger.error(f"Unsupported format: {format_type}")
            return {
                "success": False,
                "error": f"Unsupported format: {format_type}"
            }

        format_spec = self.SUPPORTED_FORMATS[format_type]

        # Format content
        formatter = getattr(self, format_spec.formatter)
        content = formatter(response, metadata)

        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backend = response.metadata.get("backend", "unknown")
            filename = f"{backend}_{timestamp}{format_spec.extension}"

        # Write to file
        output_path = self.output_dir / filename

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)

            logger.info(f"Exported to: {output_path}")

            return {
                "success": True,
                "format": format_type,
                "file_path": str(output_path),
                "content": content,
                "size_bytes": len(content.encode("utf-8"))
            }

        except Exception as e:
            logger.error(f"Export failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def export_conversation(
        self,
        messages: List[Dict[str, Any]],
        format_type: str = "markdown",
        filename: Optional[str] = None,
        title: str = "Conversation"
    ) -> Dict[str, Any]:
        """
        Export entire conversation.

        Args:
            messages: List of messages
            format_type: Export format
            filename: Optional filename
            title: Conversation title

        Returns:
            Export result
        """
        if format_type == "markdown":
            content = self.format_conversation_markdown(messages, title)
        elif format_type == "json":
            content = json.dumps({
                "title": title,
                "messages": messages,
                "exported_at": datetime.now().isoformat()
            }, indent=2)
        else:  # text
            content = self.format_conversation_text(messages, title)

        # Generate filename
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = self.SUPPORTED_FORMATS[format_type].extension
            filename = f"conversation_{timestamp}{ext}"

        output_path = self.output_dir / filename

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)

            logger.info(f"Exported conversation to: {output_path}")

            return {
                "success": True,
                "format": format_type,
                "file_path": str(output_path),
                "message_count": len(messages),
                "size_bytes": len(content.encode("utf-8"))
            }

        except Exception as e:
            logger.error(f"Conversation export failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    # ===== Formatters =====

    def format_text(
        self,
        response: Response,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format as plain text."""
        lines = []

        if metadata:
            lines.append(f"Prompt: {metadata.get('prompt', 'N/A')}")
            lines.append("")

        lines.append(response.content)
        lines.append("")

        if response.provider:
            lines.append(f"Generated by: {response.provider.value} ({response.model})")

        if response.latency_ms:
            lines.append(f"Response time: {response.latency_ms:.0f}ms")

        return "\n".join(lines)

    def format_markdown(
        self,
        response: Response,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format as Markdown."""
        lines = []

        if metadata and metadata.get("prompt"):
            lines.append(f"## Prompt\n")
            lines.append(f"{metadata['prompt']}\n")

        lines.append("## Response\n")
        lines.append(f"{response.content}\n")

        # Add metadata
        lines.append("---\n")
        lines.append("**Metadata**\n")
        lines.append(f"- Backend: {response.provider.value} ({response.model})")

        if response.latency_ms:
            lines.append(f"- Response time: {response.latency_ms:.0f}ms")

        if response.tokens_used:
            lines.append(f"- Tokens used: {response.tokens_used}")

        lines.append(f"- Generated: {response.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

        return "\n".join(lines)

    def format_json(
        self,
        response: Response,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format as JSON."""
        data = response.to_dict()

        if metadata:
            data["metadata"] = {**data.get("metadata", {}), **metadata}

        return json.dumps(data, indent=2)

    def format_code(
        self,
        response: Response,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format as code snippet."""
        # Extract code blocks from response
        content = response.content

        # Try to extract code from markdown code blocks
        import re
        code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', content, re.DOTALL)

        if code_blocks:
            # Return first code block
            code = code_blocks[0].strip()
        else:
            # No code blocks, return as-is
            code = content

        lines = []

        if metadata and metadata.get("prompt"):
            lines.append(f"# {metadata['prompt']}")
            lines.append("")

        lines.append(code)
        lines.append("")
        lines.append(f"# Generated by: {response.provider.value}")

        return "\n".join(lines)

    def format_tweet(
        self,
        response: Response,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format as Twitter/X post (280 char limit)."""
        content = response.content

        # Truncate to fit Twitter limit
        max_length = 280

        if len(content) <= max_length:
            return content

        # Truncate and add ellipsis
        truncated = content[:max_length - 3] + "..."

        return truncated

    def format_email(
        self,
        response: Response,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format as email draft."""
        lines = []

        # Subject
        subject = metadata.get("subject", "AI Response") if metadata else "AI Response"
        lines.append(f"Subject: {subject}")
        lines.append("")

        # Body
        if metadata and metadata.get("prompt"):
            lines.append(f"Re: {metadata['prompt']}")
            lines.append("")

        lines.append(response.content)
        lines.append("")

        # Signature
        lines.append("---")
        lines.append(f"Generated by {response.provider.value} AI")
        lines.append(f"{response.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

        return "\n".join(lines)

    def format_conversation_markdown(
        self,
        messages: List[Dict[str, Any]],
        title: str = "Conversation"
    ) -> str:
        """Format conversation as Markdown."""
        lines = []

        lines.append(f"# {title}\n")
        lines.append(f"*Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        lines.append("---\n")

        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            backend = msg.get("backend_name", "")

            if role == "user":
                lines.append("### 👤 User\n")
            else:
                backend_label = f" ({backend})" if backend else ""
                lines.append(f"### 🤖 Assistant{backend_label}\n")

            lines.append(f"{content}\n")
            lines.append("---\n")

        return "\n".join(lines)

    def format_conversation_text(
        self,
        messages: List[Dict[str, Any]],
        title: str = "Conversation"
    ) -> str:
        """Format conversation as plain text."""
        lines = []

        lines.append(f"{title}")
        lines.append(f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)
        lines.append("")

        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            backend = msg.get("backend_name", "")

            if role == "user":
                lines.append("USER:")
            else:
                backend_label = f" ({backend})" if backend else ""
                lines.append(f"ASSISTANT{backend_label}:")

            lines.append(content)
            lines.append("-" * 80)
            lines.append("")

        return "\n".join(lines)

    def get_clipboard_content(
        self,
        response: Response,
        format_type: str = "text"
    ) -> str:
        """
        Get formatted content for clipboard.

        Args:
            response: AI response
            format_type: Format type

        Returns:
            Formatted string for clipboard
        """
        format_spec = self.SUPPORTED_FORMATS.get(format_type)
        if not format_spec:
            return response.content

        formatter = getattr(self, format_spec.formatter)
        return formatter(response)

    def list_exports(self) -> List[Dict[str, Any]]:
        """
        List all exported files.

        Returns:
            List of export metadata
        """
        exports = []

        for file_path in self.output_dir.glob("*"):
            if file_path.is_file():
                exports.append({
                    "filename": file_path.name,
                    "size_bytes": file_path.stat().st_size,
                    "created": datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
                    "path": str(file_path)
                })

        return sorted(exports, key=lambda x: x["created"], reverse=True)
