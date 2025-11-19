"""GitHub Autopilot Widget - AI-assisted git and GitHub operations.

Provides automated git analysis, commit message generation, code review,
and PR management using AI backends.
"""

import subprocess
import re
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime

from ..core.router import Router
from ..core.backend import Context
from ..storage.database import ResponseStore
from ..utils.logger import get_logger

logger = get_logger(__name__)


class GitHubAutopilot:
    """
    GitHub Autopilot Widget.

    Provides AI-assisted git operations:
    - Analyze git diffs and explain changes
    - Generate commit messages
    - Code review with security checks
    - PR body generation
    - GitHub API integration
    """

    def __init__(
        self,
        router: Router,
        repo_path: str | Path = ".",
        github_token: Optional[str] = None,
        store: Optional[ResponseStore] = None
    ):
        """
        Initialize GitHub Autopilot.

        Args:
            router: Router for AI backend access
            repo_path: Path to git repository
            github_token: GitHub personal access token
            store: Optional response store
        """
        self.router = router
        self.repo_path = Path(repo_path).resolve()
        self.github_token = github_token
        self.store = store

        logger.info(f"GitHub Autopilot initialized for repo: {self.repo_path}")

    def get_status(self) -> Dict[str, Any]:
        """
        Get current git status.

        Returns:
            Dictionary with status information
        """
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )

            # Parse status output
            changes = {
                "modified": [],
                "added": [],
                "deleted": [],
                "untracked": []
            }

            for line in result.stdout.splitlines():
                if not line.strip():
                    continue

                status = line[:2]
                file_path = line[3:].strip()

                if "M" in status:
                    changes["modified"].append(file_path)
                elif "A" in status:
                    changes["added"].append(file_path)
                elif "D" in status:
                    changes["deleted"].append(file_path)
                elif "?" in status:
                    changes["untracked"].append(file_path)

            # Get current branch
            branch_result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )

            return {
                "branch": branch_result.stdout.strip(),
                "changes": changes,
                "total_files": sum(len(v) for v in changes.values()),
                "has_changes": any(len(v) > 0 for v in changes.values())
            }

        except subprocess.CalledProcessError as e:
            logger.error(f"Git status failed: {e}")
            return {
                "error": str(e),
                "branch": None,
                "changes": {},
                "has_changes": False
            }

    def get_diff(
        self,
        file_path: Optional[str] = None,
        staged: bool = False
    ) -> str:
        """
        Get git diff.

        Args:
            file_path: Specific file to diff (None = all files)
            staged: Get staged changes only

        Returns:
            Diff output
        """
        try:
            cmd = ["git", "diff"]

            if staged:
                cmd.append("--staged")

            if file_path:
                cmd.append(file_path)

            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )

            return result.stdout

        except subprocess.CalledProcessError as e:
            logger.error(f"Git diff failed: {e}")
            return f"Error getting diff: {e}"

    async def explain_changes(
        self,
        file_path: Optional[str] = None,
        backend: str = "claude"
    ) -> Dict[str, Any]:
        """
        Use AI to explain git changes.

        Args:
            file_path: Specific file to explain (None = all changes)
            backend: AI backend to use

        Returns:
            Explanation with risk assessment
        """
        # Get diff
        diff = self.get_diff(file_path)

        if not diff or diff.startswith("Error"):
            return {
                "error": "No changes to explain",
                "diff": diff
            }

        # Build prompt for AI
        prompt = f"""Analyze this git diff and explain:
1. What changed and why
2. Potential risks or regressions
3. Impact on system behavior
4. Security concerns (if any)

Git diff:
```diff
{diff[:5000]}  # Limit to 5000 chars
```

Provide a concise analysis."""

        # Send to AI
        context = Context(
            system_prompt="You are a senior software engineer reviewing code changes."
        )

        backend_instance = self.router.get_backend(backend)
        if not backend_instance:
            return {"error": f"Backend '{backend}' not found"}

        response = await backend_instance.send_message(prompt, context)

        return {
            "explanation": response.content,
            "backend": backend,
            "file": file_path or "all changes",
            "diff_size": len(diff),
            "latency_ms": response.latency_ms,
            "error": response.error
        }

    async def generate_commit_message(
        self,
        backend: str = "claude",
        style: str = "conventional"
    ) -> Dict[str, Any]:
        """
        Generate commit message from staged changes.

        Args:
            backend: AI backend to use
            style: Commit message style (conventional, detailed, concise)

        Returns:
            Generated commit message
        """
        # Get staged diff
        diff = self.get_diff(staged=True)

        if not diff:
            return {"error": "No staged changes"}

        # Get recent commit messages for style reference
        try:
            log_result = subprocess.run(
                ["git", "log", "--oneline", "-n", "5"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            recent_commits = log_result.stdout
        except:
            recent_commits = ""

        # Build prompt
        style_instructions = {
            "conventional": "Use conventional commits format (feat:, fix:, refactor:, etc.)",
            "detailed": "Write detailed, multi-line commit message explaining changes",
            "concise": "Write a brief, single-line commit message"
        }

        prompt = f"""Generate a commit message for these staged changes.

Style: {style_instructions.get(style, style)}

Recent commit messages for reference:
{recent_commits}

Staged changes:
```diff
{diff[:5000]}
```

Generate an appropriate commit message that:
1. Accurately describes what changed
2. Follows the project's commit style
3. Is clear and concise

Return only the commit message, no explanation."""

        # Send to AI
        backend_instance = self.router.get_backend(backend)
        if not backend_instance:
            return {"error": f"Backend '{backend}' not found"}

        context = Context(
            system_prompt="You are a software engineer writing commit messages."
        )

        response = await backend_instance.send_message(prompt, context)

        # Clean up response (remove markdown, quotes, etc.)
        message = response.content.strip()
        message = re.sub(r'^```.*\n|```$', '', message, flags=re.MULTILINE)
        message = message.strip('"\'')

        return {
            "message": message,
            "backend": backend,
            "style": style,
            "diff_size": len(diff),
            "error": response.error
        }

    async def review_code(
        self,
        file_path: Optional[str] = None,
        backend: str = "claude",
        focus: List[str] = None
    ) -> Dict[str, Any]:
        """
        AI code review of changes.

        Args:
            file_path: Specific file to review
            backend: AI backend to use
            focus: Areas to focus on (security, performance, style, etc.)

        Returns:
            Code review feedback
        """
        focus = focus or ["security", "bugs", "performance", "style"]

        # Get diff
        diff = self.get_diff(file_path)

        if not diff:
            return {"error": "No changes to review"}

        # Build prompt
        focus_areas = "\n".join(f"- {area}" for area in focus)

        prompt = f"""Perform a code review of these changes.

Focus areas:
{focus_areas}

Changes:
```diff
{diff[:6000]}
```

Provide:
1. Critical issues (security, bugs)
2. Warnings (potential problems)
3. Suggestions (improvements)
4. Overall assessment

Be specific and actionable."""

        # Send to AI
        backend_instance = self.router.get_backend(backend)
        if not backend_instance:
            return {"error": f"Backend '{backend}' not found"}

        context = Context(
            system_prompt="You are a senior software engineer performing code review. Be thorough but constructive."
        )

        response = await backend_instance.send_message(prompt, context)

        return {
            "review": response.content,
            "backend": backend,
            "focus_areas": focus,
            "file": file_path or "all changes",
            "diff_size": len(diff),
            "error": response.error
        }

    async def generate_pr_body(
        self,
        title: str,
        base_branch: str = "main",
        backend: str = "claude"
    ) -> Dict[str, Any]:
        """
        Generate Pull Request body from changes.

        Args:
            title: PR title
            base_branch: Base branch for comparison
            backend: AI backend to use

        Returns:
            Generated PR body
        """
        try:
            # Get current branch
            current_branch_result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            current_branch = current_branch_result.stdout.strip()

            # Get diff from base branch
            diff_result = subprocess.run(
                ["git", "diff", f"{base_branch}...{current_branch}"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            diff = diff_result.stdout

            # Get commit history
            log_result = subprocess.run(
                ["git", "log", f"{base_branch}..{current_branch}", "--oneline"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            commits = log_result.stdout

        except subprocess.CalledProcessError as e:
            return {"error": f"Git command failed: {e}"}

        if not diff:
            return {"error": "No changes between branches"}

        # Build prompt
        prompt = f"""Generate a Pull Request description for these changes.

PR Title: {title}
Base Branch: {base_branch}
Feature Branch: {current_branch}

Commits:
{commits}

Changes:
```diff
{diff[:8000]}
```

Generate a comprehensive PR description with:
## Summary
(1-2 sentences)

## Changes
- Bullet list of key changes

## Testing
- How to test these changes

## Notes
- Any additional context

Format as markdown."""

        # Send to AI
        backend_instance = self.router.get_backend(backend)
        if not backend_instance:
            return {"error": f"Backend '{backend}' not found"}

        context = Context(
            system_prompt="You are a software engineer writing pull request descriptions."
        )

        response = await backend_instance.send_message(prompt, context)

        return {
            "body": response.content,
            "title": title,
            "base_branch": base_branch,
            "current_branch": current_branch,
            "commit_count": len(commits.splitlines()),
            "backend": backend,
            "error": response.error
        }

    def commit_with_message(
        self,
        message: str,
        add_all: bool = False
    ) -> Dict[str, Any]:
        """
        Create a git commit with the given message.

        Args:
            message: Commit message
            add_all: Stage all changes before committing

        Returns:
            Commit result
        """
        try:
            if add_all:
                subprocess.run(
                    ["git", "add", "."],
                    cwd=self.repo_path,
                    check=True
                )

            result = subprocess.run(
                ["git", "commit", "-m", message],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )

            return {
                "success": True,
                "output": result.stdout,
                "message": message
            }

        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "error": str(e),
                "output": e.stderr
            }

    def get_file_changes_summary(self) -> List[Dict[str, Any]]:
        """
        Get a summary of changed files with metadata.

        Returns:
            List of file change cards
        """
        status = self.get_status()

        if not status.get("has_changes"):
            return []

        cards = []

        for category, files in status["changes"].items():
            for file_path in files:
                # Get file type icon
                ext = Path(file_path).suffix
                icon = self._get_file_icon(ext)

                # Get diff stats
                try:
                    stat_result = subprocess.run(
                        ["git", "diff", "--stat", file_path],
                        cwd=self.repo_path,
                        capture_output=True,
                        text=True
                    )
                    stats = stat_result.stdout.strip().split("\n")[-1] if stat_result.stdout else ""
                except:
                    stats = ""

                cards.append({
                    "file": file_path,
                    "category": category,
                    "icon": icon,
                    "stats": stats
                })

        return cards

    def _get_file_icon(self, ext: str) -> str:
        """Get emoji icon for file type."""
        icons = {
            ".py": "🐍",
            ".js": "📜",
            ".ts": "💙",
            ".jsx": "⚛️",
            ".tsx": "⚛️",
            ".md": "📄",
            ".yaml": "⚙️",
            ".yml": "⚙️",
            ".json": "📋",
            ".css": "🎨",
            ".html": "🌐",
            ".sh": "🔧",
            ".go": "🐹",
            ".rs": "🦀",
            ".java": "☕",
        }
        return icons.get(ext, "📝")
