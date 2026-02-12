"""
GitHub Webhook Handler — push, pull_request, issues, issue_comment, star, release.

Maps GitHub webhook events to cognitive events with human-readable content summaries.
"""

import json
import time
from typing import List, Mapping

from .base import WebhookHandler, WebhookEvent


class GitHubHandler(WebhookHandler):

    @property
    def provider(self) -> str:
        return "github"

    @property
    def platform(self) -> str:
        return "github-webhook"

    def supported_events(self) -> List[str]:
        return [
            "push", "pull_request", "issues", "issue_comment",
            "star", "fork", "release", "workflow_run",
        ]

    async def handle(
        self, headers: Mapping[str, str], body: bytes
    ) -> List[WebhookEvent]:
        event_type = headers.get("x-github-event", "unknown")
        delivery_id = headers.get("x-github-delivery", "")
        payload = json.loads(body)

        handler_map = {
            "push": self._handle_push,
            "pull_request": self._handle_pull_request,
            "issues": self._handle_issue,
            "issue_comment": self._handle_issue_comment,
            "star": self._handle_star,
            "release": self._handle_release,
            "workflow_run": self._handle_workflow_run,
        }

        handler = handler_map.get(event_type, self._handle_generic)
        return handler(payload, delivery_id)

    def _handle_push(self, payload: dict, delivery_id: str) -> List[WebhookEvent]:
        repo = payload.get("repository", {}).get("full_name", "unknown")
        commits = payload.get("commits", [])
        ref = payload.get("ref", "").replace("refs/heads/", "")
        pusher = payload.get("pusher", {}).get("name", "unknown")

        commit_msgs = "\n".join(
            f"- {c.get('message', '').split(chr(10))[0]}"
            for c in commits[:10]
        )
        content = (
            f"Push to {repo}/{ref} by {pusher}: "
            f"{len(commits)} commit(s)\n{commit_msgs}"
        )

        return [WebhookEvent(
            event_type="push",
            content=content,
            metadata={
                "repo": repo, "ref": ref, "pusher": pusher,
                "commit_count": len(commits), "delivery_id": delivery_id,
                "commits": [
                    {"sha": c.get("id", "")[:8], "message": c.get("message", "")[:200]}
                    for c in commits[:10]
                ],
            },
            timestamp=time.time(),
            session_id=f"github-{repo}",
            role="system",
        )]

    def _handle_pull_request(self, payload: dict, delivery_id: str) -> List[WebhookEvent]:
        action = payload.get("action", "")
        pr = payload.get("pull_request", {})
        repo = payload.get("repository", {}).get("full_name", "unknown")
        title = pr.get("title", "")
        body_text = (pr.get("body") or "")[:500]
        user = pr.get("user", {}).get("login", "unknown")
        number = pr.get("number", 0)

        content = f"PR #{number} {action} on {repo} by {user}: {title}\n{body_text}"

        return [WebhookEvent(
            event_type="pull_request",
            content=content,
            metadata={
                "repo": repo, "action": action, "pr_number": number,
                "user": user, "title": title, "delivery_id": delivery_id,
            },
            timestamp=time.time(),
            session_id=f"github-{repo}",
            role="system",
        )]

    def _handle_issue(self, payload: dict, delivery_id: str) -> List[WebhookEvent]:
        action = payload.get("action", "")
        issue = payload.get("issue", {})
        repo = payload.get("repository", {}).get("full_name", "unknown")
        title = issue.get("title", "")
        body_text = (issue.get("body") or "")[:500]
        user = issue.get("user", {}).get("login", "unknown")
        number = issue.get("number", 0)

        content = f"Issue #{number} {action} on {repo} by {user}: {title}\n{body_text}"

        return [WebhookEvent(
            event_type="issue",
            content=content,
            metadata={
                "repo": repo, "action": action, "issue_number": number,
                "user": user, "title": title, "delivery_id": delivery_id,
                "labels": [l.get("name", "") for l in issue.get("labels", [])],
            },
            timestamp=time.time(),
            session_id=f"github-{repo}",
            role="system",
        )]

    def _handle_issue_comment(self, payload: dict, delivery_id: str) -> List[WebhookEvent]:
        action = payload.get("action", "")
        comment = payload.get("comment", {})
        issue = payload.get("issue", {})
        repo = payload.get("repository", {}).get("full_name", "unknown")
        user = comment.get("user", {}).get("login", "unknown")
        number = issue.get("number", 0)
        body_text = (comment.get("body") or "")[:500]

        content = f"Comment on #{number} ({action}) on {repo} by {user}:\n{body_text}"

        return [WebhookEvent(
            event_type="issue_comment",
            content=content,
            metadata={
                "repo": repo, "action": action, "issue_number": number,
                "user": user, "delivery_id": delivery_id,
            },
            timestamp=time.time(),
            session_id=f"github-{repo}",
            role="user",
        )]

    def _handle_star(self, payload: dict, delivery_id: str) -> List[WebhookEvent]:
        action = payload.get("action", "created")
        repo = payload.get("repository", {}).get("full_name", "unknown")
        user = payload.get("sender", {}).get("login", "unknown")
        stars = payload.get("repository", {}).get("stargazers_count", 0)

        content = f"Star {action} on {repo} by {user} (total: {stars})"

        return [WebhookEvent(
            event_type="star",
            content=content,
            metadata={
                "repo": repo, "action": action, "user": user,
                "stars": stars, "delivery_id": delivery_id,
            },
            timestamp=time.time(),
            session_id=f"github-{repo}",
            role="system",
        )]

    def _handle_release(self, payload: dict, delivery_id: str) -> List[WebhookEvent]:
        action = payload.get("action", "")
        release = payload.get("release", {})
        repo = payload.get("repository", {}).get("full_name", "unknown")
        tag = release.get("tag_name", "")
        name = release.get("name", "")
        body_text = (release.get("body") or "")[:500]

        content = f"Release {tag} {action} on {repo}: {name}\n{body_text}"

        return [WebhookEvent(
            event_type="release",
            content=content,
            metadata={
                "repo": repo, "action": action, "tag": tag,
                "delivery_id": delivery_id,
            },
            timestamp=time.time(),
            session_id=f"github-{repo}",
            role="system",
        )]

    def _handle_workflow_run(self, payload: dict, delivery_id: str) -> List[WebhookEvent]:
        action = payload.get("action", "")
        run = payload.get("workflow_run", {})
        repo = payload.get("repository", {}).get("full_name", "unknown")
        name = run.get("name", "")
        conclusion = run.get("conclusion", "")
        branch = run.get("head_branch", "")

        content = f"Workflow '{name}' {action} on {repo}/{branch}: {conclusion or 'in progress'}"

        return [WebhookEvent(
            event_type="workflow_run",
            content=content,
            metadata={
                "repo": repo, "action": action, "workflow": name,
                "conclusion": conclusion, "branch": branch,
                "delivery_id": delivery_id,
            },
            timestamp=time.time(),
            session_id=f"github-{repo}",
            role="system",
        )]

    def _handle_generic(self, payload: dict, delivery_id: str) -> List[WebhookEvent]:
        repo = payload.get("repository", {}).get("full_name", "unknown")
        action = payload.get("action", "")
        content = f"GitHub event on {repo}: action={action}"

        return [WebhookEvent(
            event_type="generic",
            content=content,
            metadata={"repo": repo, "action": action, "delivery_id": delivery_id},
            timestamp=time.time(),
            session_id=f"github-{repo}",
            role="system",
        )]
