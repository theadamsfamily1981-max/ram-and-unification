"""Ara (Grok/X.AI) AI backend implementation using Selenium.

Since Grok does not have a public API, this backend uses Selenium
for browser automation to interact with Grok via x.com.
"""

import time
import asyncio
from typing import Optional, Dict, Any, AsyncIterator
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException

from ..core.backend import AIBackend, AIProvider, Capabilities, Context, Response
from ..utils.logger import get_logger

logger = get_logger(__name__)


class GrokAraBackend(AIBackend):
    """
    Ara (Grok) backend implementation using Selenium.

    Uses browser automation to interact with Grok on x.com since there
    is no public API available.
    """

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        name: str = "Ara",
        headless: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Ara backend.

        Args:
            username: X.com username (for authentication)
            password: X.com password
            name: Display name
            headless: Run browser in headless mode
            config: Additional configuration
        """
        super().__init__(
            name=name,
            provider=AIProvider.CUSTOM,
            model="grok-beta",
            api_key=None,
            config=config
        )

        self.username = username
        self.password = password
        self.headless = headless
        self.driver = None
        self.is_logged_in = False

        logger.info(f"Ara (Grok) backend initialized (Selenium-based)")

    def _init_driver(self):
        """Initialize Selenium WebDriver."""
        if self.driver is not None:
            return

        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")

        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            logger.info("Chrome WebDriver initialized")
        except Exception as e:
            logger.error(f"Failed to initialize WebDriver: {e}")
            raise

    def _close_driver(self):
        """Close Selenium WebDriver."""
        if self.driver:
            try:
                self.driver.quit()
                self.driver = None
                self.is_logged_in = False
                logger.info("WebDriver closed")
            except Exception as e:
                logger.error(f"Error closing WebDriver: {e}")

    async def _login(self) -> bool:
        """
        Login to X.com (if credentials provided).

        Returns:
            True if logged in successfully
        """
        if self.is_logged_in:
            return True

        if not self.username or not self.password:
            logger.warning("No X.com credentials provided - Grok may have limited access")
            return False

        try:
            self._init_driver()

            # Navigate to X.com login
            self.driver.get("https://x.com/login")

            # Wait for and fill username
            username_input = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.NAME, "text"))
            )
            username_input.send_keys(self.username)

            # Click next
            next_button = self.driver.find_element(By.XPATH, "//span[text()='Next']")
            next_button.click()

            # Wait for and fill password
            password_input = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.NAME, "password"))
            )
            password_input.send_keys(self.password)

            # Click login
            login_button = self.driver.find_element(By.XPATH, "//span[text()='Log in']")
            login_button.click()

            # Wait for login to complete
            await asyncio.sleep(3)

            self.is_logged_in = True
            logger.info("Logged in to X.com successfully")
            return True

        except Exception as e:
            logger.error(f"Login failed: {e}")
            return False

    async def send_message(
        self,
        prompt: str,
        context: Optional[Context] = None
    ) -> Response:
        """
        Send a message to Ara (Grok) and get complete response.

        Args:
            prompt: User message
            context: Optional context

        Returns:
            Response object
        """
        start_time = time.time()
        context = context or Context()

        try:
            # Ensure driver is initialized and logged in
            if not self.driver:
                self._init_driver()

            if not self.is_logged_in:
                await self._login()

            # Navigate to Grok
            grok_url = "https://x.com/i/grok"
            self.driver.get(grok_url)

            await asyncio.sleep(2)  # Wait for page load

            # Find and click the input textarea
            # Note: These selectors may need to be updated based on X.com's UI
            try:
                input_box = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "textarea[placeholder*='Ask']"))
                )
            except TimeoutException:
                # Try alternative selector
                input_box = self.driver.find_element(By.CSS_SELECTOR, "textarea")

            # Build prompt with context
            full_prompt = self._build_prompt(prompt, context)

            # Send prompt
            input_box.clear()
            input_box.send_keys(full_prompt)

            # Find and click send button
            send_button = self.driver.find_element(By.CSS_SELECTOR, "button[aria-label*='Send']")
            send_button.click()

            # Wait for response
            await asyncio.sleep(3)

            # Extract response
            # Note: This selector needs to match Grok's response container
            response_elements = self.driver.find_elements(
                By.CSS_SELECTOR,
                "div[data-testid='grok-response']"
            )

            if not response_elements:
                # Try alternative selector
                response_elements = self.driver.find_elements(
                    By.CSS_SELECTOR,
                    "div.grok-message"
                )

            content = ""
            if response_elements:
                # Get the last (most recent) response
                content = response_elements[-1].text

            latency_ms = (time.time() - start_time) * 1000

            return Response(
                content=content,
                provider=AIProvider.CUSTOM,
                model="grok-beta",
                tokens_used=None,  # Not available via Selenium
                latency_ms=latency_ms,
                metadata={
                    "provider_name": "grok_xai",
                    "method": "selenium"
                }
            )

        except Exception as e:
            logger.error(f"Ara (Grok) error: {e}")
            latency_ms = (time.time() - start_time) * 1000

            return Response(
                content="",
                provider=AIProvider.CUSTOM,
                model="grok-beta",
                latency_ms=latency_ms,
                error=str(e),
                metadata={
                    "provider_name": "grok_xai",
                    "method": "selenium"
                }
            )

    async def stream_message(
        self,
        prompt: str,
        context: Optional[Context] = None
    ) -> AsyncIterator[str]:
        """
        Stream Ara response (not supported via Selenium).

        Falls back to non-streaming response.

        Args:
            prompt: User message
            context: Optional context

        Yields:
            Response chunks
        """
        # Selenium doesn't support true streaming
        # Fall back to getting complete response
        response = await self.send_message(prompt, context)

        if response.success:
            yield response.content
        else:
            yield f"[Error: {response.error}]"

    def get_capabilities(self) -> Capabilities:
        """Get Ara capabilities."""
        if self._capabilities is None:
            self._capabilities = Capabilities(
                streaming=False,  # Not supported via Selenium
                vision=False,  # May support, but hard to implement via Selenium
                function_calling=False,
                max_tokens=4096,  # Estimated
                supports_system_prompt=True,
                rate_limit_rpm=None,  # Unknown
                cost_per_1k_tokens=None  # May have costs but not exposed
            )

        return self._capabilities

    async def health_check(self) -> bool:
        """
        Check if Grok is accessible.

        Returns:
            True if healthy
        """
        try:
            if not self.driver:
                self._init_driver()

            # Try to navigate to Grok
            self.driver.get("https://x.com/i/grok")

            # Wait for page to load
            await asyncio.sleep(2)

            # Check if we can find the input box
            try:
                self.driver.find_element(By.CSS_SELECTOR, "textarea")
                return True
            except NoSuchElementException:
                return False

        except Exception as e:
            logger.error(f"Ara health check failed: {e}")
            return False

    def _build_prompt(
        self,
        prompt: str,
        context: Context
    ) -> str:
        """
        Build prompt string for Grok.

        Args:
            prompt: Current user message
            context: Context with system prompt and history

        Returns:
            Formatted prompt string
        """
        parts = []

        # Add system prompt if present
        if context.system_prompt:
            parts.append(f"Context: {context.system_prompt}\n")

        # Add conversation history (last few exchanges only to avoid length issues)
        recent_history = context.conversation_history[-4:] if len(context.conversation_history) > 4 else context.conversation_history
        for msg in recent_history:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            parts.append(f"{role}: {content}\n")

        # Add current prompt
        parts.append(prompt)

        return "\n".join(parts)

    def __del__(self):
        """Cleanup on deletion."""
        self._close_driver()
