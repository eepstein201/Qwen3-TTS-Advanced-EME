/**
 * E2E tests for Phase 1c confirmation dialog flows.
 *
 * Tests voice delete, model unload, and generation cancel confirmations
 * with metadata display.
 */

import { test, expect } from "@playwright/test";

test.describe("Voice Delete Confirmation", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("http://127.0.0.1:5123");
    // Wait for UI to load
    await expect(page.locator("h1")).toBeVisible();
  });

  test("voice delete shows metadata confirmation", async ({ page }) => {
    await page.click('[data-testid="tab-voice-management"]');

    // Select a voice to delete (if any exist)
    const voices = await page.locator('[data-testid="voice-dropdown"] option').count();
    if (voices > 1) {
      await page.selectOption('[data-testid="voice-dropdown"]', { index: 1 });
    }

    // Click delete button
    await page.click('[data-testid="delete-btn"]');

    // Verify confirmation metadata is shown
    await expect(page.locator('[data-testid="status-banner"]')).toBeVisible();
    const bannerText = await page.locator('[data-testid="status-banner"]').textContent();

    // Should show voice details
    expect(bannerText).toContain("Delete");
    // May contain metadata fields if voice has them
    if (bannerText.includes("Duration:")) {
      expect(bannerText).toMatch(/Duration:|Format:/);
    }

    // Cancel by clicking elsewhere (button text should revert)
    await page.click("body");
    // Button should return to "Delete" state after timeout
    await page.waitForTimeout(6000);
    const buttonText = await page.locator('[data-testid="delete-btn"]').textContent();
    expect(buttonText).toContain("Delete");
  });

  test("voice delete cancel retains voice", async ({ page }) => {
    await page.click('[data-testid="tab-voice-management"]');

    const voicesBefore = await page.locator('[data-testid="voice-dropdown"] option').count();

    // Click delete, then cancel (wait for timeout)
    if (voicesBefore > 1) {
      await page.selectOption('[data-testid="voice-dropdown"]', { index: 1 });
      await page.click('[data-testid="delete-btn"]');
      // Wait for timeout to cancel
      await page.waitForTimeout(6000);

      // Voice count should be unchanged
      const voicesAfter = await page.locator('[data-testid="voice-dropdown"] option').count();
      expect(voicesAfter).toBe(voicesBefore);
    }
  });
});

test.describe("Model Unload Confirmation", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("http://127.0.0.1:5123");
    await expect(page.locator("h1")).toBeVisible();
  });

  test("model unload shows memory usage confirmation", async ({ page }) => {
    await page.click('[data-testid="tab-model-management"]');

    // Check if clone model is loaded
    const cloneStatus = await page.locator('[data-testid="clone-model-indicator"]').textContent();
    const isLoaded = cloneStatus.includes("Loaded");

    if (isLoaded) {
      // Click unload button
      await page.click('[data-testid="unload-clone-btn"]');

      // Verify memory display in confirmation
      await expect(page.locator('[data-testid="status-banner"]')).toBeVisible();
      const bannerText = await page.locator('[data-testid="status-banner"]').textContent();

      expect(bannerText).toContain("Unload");
      expect(bannerText).toMatch(/MB|memory/i);

      // Cancel by waiting for timeout
      await page.waitForTimeout(6000);

      // Model should still be loaded
      const cloneStatusAfter = await page.locator('[data-testid="clone-model-indicator"]').textContent();
      expect(cloneStatusAfter).toContain("Loaded");
    }
  });

  test("model unload cancel retains model", async ({ page }) => {
    await page.click('[data-testid="tab-model-management"]');

    // Check if design model is loaded
    const designStatus = await page.locator('[data-testid="design-model-indicator"]').textContent();
    const isLoaded = designStatus.includes("Loaded");

    if (isLoaded) {
      // Click unload button
      await page.click('[data-testid="unload-design-btn"]');

      // Wait for timeout to cancel
      await page.waitForTimeout(6000);

      // Model should still be loaded
      const designStatusAfter = await page.locator('[data-testid="design-model-indicator"]').textContent();
      expect(designStatusAfter).toContain("Loaded");
    }
  });
});

test.describe("Generation Cancel Confirmation", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("http://127.0.0.1:5123");
    await expect(page.locator("h1")).toBeVisible();
  });

  test("generation cancel shows progress confirmation", async ({ page }) => {
    // Navigate to Clone tab
    await page.click('[data-testid="tab-clone"]');

    // Enter some text
    await page.fill('[data-testid="text-input"]', "This is a test for generation cancel confirmation');

    // Start generation
    await page.click('[data-testid="generate-btn"]');

    // Wait for generation to start (>10% progress)
    await page.waitForTimeout(5000);

    // Click cancel button
    await page.click('[data-testid="cancel-btn"]');

    // Verify progress display in status
    await expect(page.locator('[data-testid="status-banner"]')).toBeVisible();
    const bannerText = await page.locator('[data-testid="status-banner"]').textContent();

    // Should show progress info if generation is active
    if (bannerText.includes("Cancel")) {
      expect(bannerText).toMatch(/Progress:|Chunks:|ETA:/);
    }

    // Wait for timeout to cancel
    await page.waitForTimeout(6000);

    // Button should return to "Stop" state
    const buttonText = await page.locator('[data-testid="cancel-btn"]').textContent();
    expect(buttonText).toContain("Stop");
  });

  test("cancel under 10% proceeds immediately", async ({ page }) => {
    await page.click('[data-testid="tab-clone"]');

    // Enter short text
    await page.fill('[data-testid="text-input"]', "Short text");

    // Start generation
    await page.click('[data-testid="generate-btn"]');

    // Wait a very short time (<10%)
    await page.waitForTimeout(1000);

    // Click cancel immediately
    await page.click('[data-testid="cancel-btn"]');

    // Should cancel quickly without confirmation dialog
    // Wait for status to show "Generation cancelled" or similar
    await page.waitForTimeout(3000);

    const statusText = await page.locator('[data-testid="status"]').textContent();
    expect(statusText).toMatch(/cancel|stop|abort/i);
  });
});
