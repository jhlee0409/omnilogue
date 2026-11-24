/**
 * The collapsed thinking header previews the first line as plain text:
 * Gemini thoughts open with `**Subject**`, which showed its asterisks.
 */
import { afterEach, describe, expect, it } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import { ExpandKeyProvider } from "@/contexts/CaptureExpandContext";
import { ThinkingRenderer } from "./ThinkingRenderer";

const preview = (thinking: string) => {
  render(
    <ExpandKeyProvider value="msg-thinking">
      <ThinkingRenderer thinking={thinking} />
    </ExpandKeyProvider>
  );
  return screen.getByRole("button").textContent ?? "";
};

afterEach(cleanup);

describe("ThinkingRenderer preview", () => {
  it("drops markdown markers from the first line", () => {
    const text = preview("**Evaluating Defaults for Tauri**\n\nI am checking the menu.");
    expect(text).toContain("Evaluating Defaults for Tauri...");
    expect(text).not.toContain("**");
  });

  it("keeps a single short line without an ellipsis", () => {
    expect(preview("**Done**").endsWith("Done")).toBe(true);
  });
});
