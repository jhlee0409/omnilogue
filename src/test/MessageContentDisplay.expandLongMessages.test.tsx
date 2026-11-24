import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ReactNode } from "react";

vi.mock("react-i18next", () => ({
  initReactI18next: { type: "3rdParty", init: () => {} },
  useTranslation: () => ({ t: (key: string) => key, i18n: { language: "en" } }),
}));

import { MessageContentDisplay } from "../components/messageRenderer/MessageContentDisplay";
import { ExpandKeyProvider } from "@/contexts/CaptureExpandContext";
import { useExpandRegistry } from "@/store/expandRegistryStore";
import { useAppStore } from "@/store/useAppStore";

/**
 * "Expand long messages by default" (#574): an opt-in, persisted view
 * setting that only changes the starting state of a long message's text.
 * A message's own Show more / Show less choice wins, and the other
 * collapsibles (here a long markdown table) keep their own default.
 */

const LONG_TEXT = Array.from({ length: 20 }, (_, i) => `line ${i + 1}`).join("\n");
const LONG_TABLE = ["| a |", "| - |", "| 1 |", "| 2 |", "| 3 |", "| 4 |"].join("\n");

const wrapper = ({ children }: { children: ReactNode }) => <ExpandKeyProvider value="m-1">{children}</ExpandKeyProvider>;
const textToggle = () => screen.getByRole("button", { name: /messageContentDisplay\.(showMore|showLess)/ });
const setPreference = (expand: boolean) => act(() => useAppStore.setState({ expandLongMessages: expand }));

beforeEach(() => {
  localStorage.clear();
  useExpandRegistry.getState().clearAll();
  useAppStore.setState({ expandLongMessages: false });
});

afterEach(cleanup);

describe("expand long messages setting", () => {
  it("is off by default, and the toggle persists across a reload of settings.json", async () => {
    await useAppStore.getState().loadUpdateSettings();
    expect(useAppStore.getState().expandLongMessages).toBe(false);

    await act(async () => useAppStore.getState().toggleExpandLongMessages());
    await vi.waitFor(() => expect(localStorage.getItem("webui:settings.json:expandLongMessages")).toBe("true"));
    useAppStore.setState({ expandLongMessages: false });
    await useAppStore.getState().loadUpdateSettings();
    expect(useAppStore.getState().expandLongMessages).toBe(true);
  });
});

describe("long message text", () => {
  it("starts collapsed while the setting is off and expanded once it is on", () => {
    render(<MessageContentDisplay content={LONG_TEXT} messageType="user" />, { wrapper });
    expect(textToggle()).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByText(/line 20/)).toBeNull();

    setPreference(true);
    expect(textToggle()).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText(/line 20/)).toBeInTheDocument();
  });

  it("keeps a message's own choice when the setting changes", () => {
    setPreference(true);
    render(<MessageContentDisplay content={LONG_TEXT} messageType="user" />, { wrapper });
    fireEvent.click(textToggle()); // Show less on this message
    expect(textToggle()).toHaveAttribute("aria-expanded", "false");

    setPreference(false);
    setPreference(true);
    expect(textToggle()).toHaveAttribute("aria-expanded", "false");
  });

  it("does not expand other collapsibles such as a long table", () => {
    setPreference(true);
    render(<MessageContentDisplay content={LONG_TABLE} messageType="assistant" />, { wrapper });
    expect(screen.getByRole("button", { name: /messageContentDisplay\.showMoreRows/ })).toHaveAttribute(
      "aria-expanded",
      "false",
    );
  });
});
