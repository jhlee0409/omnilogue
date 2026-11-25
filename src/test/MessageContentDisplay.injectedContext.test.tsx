import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ReactNode } from "react";

vi.mock("react-i18next", () => ({
  initReactI18next: { type: "3rdParty", init: () => {} },
  useTranslation: () => ({ t: (key: string) => key, i18n: { language: "en" } }),
}));

import { MessageContentDisplay } from "../components/messageRenderer/MessageContentDisplay";
import { ExpandKeyProvider } from "@/contexts/CaptureExpandContext";
import { useExpandRegistry } from "@/store/expandRegistryStore";

/**
 * Context the client injects into a user turn (Codex `<environment_context>`)
 * is a collapsed card, not a user speech bubble; text the user typed next to
 * it, or a search, keeps the bubble.
 */

const ENV = "<environment_context>\n  <cwd>/work/app</cwd>\n  <shell>zsh</shell>\n</environment_context>";
const wrapper = ({ children }: { children: ReactNode }) => <ExpandKeyProvider value="m-ctx">{children}</ExpandKeyProvider>;

beforeEach(() => {
  localStorage.clear();
  useExpandRegistry.getState().clearAll();
});

afterEach(cleanup);

describe("injected context in a user turn", () => {
  it("is a collapsed card that opens to the context", () => {
    render(<MessageContentDisplay content={ENV} messageType="user" />, { wrapper });
    const header = screen.getByRole("button", { name: /messageContentDisplay\.injectedContext/ });
    expect(screen.queryByText(/\/work\/app/)).toBeNull();
    fireEvent.click(header);
    expect(screen.getByText(/\/work\/app/)).toBeTruthy();
  });

  it("keeps the bubble for typed text and while searching", () => {
    render(<MessageContentDisplay content={`${ENV}\nfix the bug`} messageType="user" />, { wrapper });
    expect(screen.getByText(/fix the bug/)).toBeTruthy();
    cleanup();
    render(<MessageContentDisplay content={ENV} messageType="user" searchQuery="zsh" />, { wrapper });
    expect(screen.queryByRole("button", { name: /messageContentDisplay\.injectedContext/ })).toBeNull();
  });

  it("treats a lone local-command caveat as injected context, and as a plain bubble while searching", () => {
    const caveat = "<local-command-caveat>Caveat: generated while running local commands.</local-command-caveat>";
    render(<MessageContentDisplay content={caveat} messageType="user" />, { wrapper });
    expect(screen.getByRole("button", { name: /messageContentDisplay\.injectedContext/ })).toBeTruthy();
    cleanup();
    render(<MessageContentDisplay content={caveat} messageType="user" searchQuery="caveat" />, { wrapper });
    expect(screen.getByText(/generated while running/)).toBeTruthy();
    expect(screen.queryByRole("button", { name: /messageContentDisplay\.injectedContext/ })).toBeNull();
  });
});
