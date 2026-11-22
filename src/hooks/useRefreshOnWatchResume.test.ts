/**
 * Turning live refresh back on (#828) refreshes the open session once,
 * through the watcher's own refresh, and only on that transition.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderHook } from "@testing-library/react";
import type { ClaudeProject, ClaudeSession } from "@/types";
import { useAppStore } from "@/store/useAppStore";
import { useRefreshOnWatchResume } from "./useRefreshOnWatchResume";

const PROJECT = { path: "/p/alpha" } as unknown as ClaudeProject;
const SESSION = { file_path: "/p/alpha/s1.jsonl" } as unknown as ClaudeSession;

const pristine = useAppStore.getState();
const triggerSessionRefresh = vi.fn(async () => {});

function renderWatching(watching: boolean) {
  return renderHook(({ on }) => useRefreshOnWatchResume(on), {
    initialProps: { on: watching },
  });
}

beforeEach(() => {
  useAppStore.setState(pristine, true);
  useAppStore.setState({
    selectedProject: PROJECT,
    selectedSession: SESSION,
    triggerSessionRefresh,
  });
  triggerSessionRefresh.mockClear();
});

describe("useRefreshOnWatchResume", () => {
  it("refreshes the open session when watching resumes", () => {
    const { rerender } = renderWatching(false);
    expect(triggerSessionRefresh).not.toHaveBeenCalled();

    rerender({ on: true });
    expect(triggerSessionRefresh).toHaveBeenCalledTimes(1);
    expect(triggerSessionRefresh).toHaveBeenCalledWith(PROJECT.path, SESSION.file_path);

    rerender({ on: true });
    expect(triggerSessionRefresh).toHaveBeenCalledTimes(1);
  });

  it("does nothing at launch or when watching stops", () => {
    const { rerender } = renderWatching(true);
    rerender({ on: false });
    expect(triggerSessionRefresh).not.toHaveBeenCalled();
  });

  it("does nothing without an open session", () => {
    useAppStore.setState({ selectedSession: null });
    const { rerender } = renderWatching(false);
    rerender({ on: true });
    expect(triggerSessionRefresh).not.toHaveBeenCalled();
  });
});
