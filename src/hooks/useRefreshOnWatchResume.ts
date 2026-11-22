import { useEffect, useRef } from "react";
import { useAppStore } from "@/store/useAppStore";

/**
 * Bring the open session up to date when file watching resumes (#828).
 *
 * With desktop live refresh off (#820) the backend runs no watcher, so a
 * session file written meanwhile sends no event. When watching resumes, the
 * open session and its project's session list are refreshed once, through
 * the same path a watcher event takes (`triggerSessionRefresh`: its quiet
 * delay, and its deferral while the reader is scrolled up). Each window runs
 * this for its own selection, since the setting is shared.
 */
export function useRefreshOnWatchResume(watching: boolean): void {
  const wasWatching = useRef(watching);

  useEffect(() => {
    const resumed = watching && !wasWatching.current;
    wasWatching.current = watching;
    if (!resumed) return;
    const { selectedProject, selectedSession, triggerSessionRefresh } = useAppStore.getState();
    if (!selectedProject || !selectedSession) return;
    void triggerSessionRefresh(selectedProject.path, selectedSession.file_path);
  }, [watching]);
}
