/**
 * Shared remark/rehype plugin configuration for every ReactMarkdown instance
 * in the app (GFM + LaTeX math via KaTeX + CJK-friendly emphasis). Import
 * this instead of constructing a local plugin array so the renderers can't
 * drift apart when a plugin or option changes.
 */

import remarkCjkFriendly from "remark-cjk-friendly";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";

/**
 * Module-level plugin arrays for stable reference across renders.
 * `remarkCjkFriendly`: CommonMark leaves `**"표현"**이` or `**`code`**를`
 * unbolded (a closing `**` after punctuation must be followed by a space or
 * punctuation); Korean, Japanese and Chinese text attaches particles
 * directly, so assistants' replies showed literal `**`.
 */
export const REMARK_PLUGINS = [remarkGfm, remarkCjkFriendly, remarkMath];
export const REHYPE_PLUGINS = [rehypeKatex];
