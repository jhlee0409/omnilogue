/**
 * Bold and italic next to Korean, Japanese and Chinese text: CommonMark does
 * not close `**…**` after punctuation when a particle follows directly, so
 * replies showed literal `**` (remark-cjk-friendly).
 */
import { describe, expect, it } from "vitest";
import { render } from "@testing-library/react";
import { Markdown } from "./Markdown";

const strongTexts = (markdown: string) =>
  [...render(<Markdown>{markdown}</Markdown>).container.querySelectorAll("strong")].map(
    (node) => node.textContent
  );

describe("Markdown emphasis next to CJK text", () => {
  it.each([
    ['**"표현"**이 안 됩니다', '"표현"'],
    ["**`git restore .`**으로 되돌립니다", "git restore ."],
    ["**「設定」**を開く", "「設定」"],
    ["**（配置）**文件", "（配置）"],
  ])("bolds %s", (markdown, bold) => {
    expect(strongTexts(markdown)).toEqual([bold]);
  });

  it("keeps Latin emphasis rules", () => {
    expect(strongTexts("**bold** text and 2**3**4")).toEqual(["bold", "3"]);
    expect(strongTexts("a ** b ** c")).toEqual([]);
  });
});
