# Paper

- `draft.md` — the canonical content draft (Markdown, easy to diff/edit).
- `main.tex` — self-contained LaTeX port of `draft.md`. Compiles as-is.
- `figs/` — benchmark figures (copied from `results/benchmark/`).

## Compile (no local TeX needed for review)

`main.tex` uses only standard packages and a manual `thebibliography`
(no bibtex pass). Any of:

- **Overleaf**: upload `paper/` (incl. `figs/`), compile `main.tex`.
- **Local**: `cd paper && pdflatex main.tex && pdflatex main.tex`
  (twice for refs). Needs a TeX install (not present on the dev box).

## Target venue: NeurIPS ICBINB

ICBINB ("I Can't Believe It's Not Better") is the natural home for an
honest negative-results + benchmark paper. ICBINB uses the **NeurIPS
style file**. To produce the camera-ready:

1. Download the official `neurips_<year>.sty` (+ `.bst` if using bibtex)
   from the ICBINB / NeurIPS site.
2. In `main.tex`, replace
   `\documentclass[11pt]{article}` + the `geometry` line with
   `\documentclass{article}\usepackage{neurips_<year>}` (or the
   ICBINB-provided wrapper) per the template's instructions.
3. The body (sections, table, figures, `thebibliography`) is unchanged.
   Optionally convert `thebibliography` to `\bibliography` + the
   provided `.bst` if the template prefers bibtex.
4. Add the author block / anonymization per the venue's review rules
   (ICBINB is typically double-blind for review — use the anonymous
   variant of the style for submission, deanonymized for camera-ready).

Keep `draft.md` as the source of truth for content edits; re-port
prose changes into `main.tex` (they are structurally 1:1).

## Status

Content complete and honesty-hedged (see `docs/strategy.md`,
`docs/research_log.md`). Remaining before submission: venue style swap
(above), final author/affiliation + anonymization, and the optional
robustness items noted in the Limitations section.
