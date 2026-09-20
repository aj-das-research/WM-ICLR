# Personal paper visual skill availability review

Checked 2026-09-20T17:20:28.186183+00:00 using the installed `skill-creator` workflow.

All three skills were already installed under the personal Codex skill root `/scratch/abhijit.das/codex/skills` (`CODEX_HOME=/scratch/abhijit.das/codex`), so no repository was reinstalled. They are available across projects using this Codex installation. This check does not imply installation on other servers or user accounts.

## Narrow changes

- Expanded only the `paper-visual-design` discovery description to name method/algorithm figures, quantitative plots, qualitative comparisons, and figure improvements explicitly alongside architecture, conceptual illustrations and graphical abstracts.
- Preserved the complete Markdown body of all three skills byte for byte, including all nine personal principles and all companion customizations.
- Fixed a pre-existing `codex-paper-figure-skill` validator error by moving its existing author and version values under supported YAML `metadata`. The author, version and workflow are unchanged.
- Preserved existing UI metadata and implicit invocation policies. The personal skill explicitly allows implicit invocation; both companions retain the default enabled behavior.

## Validation

| Skill | Creator validation | Instructions preserved | Automatic invocation |
|---|---|---|---|
| `paper-visual-design` | Passed | Body byte-identical | Enabled |
| `paper-figure-creation` | Passed | Body byte-identical | Enabled |
| `codex-paper-figure-skill` | Passed | Body byte-identical | Enabled |

Both relative companion links in `paper-visual-design/SKILL.md` resolve. All UI default prompts invoke the correct explicit skill name and short descriptions satisfy the metadata length constraint. Creator validation checks structure; it is not proof of aesthetic quality or future model behavior. No new scripts, dependencies or publication permissions were introduced.

## Save location and explicit invocation

`/scratch/abhijit.das/codex/skills/paper-visual-design/SKILL.md`

Example: `Use $paper-visual-design to improve Figures 1 and 2 from this manuscript and code, preserving scientific meaning and checking the rendered pages.`

Companions remain available as `$paper-figure-creation` and `$codex-paper-figure-skill`.

## File hashes

- `paper-visual-design/SKILL.md`
  - Before: `f4ae7a322a3d70dac34ac6f96c3d8a6bceb2427ba677fe0a544213620dc69a64`
  - After: `5aa78775835d96f7a53973591defadf80494a4858014b8016affd078cbe49e64`
- `paper-figure-creation/SKILL.md`
  - Before: `31ffa6f46188c364359a0d9232c3902a66456a90aef3d4fd0738b216d46f2b9b`
  - After: `31ffa6f46188c364359a0d9232c3902a66456a90aef3d4fd0738b216d46f2b9b`
- `codex-paper-figure-skill/SKILL.md`
  - Before: `e7cefd7bae7151952a19fc3296ad845e70012e4caee83c7961c7944a64ee9df6`
  - After: `e9e90000e647647729573a08ebd88e1e2064497c9a23eb23a5aaa313e32423a6`
