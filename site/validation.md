# Local project page validation

Validated on 2026-09-18. This validates the page and exported source entry points, not an empirical model advantage or a fresh full training campaign.

## Functional checks

- Actual local Firefox 152 / Mozilla GeckoDriver 0.36 browser execution. Browser/IAB tools were unavailable; Chromium was not installed, so the already available system Firefox was used without adding browser dependencies to training.
- Desktop viewport 1440×1014: actual SVG loaded, status JSON rendered, no status error, navigation and status refresh worked, the protocol disclosure opened, and command copying produced success feedback.
- A same-origin 390px iframe supplied a true390px responsive layout viewport despite Firefox's 500px minimum outer window. Content width 378px; no page-level horizontal overflow.
- All 9 unique local HTTP artifact/asset links returned 200. All in-page anchors resolve. The standalone export has the referenced assets, PDF, source ZIP, protocol and status files without depending on parent-directory URLs.
- Above-the-fold visible copy was compared against the design brief and matched exactly.
- Source ZIP integrity and required lockfile/license/provenance files were checked after extraction. All Python source parses; the setup shell passes `bash -n`; the downloader imports and validates its artifact manifest. `bootstrap_sources.py --check` and `download_artifacts.py --verify-only` report the expected absent external checkouts/weights without network downloads. A fresh dependency installation and full retraining were not performed by this webpage validation.

Machine-readable evidence is in `validation/browser.json`, `validation/links.json`, and `validation/source_archive.json`. Final review screenshots in that directory are local QA evidence and are excluded from the standalone export. Logs and intermediate extracted archives are excluded/removed.

## Design fidelity review

Built-in Image Gen supplied the three coordinated concept references in `design/`, following the applied frontend app builder skill. `view_image` was used on all three concepts and their corresponding browser screenshots in the same QA pass, plus the narrow rendering.

| Comparison | Evidence and disposition |
|---|---|
| Copy and hierarchy | The exact hero, navigation, action labels and pending-results statement are preserved. No author, acceptance, score or public project URL was added. |
| Layout and rhythm | Centered opening; open method columns; a split matrix with protocol facts; ruled progress rows; two-column reproduction section. Thin rules and whitespace match the concept family. |
| Typography | System sans with 76px desktop headline, 32px section headings, 24px context titles and explicit control typography; no external font/CDN dependency. Browser headline size verified. |
| Palette | White background, ink text, muted gray, blue observation path and rust dynamics path. No gradients, shadows, overlays or decorative badges. |
| Scientific asset | The actual method SVG is copied byte-for-byte. Its complete content and aspect ratio take precedence over the generative mockup's compressed version; it can be opened independently. This is an intentional semantic-fidelity adjustment. |
| Status and downloads | Generated placeholder status text is replaced by real timestamped completion counts. Separate artifact snapshot time, required protocol disclosure, refresh feedback, and working source/demo links are intentional functional additions. |
| Responsive behavior | Columns stack; navigation remains usable; the matrix fits; the real vector remains available for magnified inspection; no 390px overflow. |

The implementation was visually verified against the selected design system and section concepts, with the source-driven adjustments above. There are no remaining observed clipping or broken-control issues. Exact raster pixel identity is not claimed: the generated references have slightly different canvas dimensions and the production method figure is deliberately unmodified.

## Runtime and export

Default local address: `http://127.0.0.1:8008`. PID is recorded in `server.pid`. The server refreshes status and exported artifacts every 60 seconds, publishes downloadable files atomically, and does not alter GPU jobs or the paper watcher. The page shows status and artifact snapshot times separately. Exported standalone bundles remain snapshots until rebuilt. No external publication was performed.
