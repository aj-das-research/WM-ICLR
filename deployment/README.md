# Project page deployment

The static project page is published from the `gh-pages` branch root.
`main` contains source, research artifacts and `site/publish.py`; `gh-pages`
contains the 16 verified exported files and `.nojekyll`.

Regenerate the source-backed page with `python site/publish.py`. For a clean
source checkout, use `python site/publish.py --snapshot-only --output /tmp/wm-pages`.
Keep the deployment checkout separate from research data and commit/push only
that verified export. Repository Pages source is `gh-pages`, path `/`.

`github-pages.yml` is an inactive alternative workflow template for accounts
with workflow-management permission. No credentials belong in this directory.
The synchronization implementation stores credentials outside the repository
and updates branches by fast-forward; conflicting remote edits require merging.
