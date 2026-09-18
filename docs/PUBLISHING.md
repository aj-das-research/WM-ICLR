# GitHub, Overleaf and project-page synchronization

This project's server runs a Git bridge between the research workspace,
GitHub `main`, and the manuscript's Overleaf Git repository. GitHub Pages is
built from `gh-pages`. This uses Overleaf's Git interface, not its separate
GitHub integration button; avoid enabling a second competing synchronization
workflow on the same branches.

Edit manuscript sources on Overleaf, on GitHub under `paper/`, or in the local
workspace. The next cycle fetches both histories and merges independent text
changes into the workspace. It rebuilds the paper, independently compiles the
Overleaf upload and compares PDF text, checks the public source snapshot for
secrets, and publishes the paper, code and static demo. Concurrent edits to the
same text, incompatible binary edits and deletion conflicts stop publication
and preserve the versions for resolution. Pushes are never forced.

The server timer checks every five minutes and waits for 30 seconds without
changes before publishing local work. The paper compiler and GitHub Pages can
add deployment time. The timer runs while this server's user service manager
is active; it cannot synchronize while the server or its user manager is down.
The working repository retains its original local history. A separate curated
checkout provides the public history, excluding datasets, credentials, caches
and ordinary-Git checkpoint binaries.

## Commands on the configured publishing server

```bash
python scripts/publishing/sync_project.py
systemctl --user list-timers shiftwm-sync.timer
journalctl --user -u shiftwm-sync.service -n 40 --no-pager
systemctl --user stop shiftwm-sync.timer
systemctl --user start shiftwm-sync.timer
```

The latest verified commit IDs and import details are in
`~/.local/share/shiftwm/sync/receipt.json`. A stopped sync writes a useful reason
to the journal; merge conflicts also create `conflicts.json` there. Original
workspace versions are backed up in the private `imports/` directory. Base
versions are content-addressed in `blobs/` and the relevant remote versions
remain in the dedicated GitHub and Overleaf checkouts.

To resolve a conflict, reconcile the local source with the named remote file,
keeping the desired shared version, then rerun the command. Identical local
and remote versions resolve directly. For a genuinely different combined
version, commit that resolution to the appropriate remote checkout as well,
then rerun; do not discard the private merge bases. A failed concurrent push
preserves our outgoing commit under a `sync-backup/` branch before returning
the publisher checkout to the fetched remote tree and retrying reconciliation.

## Source and generated-file boundaries

- Edit `site/index.html`, `site/styles.css` and `site/app.js` on `main` for the
  project page. `gh-pages` is generated; unexpected remote edits to that branch
  stop publication until they have been ported to the source and reconciled.
  After porting them, run `sync_project.py --acknowledge-pages-commit FULL_SHA`
  with that exact remote commit; a newer remote edit will still stop the sync.
- Edit LaTeX, bibliography and source diagrams rather than generated PDFs or
  regenerated site assets. Remote edits to known regenerated artifacts stop
  instead of disappearing during a rebuild. New paper images can be added on
  Overleaf and are imported into `paper/` with their paths intact.
- Registered real-video training sources/configurations and the official
  conference template are protected against automatic remote modification.
  Experimental changes belong in a new versioned namespace so completed
  results retain their original source contract.
- Datasets and neural weights do not enter ordinary Git. Downloadable model
  packages are separate GitHub Release assets with model cards and checksums.
- GitHub and Overleaf publication are sequential, not an atomic transaction.
  After a transient failure, successful individual pushes are recorded and
  the next cycle resumes. An unresolved conflict intentionally pauses updates.

## Install on another authorized server

Create clean publisher checkouts for GitHub `main`, GitHub `gh-pages`, and
Overleaf `main`. Configure a Git author in each. Save authentication with a
credential manager or separate Git credential-store files outside the project
(directory mode `0700`, files `0600`). Do not paste tokens into repository
URLs, source files, issue bodies or this document.

The private `~/.config/shiftwm/publishing.json` contains these keys:

```json
{
  "github_repository": "OWNER/REPOSITORY",
  "github_checkout": "/absolute/path/to/public-main",
  "pages_checkout": "/absolute/path/to/public-pages",
  "github_credential_file": "/private/path/github.credentials",
  "overleaf_git": "https://git.overleaf.com/PROJECT_ID",
  "overleaf_checkout": "/absolute/path/to/overleaf-checkout",
  "credential_file": "/private/path/overleaf.credentials"
}
```

Initialize once from the clean **last published** checkouts, before fetching
new remote edits, so those versions become the merge bases:

```bash
python scripts/publishing/sync_project.py --initialize
python scripts/publishing/sync_project.py
python scripts/publishing/install_sync_timer.py
```

The server also needs Git, Python, TeX Live, `pdftotext`, `pdftocairo`,
`pdftoppm`, FFmpeg and the project's local research artifacts used by
`site/publish.py`. Credentials never appear in the public snapshot. The static
Pages branch can also be built from committed site assets without datasets:
`python site/publish.py --snapshot-only`.
