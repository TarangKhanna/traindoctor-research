# Public evidence privacy

Personal attachments, resumes and interview material do not belong in this repository.
The ignore rules cover local attachment directories and resume/CV document names;
review staged files before publishing, since Git can force-add ignored files.
Training checkpoint resume code remains public.

The 2026-10-30 audit scanned both repositories’ advertised public refs and reachable
blob contents, including ZIP/tar members, for personal document names and known
resume contact fields. No such document or contact match was found in file contents. Commit author/committer
metadata does contain a personal email; that historical metadata remains public. This is a
bounded scan, not proof that every possible secret or personal datum is absent.
Author attribution, GitHub identity and license notices remain public.

Older research execution receipts contained owner-home directory paths. Current
receipts replace that prefix with `<LOCAL_HOME>`. These are redacted derivatives
of prior executions, not new experiments or byte-identical originals.
[The manifest](PRIVACY_REDACTION_20261030.json) records original and redacted SHA-256
values, including changed archive members. Unchanged archive member contents,
including tensors and submitted repair source, retain their original bytes.
File-checksum sidecars describe the redacted files; embedded historical tool-output
hashes retain their original-input meaning. No experimental metric or verdict was changed.

Older commits and other published branches still contain historical host paths.
A normal cleanup commit does not erase that history. Removing those paths from
all refs would require a separately approved history rewrite, and cannot recall
existing clones or cached copies. The separately supplied resume stays outside
both public repositories in a local ignored attachment directory.
