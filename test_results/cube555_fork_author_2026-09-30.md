# Cube555 fork attribution — 2026-09-30

Confirmed defect: first-cell KAGGLE_OWNER/SLUG and publication kaggle_username could identify a fork owner while tools/cube555/run.py fixed author.name to Ivan Litvak. New configuration derives author from explicit publication author_name, else kaggle_username, else kaggle_owner; absent publication preserves historical local fallback. Caller metadata is copied and unchanged.

106 Python contract/retry/results tests PASS, including actual publication-context authors for another owner, owner-only metadata, and display-name override. Old archives/retries are not rebuilt. No native, neural scoring, model, beam, depth, collection, history or reflection changes. Notebook generatorversion14 prepared; actual public QuickSave and pin verification stillpending.

Current hardware gate remains sole longPID1020 jobRUNNING; fresh logsdepth32/fullfrontierbothranks,379.601sec/GPU4965/4985MiB. Current originalGitblob matchesmain1050/staging1410,0mismatch; all2000promotion pending.
