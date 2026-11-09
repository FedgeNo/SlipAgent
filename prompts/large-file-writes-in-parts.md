# Writing Large Files in Parts

NVIDIA's free hosted endpoints can send a tool call only after generating all
of it, and the connection can close while a very long call is still being
generated. The call is then lost and nothing is written. The largest successful
`write_file` call measured on one of them was about 29,500 characters, so keep
each call's new text under about **22,000 characters**.

Size estimates are unreliable, so judge size by structure. Split when the text
would exceed any of these:

- roughly 400 lines of ordinary code, or 250 lines of long-line code or tables;
- about 3,000 words of prose, such as a report or Markdown document;
- a dozen substantial functions, or several report sections.

Build a large file across several calls, in order:

1. `write_file` the first part. End it with a marker line that occurs nowhere
   else in the file, such as `SLIPAGENT-CONTINUE-checker-part-2`.
2. For each next part, call `edit_file` with that exact marker line as
   `old_string`, and the next part followed by the next marker
   (`SLIPAGENT-CONTINUE-checker-part-3`) as `new_string`.
3. For the last part, replace the marker with the final text and no new marker.
   Read the end of the file to confirm that no marker remains.

Split at natural boundaries, such as between functions or sections, so each part
is whole. The final file is exactly the concatenation of the parts. If a call
fails, read the end of the file and continue from the marker that is there;
do not rewrite parts that are already written. Smaller edits to existing files
still use `edit_file` or `edit_hashlines` as usual.
