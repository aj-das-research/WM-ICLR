# Closest conceptual antecedents

Primary sources checked 22 September 2026:

- Finn, Goodfellow, Levine (2016), https://arxiv.org/abs/1605.07157 and
  https://proceedings.neurips.cc/paper/2016/file/d9d4f495e875a2e075a1a4a6e1b9770f-Paper.pdf.
  Section 3 describes action-conditioned transformations and masked composition
  of visual content from prior frames (DNA/CDNA).
- Ebert, Finn, Lee, Levine (2017), https://arxiv.org/html/1710.05268v1,
  Section 4, equations 1–2. SNA explicitly retains the first real image through
  a skip connection, combined with transformed recursive predictions.

Neither was cited in the active manuscript before this audit. Their existence
materially narrows the novelty claim. Added citations and a main-text comparison:
fixed measured memory is not new; the specific pooled pretrained feature-grid
mixing/correction parameterization and its empirical study are this paper's scope.
No numerical comparison or reproduction of those pixel models was performed.
