# Interface design

The web interface applies the apple-design skill through readable hierarchy,
predictable navigation, immediate feedback, and restrained materials. It keeps
Streamlit's native controls, keyboard behavior, sidebar collapse, and routing.
The simulation, models, governance gates, and committed results are unchanged.

- `app/styles.css` owns system typography, spacing, surfaces, control feedback,
  and accessibility media queries. Styles use Streamlit test IDs rather than
  generated CSS class names. Recheck these hooks when upgrading Streamlit.
- `app/common.py` owns grouped navigation and semantic colors. Blue identifies
  navigation/current defense; red, green, and amber communicate attack, success,
  and caution. Status badges always include words, not just color.
- The default is graphite with solid content panels. Only chrome uses blur.
  Surfaces derive from Streamlit theme variables. No external font, script,
  animation library, or network request is added.
- Press feedback is immediate. There are no new drag gestures, artificial waits,
  input locks, or decorative entrance animations. Reduced motion removes press
  scaling, reduced transparency removes blur, and higher contrast strengthens
  surface boundaries. Native widget behavior remains Streamlit's responsibility.
- The Atlas exposes search and common filters first. Coverage and specialist
  filters remain available through named disclosures. Search combines all words
  across names, categories, objectives, mechanisms, and observed signals, alongside
  the selected filters. Clear filters restores the full catalog and default sort.

## Verification

Run `python -m pytest tests/test_app_navigation.py -q` for offline page rendering,
combined search/filter behavior, empty-result feedback, and reset recovery.

For visual changes, check the Atlas, Overview, and Judge Demo at a desktop width
and at 390px: navigation collapse/reopen, heading and metric wrapping, filter
labels, chart containment, keyboard focus, and empty-result recovery. Inspect
reduced-motion, reduced-transparency, and higher-contrast media rules when adding
new effects. Treat these checks as UI verification, not validation of the fraud
model against real payment data.
