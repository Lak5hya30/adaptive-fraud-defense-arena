# Interface design

The interface uses one light appearance, readable hierarchy, predictable
navigation, and immediate feedback. It keeps
Streamlit's native controls, keyboard behavior, sidebar collapse, and routing.
The simulation, models, governance gates, and committed results are unchanged.

- `app/styles.css` owns system typography, spacing, surfaces, control feedback,
  and accessibility media queries. Styles use Streamlit test IDs rather than
  generated CSS class names. Recheck these hooks when upgrading Streamlit.
- `app/common.py` owns grouped navigation. `app/ui.py` supplies reusable
  `page_header`, responsive `columns`, `Metric` / `metric_grid`, `card`, `badge`,
  `progress_steps`, and `plot` components. All pages use shared headers and rows;
  all Plotly charts use the shared renderer. Blue identifies
  navigation/current defense; red, green, and amber communicate attack, success,
  and caution. Status badges always include words, not just color.
- The app is light-only: a pale canvas, white cards, and opaque navigation.
  `.streamlit/config.toml` sets a light theme and a minimal toolbar, removing
  the viewer theme selector. Plotly figures explicitly use `plotly_white` and
  `theme=None`, so chart surfaces do not follow an alternative viewer theme.
  No external font, script, animation library, or network request is added.
- Rows respond to their actual available width, including nested columns and
  the space left beside the sidebar. Metric grids align label/value/note rows
  using CSS subgrid with a flex fallback. Native metrics use content height,
  wrap long values, and never extend over following captions. Navigation link
  wrappers reset Streamlit's negative margins so their hit areas do not overlap.
- Wide tables, code, and tab strips keep their own scrolling rather than
  overflowing the page. On narrow displays, the native sidebar can be opened
  as a navigation drawer and dismissed with its always-visible collapse control.
- Press feedback is immediate. There are no new drag gestures, artificial waits,
  input locks, or decorative entrance animations. Reduced motion removes press
  scaling, surfaces are already opaque, and higher contrast strengthens
  surface boundaries. Native widget behavior remains Streamlit's responsibility.
- The Atlas exposes search and common filters first. Coverage and specialist
  filters remain available through named disclosures. Search combines all words
  across names, categories, objectives, mechanisms, and observed signals, alongside
  the selected filters. Clear filters restores the full catalog and default sort.

## Verification

Run `python -m pytest tests/test_app_navigation.py tests/test_artifact_consistency.py -q`
for offline page rendering, combined search/filter behavior, reset recovery,
all eight Judge Demo stages (including Back and Restart), all six Unseen Attack
Demo beats, and consistency of the committed results.

Check every page at 1440px, 820px, and 390px. Measure sibling component bounds,
metric-content containment, and horizontal page overflow; also inspect screenshots
and open each tab and demo state. Test with the narrow navigation drawer dismissed
for content checks, and separately check opening/closing it. Inspect keyboard focus
and reduced-motion/higher-contrast preferences when adding effects. These are UI
checks, not validation of the fraud model against real payment data.
